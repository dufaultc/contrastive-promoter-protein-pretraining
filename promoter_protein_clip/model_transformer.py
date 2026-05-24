import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer

from promoter_protein_clip.loss import CLIPLoss


class KmerTokenizer:
    def __init__(self, k=3, stride=1, max_length=512):
        self.k = k
        self.stride = stride
        self.max_length = max_length
        self.vocab = self._build_vocab()
        self.pad_token_id = self.vocab['<PAD>']
        self.unk_token_id = self.vocab['<UNK>']
        self.cls_token_id = self.vocab['<CLS>']

    def _build_vocab(self):
        # Generate all 4^k combinations
        bases = ['A', 'C', 'G', 'T', "N"]
        kmers = [''.join(i) for i in itertools.product(bases, repeat=self.k)]

        vocab = {kmer: i + 3 for i, kmer in enumerate(kmers)}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        vocab['<CLS>'] = 2
        return vocab

    def __call__(self, text_batch, padding=True, truncation=True, return_tensors=None):
        tokenized_batch = []
        for text in text_batch:
            text = text.upper()
            tokens = [self.cls_token_id]
            for i in range(0, len(text) - self.k + 1, self.stride):
                tokens.append(self.vocab.get(text[i : i + self.k], self.unk_token_id))

            if truncation and len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]

            if padding and len(tokens) < self.max_length:
                tokens += [self.pad_token_id] * (self.max_length - len(tokens))

            tokenized_batch.append(tokens)

        if return_tensors == 'pt':            
            return torch.tensor(tokenized_batch, dtype=torch.long)
        
        return tokenized_batch

    def get_vocab_size(self):
        return len(self.vocab)


### RotaryPE copied from https://www.kaggle.com/code/aeryss/rotary-postional-encoding-rope-pytorch
class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]
        return self.cos_cached, self.sin_cached

# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1) # dim=-1 triggers a bug in torch < 1.8.0

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
###


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm = nn.LayerNorm(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, rope_cos, rope_sin, mask=None):
        B, L, D = x.shape

        residual = x
        x = self.norm(x)

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)        
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)        

        attn_out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask,
        )        
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)        
        attn_out = self.out_proj(attn_out)        

        x = residual + attn_out        

        residual = x
        x = self.norm(x)
        x = self.ff(x)
        x = residual + x 

        return x


class PromoterTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, depth=4, num_heads=4, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rope = Rotary(embed_dim // num_heads)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim=embed_dim*4)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):        
        x_emb = self.embedding(x)        
        cos, sin = self.rope(x_emb)        
        attn_mask = (x != 0).unsqueeze(1).unsqueeze(2)        

        for layer in self.layers:
            x_emb = layer(x_emb, cos, sin, attn_mask)        
        x_emb = self.norm(x_emb)         
        return x_emb[:, 0, :]


class SequenceCLIPTransformer(nn.Module):
    def __init__(
        self, 
        protein_model_name="facebook/esm2_t6_8M_UR50D",
        promoter_embedding_dim=256,
        promoter_kmer_size=3,
        promoter_kmer_stride=1,
        projection_dim=256,
        protein_max_len=1024,
        max_promoter_length=512,
        depth = 6,
        num_heads = 8,
        device=None,
        train_embeds_dict =None,
        val_embeds_dict =None,
        unfreeze = False,
    ):
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.train_embeds_dict = train_embeds_dict
        
        # tokenizers
        self.protein_tokenizer = EsmTokenizer.from_pretrained(protein_model_name)
        self.protein_max_len = protein_max_len
        self.promoter_tokenizer = KmerTokenizer(
            k=promoter_kmer_size, 
            stride=promoter_kmer_stride, 
            max_length=max_promoter_length,
        )
        self.max_promoter_length = max_promoter_length

        # encoders
        self.protein_encoder = EsmModel.from_pretrained(protein_model_name)
        # freeze all ESM2 model parameters
        if not unfreeze:
            for param in self.protein_encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.protein_encoder.parameters():
                param.requires_grad = True
        self.protein_hidden_size = self.protein_encoder.config.hidden_size
        self.promoter_encoder = PromoterTransformer(
            vocab_size=self.promoter_tokenizer.get_vocab_size(),
            embed_dim=promoter_embedding_dim,
            depth=depth,
            num_heads=num_heads,
            max_len=max_promoter_length,
        )
        for param in self.promoter_encoder.parameters():
            param.requires_grad = True
        # projection heads
        self.protein_projection = nn.Linear(self.protein_hidden_size, projection_dim)        
        self.promoter_projection = nn.Linear(promoter_embedding_dim, projection_dim) 
        
        self.loss = CLIPLoss(device=device)

        self.to(self.device)

    def forward(self, protein_strings, promoter_strings):
        if self.train_embeds_dict is not None:
            protein_embedding = self.protein_projection(protein_strings)
        else:              
            protein_features = self.protein_encoder(**protein_strings)
            protein_embedding = self.protein_projection(protein_features.last_hidden_state[:, 0, :])

        promoter_features = self.promoter_encoder(promoter_strings)
        promoter_embedding = self.promoter_projection(promoter_features)

        protein_embedding = F.normalize(protein_embedding, p=2, dim=-1)
        promoter_embedding = F.normalize(promoter_embedding, p=2, dim=-1)

        loss = self.loss(protein_embedding, promoter_embedding)

        return {
            'protein_embedding': protein_embedding,
            'promoter_embedding': promoter_embedding,
            'loss': loss,
        }
