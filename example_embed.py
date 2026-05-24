from promoter_protein_clip.model_transformer import SequenceCLIPTransformer
import torch
from huggingface_hub import hf_hub_download

repo_id = "dufaultc/contrastive-promoter-protein-pretraining"
filename = "C3P_100M.pth"

promoter = "ACTCGCTATGCGCTATATCTCTATACGGTTGCCGCGCGGTGTCCCCCGGTAAACTCGCTATGCGCTATATCTCTATACGGTTGCCGCGCGGTGTCCCCCGGTAAACTCGCTATGCGCTATATCTCTATACGGTTGCCGCGCGGTGTCCCCCGGTAA"
checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
checkpoint= torch.load(checkpoint_path, map_location=device, weights_only=False)    
model = SequenceCLIPTransformer(
    protein_model_name=checkpoint['args'].protein_model,
    promoter_embedding_dim=checkpoint['args'].promoter_embedding_dim,
    depth=checkpoint['args'].transformer_depth,
    num_heads=checkpoint['args'].transformer_num_heads,
    max_promoter_length=checkpoint['args'].max_promoter_length,
    projection_dim=checkpoint['args'].projection_dim,       
 ).to(device)
model = torch.compile(model)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
batch_inputs = model.promoter_tokenizer(
    [promoter],
    padding=True,
    truncation=True,
    return_tensors="pt",
)
with torch.no_grad():
    embeddings = (
        model.promoter_encoder(batch_inputs.to(device)).cpu().numpy()
    )   
print(embeddings)