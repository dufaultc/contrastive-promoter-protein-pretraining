import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from huggingface_hub import hf_hub_download
from promoter_protein_clip.model_transformer import SequenceCLIPTransformer
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForMaskedLM

try:
    from evo import Evo, score_sequences
except ImportError:
    Evo = None

try:
    from evo2 import Evo2
except ImportError:
    Evo2 = None


parser = argparse.ArgumentParser(description="")
parser.add_argument('--model', type=str, choices=['c3p', 'promogen', 'evo', 'evo2', 'glm2'], default='c3p', help='Model to use')
parser.add_argument('--num_promoters', type=int, default=100, help='Number of promoters (with binding sites) to consider')
parser.add_argument('--promoter_len', type=int, default=100, help='How much of promoter to consider')
parser.add_argument('--model_file_name', type=str, default="C3P_100M.pth", help='Which C3P checkpoint to load')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size inference')
args = parser.parse_args()

promoter_len = args.promoter_len
batch_size = args.batch_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#################################################
# Scoring sequence pseudolikelihood, only needed for glm2 (very very very slow)
def score_pll(
    sequence,
    glm2_tokenizer,
    model,
    batch_size,
    device='cuda'
):
    tokens = []
    i = 0
    while i < len(sequence):
        if sequence[i:i+3] == '<+>':
            tokens.append('<+>')
            i += 3
        else:
            tokens.append(sequence[i])
            i += 1

    masked_seqs = []
    mask_indices = []
    target_ids = []

    for idx, tok in enumerate(tokens):
        if tok == '<+>':
            continue
        
        masked_tokens = tokens.copy()
        masked_tokens[idx] = '<mask>'
        masked_seqs.append("".join(masked_tokens))
        
        mask_indices.append(idx)
        target_ids.append(glm2_tokenizer.convert_tokens_to_ids(tok))
        
    likelihoods = []
    model.eval()
    
    with torch.inference_mode():
        for batch_index in range(0, len(masked_seqs), batch_size):
            batch_seqs = masked_seqs[batch_index : batch_index + batch_size]
            
            batch_mask_idx = torch.tensor(mask_indices[batch_index : batch_index + batch_size], device=device)
            batch_target_id = torch.tensor(target_ids[batch_index : batch_index + batch_size], device=device)
            
            yep = glm2_tokenizer(batch_seqs, return_tensors='pt', padding=True)            
            out = model(yep.input_ids.to(device))            
            log_probs = F.log_softmax(out['logits'], dim=-1)
            
            b_idx = torch.arange(len(batch_seqs), device=device)

            correct_token_likelihoods = log_probs[b_idx, batch_mask_idx, batch_target_id]
            
            likelihoods.extend(correct_token_likelihoods.cpu().tolist())
    return sum(likelihoods) / len(likelihoods)


##################################################
# Figuring out where the binding sites are in the promoters, storing them in a big binary mask
binding_sites_mask_positive = np.zeros(5000000, dtype=bool)
binding_sites_mask_negative = np.zeros(5000000, dtype=bool)
def add_sites_from_json(json_path, site_type):
    with open(json_path, 'r') as f:
        data = json.load(f)
    for operon in data['data']['getAllOperon']['data']:
        strand = operon['operon'].get('strand')
        for tu in operon['transcriptionUnits']:
            if tu['promoter'] is not None:
                if site_type == 'regulator' and tu['promoter'].get('regulatorBindingSites') is not None:
                    for rbs in tu['promoter']['regulatorBindingSites']:
                        for interaction in rbs['regulatoryInteractions']:
                            confidence_level = interaction.get('confidenceLevel')
                            site = interaction.get('regulatorySite')
                            if site is not None and (confidence_level == 'S' or confidence_level == 'C'):
                                left = site.get('leftEndPosition')
                                right = site.get('rightEndPosition')
                                if left is not None and right is not None:
                                    if strand == 'forward':
                                        binding_sites_mask_positive[left:right + 1] = True
                                    else:
                                        binding_sites_mask_negative[left:right + 1] = True

                elif site_type == 'sigma' and tu['promoter'].get('boxes') is not None:
                    boxes = []
                    for box in tu['promoter']['boxes']:
                        left = box.get('leftEndPosition')
                        right = box.get('rightEndPosition')
                        if left is not None and right is not None:
                            if strand == 'forward':
                                binding_sites_mask_positive[left:right + 1] = True
                            else:
                                binding_sites_mask_negative[left:right + 1] = True
                            boxes.append(np.arange(left, right + 1))


regulator_path = "data/regulator_sites.json"
sigma_path = "data/sigma_box_sites.json"
df_path = "data/ecoli_pairs_with_annotations.csv"

add_sites_from_json(regulator_path, 'regulator')
add_sites_from_json(sigma_path, 'sigma')

df = pd.read_csv(df_path)
df = df[df['operon_member'] == False]
df = df.reset_index(drop=True)

################################################################
#Loading the requested model
if args.model == 'c3p':
    repo_id = "dufaultc/contrastive-promoter-protein-pretraining"
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=args.model_file_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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

elif args.model == 'promogen':    
    llm_tokenizer = AutoTokenizer.from_pretrained("jinyuan22/promogen2-base")
    llm_model = GPT2LMHeadModel.from_pretrained("jinyuan22/promogen2-base").to(device)
    llm_model = torch.compile(llm_model)
    llm_model.eval()

    # Taken from https://huggingface.co/jinyuan22/promogen2-base
    @torch.no_grad()
    def score_promogen(seq, tag="none"):
        # Format input with specified tag
        if tag == "none":
            inputs = llm_tokenizer(f"<|bos|>5{seq}3<|eos|>", return_tensors="pt")
        else:
            inputs = llm_tokenizer(f"<|bos|>{tag}5{seq}3{tag}<|eos|>", return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        pred = llm_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        return pred['loss'].item()

elif args.model == 'evo':
    if Evo is None:
        raise ImportError("Evo package is not installed. Please install evo.")
    evo_model = Evo("evo-1-8k-base")
    evo_model, evo_tokenizer = evo_model.model, evo_model.tokenizer
    evo_model = evo_model.to(device)
    evo_model.eval()

elif args.model == 'evo2':
    if Evo2 is None:
        raise ImportError("Evo2 package is not installed. Please install evo2.")
    evo2_model = Evo2("evo2_7b")

elif args.model == 'glm2':
    glm2_tokenizer = AutoTokenizer.from_pretrained("tattabio/gLM2_650M", trust_remote_code=True)
    glm2_model = AutoModelForMaskedLM.from_pretrained(
        "tattabio/gLM2_650M", torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    glm2_model = torch.compile(glm2_model)
    glm2_model.eval()

roc_scores_list = []
auprc_scores_list = []

total_roc_score = 0
total_auprc_score = 0

bases = ['A', 'C', 'G', 'T']
autocast_dtype = torch.bfloat16 if args.model in ['evo', 'evo2'] else torch.float16


#######################################################
# For every promoter, we first check if any binding sites overlap the sequence, if not we skip to the next
# - Next, we create a list of sequences (seqs_to_test) containing the wildtype promoter and all possible single nucleotide varations of the promoter
#   (we format this differently depending on what the model takes as input)
# - We also create a list (metadata) of the same length which indicates what mutations were applied to each mutant promoter and at what position
# - Then, we score each sequence, for c3p models this represents the cosine distance of the sequence from the protein embedding,
#   for the others it is the sequence likelihood/pseudolikelihoood (calculated differently depending on the model)
# - Then, we calcualte the impact of mutations at each position as the average (over the 3 possible variation) absolute difference between the mutant sequence scores and the wt sequence score.
# - We calcaulte the AUROC and AUPRC of using this mutation impact score to prediction binding sites of the promoter in question zero shot
# We repeat the above for args.num_promoters and output the average AUROC and AUPRC

idx = 0
with tqdm.tqdm(total=args.num_promoters) as pbar:
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=autocast_dtype):
            for i, row in df.iterrows():
                promoter = row['promoter']
                protein_wt = row['protein']
                strand = int(row['strand'])
                promoter_start = int(row['promoter_start'])
                promoter_end = int(row['promoter_end'])

                full_prom_len = len(promoter)
                if full_prom_len < promoter_len:
                    continue

                if strand == 1:
                    genomic_positions = np.arange(promoter_end + 1 - promoter_len, promoter_end + 1)
                    pos_in_sites = binding_sites_mask_positive[genomic_positions]
                else:
                    genomic_positions = np.flip(np.arange(promoter_start + 1, promoter_start + 1 + promoter_len))
                    pos_in_sites = binding_sites_mask_negative[genomic_positions]

                if pos_in_sites.sum() == 0:
                    continue
                
                
                print(f"{idx} {row['gene']} {pos_in_sites.sum()} {max(genomic_positions)} {min(genomic_positions)}")

                promoter_wt = promoter[-promoter_len:]

                seqs_to_test = []
                metadata = []
                #########################################
                # Create input sequences
                # For c3p and promogen we dont include protein sequence
                # For evo and evo 2 we do, as DNA
                # For glm2 we do, as AA
                
                if args.model in ['c3p', 'promogen']:
                    seqs_to_test.append(promoter_wt)
                    metadata.append(('WT',))

                    for p in range(promoter_len):
                        wt_p = promoter_wt[p]
                        for a in bases:
                            if a == wt_p:
                                continue
                            mut = promoter_wt[:p] + a + promoter_wt[p + 1:]
                            seqs_to_test.append(mut)
                            metadata.append(('single', p, a))

                elif args.model in ['evo', 'evo2']:
                    gene_seq = row['gene_seq']
                    seqs_to_test.append(promoter_wt + gene_seq)
                    metadata.append(('WT',))

                    for p in range(promoter_len):
                        wt_p = promoter_wt[p]
                        for a in bases:
                            if a == wt_p:
                                continue
                            mut = promoter_wt[:p] + a + promoter_wt[p + 1:] + gene_seq
                            seqs_to_test.append(mut)
                            metadata.append(('single', p, a))

                elif args.model == 'glm2':
                    seqs_to_test.append('<+>' + promoter_wt.lower() + '<+>' + protein_wt[:1024])
                    metadata.append(('WT',))

                    for p in range(promoter_len):
                        wt_p = promoter_wt[p].lower()
                        for a in bases:
                            if a.lower() == wt_p:
                                continue
                            mut = '<+>' + promoter_wt[:p].lower() + a.lower() + promoter_wt[p + 1:].lower() + '<+>' + protein_wt[:1024]
                            seqs_to_test.append(mut)
                            metadata.append(('single', p, a))



                #####################################################
                # Seqence scoring
                
                if args.model == 'c3p': # cosine distance
                    protein_inputs = model.protein_tokenizer(
                        protein_wt, padding=True, truncation=True, max_length=model.protein_max_len, return_tensors="pt"
                    ).to(device)
                    protein_features = model.protein_encoder(protein_inputs['input_ids'])
                    protein_embedding = model.protein_projection(torch.mean(protein_features.last_hidden_state, dim=1))
                    protein_embedding = F.normalize(protein_embedding, p=2, dim=-1)

                    sim_scores = []
                    for batch_start in range(0, len(seqs_to_test), batch_size):
                        batch_seqs = seqs_to_test[batch_start: batch_start + batch_size]
                        promoter_inputs = model.promoter_tokenizer(
                            batch_seqs, padding=True, truncation=True, return_tensors="pt"
                        ).to(device)

                        promoter_features = model.promoter_encoder(promoter_inputs)
                        promoter_embeddings = model.promoter_projection(promoter_features)
                        promoter_embeddings = F.normalize(promoter_embeddings, p=2, dim=-1)

                        batch_scores = (protein_embedding @ promoter_embeddings.t()).detach().cpu().numpy().flatten()
                        sim_scores.extend(batch_scores)

                    score_dict = {meta: score for meta, score in zip(metadata, sim_scores)}

                elif args.model == 'promogen': # sequence likelihood
                    likelihood_scores = [score_promogen(seq) for seq in seqs_to_test]
                    score_dict = {meta: score for meta, score in zip(metadata, likelihood_scores)}

                elif args.model == 'evo': # sequence likelihood
                    sim_scores = []
                    for batch_start in range(0, len(seqs_to_test), batch_size):
                        batch_seqs = seqs_to_test[batch_start: batch_start + batch_size]
                        sim_scores.extend(score_sequences(batch_seqs, evo_model, evo_tokenizer))
                    score_dict = {meta: score for meta, score in zip(metadata, sim_scores)}

                elif args.model == 'evo2': # sequence likelihood
                    sim_scores = []
                    for batch_start in range(0, len(seqs_to_test), batch_size):
                        batch_seqs = seqs_to_test[batch_start: batch_start + batch_size]
                        sim_scores.extend(evo2_model.score_sequences(batch_seqs))
                    score_dict = {meta: score for meta, score in zip(metadata, sim_scores)}

                elif args.model == 'glm2': # sequence pseudolikelihood
                    sim_scores = []
                    for seq in seqs_to_test:
                        sim_scores.append(score_pll(seq, glm2_tokenizer, glm2_model, batch_size=batch_size, device=device))
                    score_dict = {meta: score for meta, score in zip(metadata, sim_scores)}

                ##########################################################
                # Scoring mutation impact (same for all models)
                S_wt = score_dict[('WT',)]
                position_scores = np.zeros(promoter_len)

                for p in range(promoter_len):
                    wt_p = promoter_wt[p]
                    total_impact = 0
                    for a in bases:
                        if a == wt_p:
                            continue
                        S_mut = score_dict[('single', p, a)]
                        impact = abs(S_wt - S_mut)
                        total_impact += impact

                    avg_impact = total_impact / 3.0
                    position_scores[p] = avg_impact

                roc_score = roc_auc_score(pos_in_sites, position_scores)
                auprc_score = average_precision_score(pos_in_sites, position_scores)
                roc_scores_list.append(roc_score)
                auprc_scores_list.append(auprc_score)
                print(sum(roc_scores_list) / len(roc_scores_list))

                total_roc_score += roc_score
                total_auprc_score += auprc_score
                idx += 1
                pbar.update(1)
                if idx == args.num_promoters:
                    break

sem_auroc = (np.std(roc_scores_list, ddof=1) / np.sqrt(len(roc_scores_list))) * 2
sem_auprc = (np.std(auprc_scores_list, ddof=1) / np.sqrt(len(auprc_scores_list))) * 2

print(len(roc_scores_list))
print(f'avg_auroc_score {total_roc_score / idx}  +/- {sem_auroc}')
print(f'avg_auprc_score {total_auprc_score / idx} +/- {sem_auprc}')