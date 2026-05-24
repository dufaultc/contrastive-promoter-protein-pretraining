import subprocess
import tempfile

import torch
from tqdm import tqdm
import argparse
import os
from transformers import EsmModel, EsmTokenizer
import pickle
import tqdm
import pandas as pd
import torch.multiprocessing as mp
import math

def get_args():
    parser = argparse.ArgumentParser(description="Embed protein sequences")
    
    parser.add_argument('--data_path', type=str, default='data/full/enterobacteria_training_data.csv',
                        help="Path to the csv containing the protein data")             
    parser.add_argument('--protein_model', type=str, default='facebook/esm2_t6_8M_UR50D',
                        help="Name of the pre-trained ESM model to use.")  
    parser.add_argument('--batch_size', type=int, default=32)   
    parser.add_argument('--save_dir', type=str, default='data',
                        help="Directory to save the embeddings")           
    parser.add_argument("--save_name", type=str,default = 'embeds.pickle')
    parser.add_argument('--n_gpus', type=int, default=4,
                        help="number of parallel processes")   
    parser.add_argument('--cluster_proteins', action='store_true', default=False,
                        help='Cluster the extracted proteins, such that each embedding is the embedding of its representative')     
    parser.add_argument('--cluster_min_identity', type=float, default=0.9,
                        help="Minimum sequence identity of proteins in the same cluster")
    parser.add_argument('--cluster_min_coverage', type=float, default=0.9,
                        help="Minimum overlap percentage of high identity region of proteins in the same cluster")        
    parser.add_argument("--prot_to_rep_map_save_name", type=str,default = None)         
    return parser.parse_args()
    
def embed_proteins(gpu_num, protein_strings, args):
    torch.cuda.set_device(gpu_num)
    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") 
    protein_strings.sort(key=len)
    protein_tokenizer = EsmTokenizer.from_pretrained(args.protein_model)
    protein_encoder = EsmModel.from_pretrained(args.protein_model).to(device)
    batch_size = args.batch_size

    protein_encoder = torch.compile(protein_encoder)

    protein_encoder.eval()
    embeds_dict = {}
    with torch.inference_mode():
        for i in tqdm.tqdm(range(0, len(protein_strings), batch_size)):
            batch_strings = protein_strings[i:i+batch_size]     
            protein_inputs = protein_tokenizer(
                batch_strings,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors='pt',            
            )
            input_ids = protein_inputs['input_ids'].to(device)
            attention_mask = protein_inputs["attention_mask"].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = protein_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
                expanded_mask = attention_mask.unsqueeze(-1).expand(outputs.size()).to(outputs.dtype)
                masked_embeddings = outputs * expanded_mask
                sum_embeddings = torch.sum(masked_embeddings, dim=1)
                out_raw = sum_embeddings / expanded_mask.sum(dim=1)                

            out_raw_np = out_raw.cpu().numpy()

            for j, prot in enumerate(batch_strings):         
                embeds_dict[prot] = out_raw_np[j]

    #return embeds_dict       
    temp_file = os.path.join(args.save_dir, f"temp_batch_{gpu_num}.pkl")
    with open(temp_file, 'wb') as f:
        pickle.dump(embeds_dict, f)

def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    
    df = pd.read_csv(args.data_path)
    full_protein_strings = list(set(df['protein'].to_list()))    
    del df    

    # Write the sequences to a temporary fasta file, then cluster them
    if args.cluster_proteins:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_fasta = os.path.join(tmp_dir, "all_proteins.fasta")
            output_prefix = os.path.join(tmp_dir, "out")       
            with open(input_fasta, "w") as f:
                for i, seq in enumerate(full_protein_strings):
                    f.write(f">seq_{i}\n{seq}\n")  
            cmd = [
                "mmseqs", "easy-cluster",
                input_fasta, output_prefix, tmp_dir,
                "--min-seq-id", str(args.cluster_min_identity),
                "-c", str(args.cluster_min_coverage),
                "--cov-mode", "0" #bidirectional coverage
                ]
            print("Running clustering")
            subprocess.run(cmd, check=True)
            mapping = {}
            header_to_seq = {f"seq_{i}": seq for i, seq in enumerate(full_protein_strings)}
            cluster_tsv = f"{output_prefix}_cluster.tsv"
            with open(cluster_tsv, "r") as f:
                for line in f:
                    rep_header, member_header = line.strip().split("\t")
                    mapping[header_to_seq[member_header]] = header_to_seq[rep_header]
        original_length = len(full_protein_strings)
        full_protein_strings = list(set(list(mapping.values())))
        new_length = len(full_protein_strings)
        print(f"From {original_length} to {new_length}")
            
        if args.prot_to_rep_map_save_name is None:
            mapping_path = os.path.join(args.save_dir, f"mapping_{args.save_name}")
        else:
            mapping_path = os.path.join(args.save_dir, args.prot_to_rep_map_save_name)
        with open(mapping_path, 'wb') as handle:
            pickle.dump(mapping, handle)   


    split_index = math.ceil(len(full_protein_strings)/args.n_gpus)

    full_embeds_dict = dict()
    
    processes = []
    for i in range(args.n_gpus):
        process = mp.Process(target=embed_proteins, args=(i, full_protein_strings[i*split_index : (i+1)*split_index], args))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    for i in range(args.n_gpus):
        temp_file = os.path.join(args.save_dir, f"temp_batch_{i}.pkl")
        with open(temp_file, 'rb') as f:
            full_embeds_dict.update(pickle.load(f))
        os.remove(temp_file)
    
        
    with open(os.path.join(args.save_dir, args.save_name), 'wb') as handle:
        pickle.dump(full_embeds_dict, handle)        
            
if __name__ == "__main__":
    main()            