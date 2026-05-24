import random
import pickle
import torch
from datasets import load_dataset

def get_hf_sequence_dataset(
    csv_file_path,
    random_shift=False,
    max_promoter_length=100,
    random_shift_min=99,
    min_promoter_length=100,
):
    ds = load_dataset("csv", data_files=csv_file_path, split="train", num_proc =8)
    ds = ds.filter(lambda x: len(x["promoter"]) >= min_promoter_length, num_proc=8)

    def transform_fn(batch):
        out_proteins = []
        out_promoters = []
        out_accession = []

        for protein, promoter, accession in zip(batch["protein"], batch["promoter"],  batch["accession"]):
            promoter = promoter[-max_promoter_length:]
            if random_shift:
                promoter = get_random_str(promoter, random.randrange(random_shift_min, len(promoter)))
            
            out_proteins.append(protein)
            out_promoters.append(promoter)
            out_accession.append(accession)

        return {"protein": out_proteins, "promoter": out_promoters, "accession": out_accession}
    ds.set_transform(transform_fn)
    return ds

# from https://stackoverflow.com/questions/58334846/randomly-extracting-substrings-of-equal-length-from-the-main-string
def get_random_str(main_str, substr_len):
    idx = random.randrange(0, len(main_str) - substr_len + 1)    # Randomly select an "idx" such that "idx + substr_len <= len(main_str)".
    return main_str[idx : (idx+substr_len)]

def get_collate_fn(protein_tokenizer, promoter_tokenizer, protein_max_len, embeds_dict = None, mapping_dict = None):
    if embeds_dict is not None:
        with open(embeds_dict, 'rb') as handle:
            embeds_dict = pickle.load(handle)  
    if mapping_dict is not None:      
        with open(mapping_dict, 'rb') as handle:
            mapping_dict = pickle.load(handle)          
    def collate_fn(batch):
        protein_strings = [item['protein'] for item in batch]
        promoter_strings =  [item['promoter'] for item in batch]
        if embeds_dict is not None:
            embeds = []
            for prot in protein_strings:
                if mapping_dict is not None:
                    embeds.append(torch.tensor(embeds_dict[mapping_dict[prot]]))
                else:
                    embeds.append(torch.tensor(embeds_dict[prot]))
            protein_inputs = torch.stack(embeds)                    
        else:
            protein_inputs = protein_tokenizer(
                protein_strings,
                padding=True,
                truncation=True, 
                max_length=protein_max_len,
                return_tensors='pt',
            )
        promoter_inputs = promoter_tokenizer(
            promoter_strings,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        return protein_inputs, promoter_inputs
    return collate_fn
