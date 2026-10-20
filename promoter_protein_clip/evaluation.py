import math
import json
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score,matthews_corrcoef
from warnings import simplefilter
import tqdm
import torch

def load_regulon_data(df, regulon_data_path):
    with open(regulon_data_path, "r") as f:
        regulon_json = json.load(f)

    regulon_dict = {
        entry["_id"]: [
            subentry["name"] for subentry in entry["regulates"]["genes"]
        ]
        for entry in regulon_json["data"]["getAllRegulon"]["data"]
    }    
    inv_regulons_dict = defaultdict(set)
    for k, values in regulon_dict.items():
        for v in values:
            if k not in inv_regulons_dict[v]:
                inv_regulons_dict[v].add(k)
    inv_regulons_dict = dict(inv_regulons_dict)
    for k,v in inv_regulons_dict.items():
        inv_regulons_dict[k] = " ".join(sorted(list(v)))    
    df['regulons'] = df['gene'].map(inv_regulons_dict)
    return df

def load_imodulon_data(df, gene_presence_path):
    gene_presence = pd.read_csv(gene_presence_path)          
    
    imodulons_dict = defaultdict(set)
    for gene, imodulon in zip(gene_presence['Gene'].tolist(), gene_presence['iModulon'].tolist()):
        imodulons_dict[gene].add(str(imodulon))
    imodulons_dict = dict(imodulons_dict)    
    for k,v in imodulons_dict.items():
        imodulons_dict[k] = " ".join(sorted(list(v)))        
    df['imodulons'] = df['locus_tag'].map(imodulons_dict)        
    
    return df

def load_sigmulon_data(df, sigmulon_data_path):
    with open(sigmulon_data_path, "r") as f:
        sigmulon_json = json.load(f)

    sigmulon_dict={}
    for sigmulon in sigmulon_json['data']['getAllSigmulon']['data']:
        sigmulon_dict[sigmulon['sigmaFactor']['gene']['name']] = list(set([gene['name'] for promoter in sigmulon['transcribedPromoters'] for gene in promoter['transcribedGenes']]))  
    
    inv_sigmulons_dict = defaultdict(set)
    for k, values in sigmulon_dict.items():
        for v in values:
            if k not in inv_sigmulons_dict[v]:
                inv_sigmulons_dict[v].add(k)
    inv_sigmulons_dict = dict(inv_sigmulons_dict)
    for k,v in inv_sigmulons_dict.items():
        inv_sigmulons_dict[k] = " ".join(list(v))   
    df['sigmulons'] = df['gene'].map(inv_sigmulons_dict)
    return df

def evaluate_knn(
        df, 
        column_label, 
        nearest_neighbours=None,
        min_members=10,      
        k=1,
):
    unique_values = set()
    for x in df[column_label]:
        if pd.notna(x) and x != set():
            unique_values.update(x.split())
    
    results = {}
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_average_precision = 0
    total_f1 = 0
    total_mcc = 0
    valid_classes = 0
    
    remaining_indices = df.index.tolist()
    remaining_indices_set = set(remaining_indices)

    for category in unique_values:
        is_positive = df[column_label].apply(
            lambda x: category in x.split() if pd.notna(x) else False
        )
        true_dict = is_positive.to_dict()
        
        pos_indices = np.where(is_positive.values)[0]
        
        if len(pos_indices) < min_members:
            continue
        y_test = is_positive.values.astype(int)
        y_pred = np.zeros_like(y_test)
        y_conf = np.zeros_like(y_test)
        
        for i,gene_idx in enumerate(remaining_indices): #range(len(df)):
            valid_neighbors = []
            for n_idx in nearest_neighbours[gene_idx]:
                if n_idx == gene_idx or n_idx not in remaining_indices_set:
                    continue
                valid_neighbors.append(n_idx)
                if len(valid_neighbors) == k:
                    break
            
            if len(valid_neighbors) > 0:
                pos_votes = sum(1 for n in valid_neighbors if true_dict[n])
                if pos_votes > (len(valid_neighbors) / 2):
                    y_pred[i] = 1
                y_conf[i] = pos_votes/ len(valid_neighbors)
                    
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        average_precision = average_precision_score(y_test, y_conf)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        results[f"{column_label}_{category}"] = {
            "accuracy": acc,
            "precision" : precision,
            "average_precision": average_precision,
            "recall": recall,
            "f1" : f1,
            "mcc" : mcc,
            "num_test_samples": len(y_test)
        }
        
        total_accuracy += acc
        total_precision += precision
        total_average_precision += average_precision
        total_recall += recall
        total_f1 += f1
        total_mcc += mcc
        valid_classes += 1

    if valid_classes == 0:
        return {"mean_accuracy": np.nan, "class_results": results}
        
    mean_acc = total_accuracy / valid_classes
    mean_precision = total_precision / valid_classes
    mean_average_precision = total_average_precision/valid_classes
    mean_recall = total_recall / valid_classes 
    mean_f1 = total_f1 / valid_classes        
    mean_mcc = total_mcc/valid_classes
    
    return {
        "mean_accuracy": mean_acc, 
        "mean_precision": mean_precision, 
        "mean_average_precision": mean_average_precision,
        "mean_recall": mean_recall,
        "mean_f1" : mean_f1,
        "mean_mcc" : mean_mcc,
        "num_evaluated_classes": valid_classes,
        "class_results": results
    }
    
    
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
def evaluate_map(
        df,
        column,
        nearest_neighbours,
        min_members=10,
        at_r=False
):
    df = df[df["operon_member"] == False]

    unique_values = set()
    for x in df[column]:
        if pd.notna(x) and x != set():
            unique_values.update(x.split())
    results = {}
    total_ap = 0
    valid_classes = 0

    for k in unique_values:
        category_name = f"{column}_{k}"
        
        in_category_mask = df[column].apply(lambda x: k in x.split() if pd.notna(x) and x != set() else False)
        category_indices = set(df[in_category_mask].index)
        
        if len(category_indices) < min_members:
            continue

        class_ap_sum = 0
        evaluated_queries = 0

        for gene_idx in category_indices:  
            num_relevant = len(category_indices) - 1
            if num_relevant == 0:
                continue

            hits = 0
            ap_sum = 0
            rank = 1

            for neighbour_idx in nearest_neighbours[gene_idx]:
                if neighbour_idx == gene_idx:
                    continue
                if at_r and rank > num_relevant:
                    break
                
                if neighbour_idx in category_indices:
                    hits += 1
                    ap_sum += hits / rank
                
                if not at_r and hits == num_relevant:
                    break
                rank += 1
            
            query_ap = ap_sum / num_relevant
            class_ap_sum += query_ap
            evaluated_queries += 1

        if evaluated_queries > 0:
            class_map = class_ap_sum / evaluated_queries
            
            results[category_name] = {
                "average_precision": class_map,
                "num_evaluated_queries": evaluated_queries,
                "num_members": len(category_indices)
            }
            
            total_ap += class_map
            valid_classes += 1

    mean_ap = total_ap / valid_classes    
    return {
        "mean_average_precision": mean_ap,
        "num_evaluated_classes": valid_classes,
        "class_results": results
    }    
    
    
def top_k_accuracy(
        df,
        column_label,        
        nearest_neighbours,
        k=1,
        full_set=False,
        ):
    try:
        df = df[df["operon_member"] == False]              
    except:
        pass
    
    correct_matches = 0
    total_valid_genes = 0
    
    for gene_idx, row in df.iterrows():
        target_val = row[column_label]
        if pd.isna(target_val):
            continue            
        target_labels = set(target_val.split())            
            
        total_valid_genes += 1
        match_found = False
        neighbors_checked = 0
                
        for neighbour_idx in nearest_neighbours[gene_idx]:
            if neighbour_idx == gene_idx:
                continue
                        
            if neighbour_idx in df.index:
                neighbour_val = df.loc[neighbour_idx, column_label]                
                if pd.notna(neighbour_val):                    
                    neighbour_labels = set(neighbour_val.split())
                    if full_set:
                        if target_labels == neighbour_labels:
                            match_found = True
                            break                            
                    else:
                        if target_labels.intersection(neighbour_labels):
                            match_found = True
                            break                       
                else:
                    continue                     
            neighbors_checked += 1
            if neighbors_checked >= k:
                break                
        if match_found:
            correct_matches += 1
    p = (correct_matches / total_valid_genes if total_valid_genes > 0 else 0.0)
    return p, math.sqrt((p*(1-p))/total_valid_genes)    

def compute_promoter_embeddings(model, df, device, reverse_promoter=False,max_promoter_length=100, batch_size=256):
    promoter_strings = df["promoter"].to_list()
    if reverse_promoter:
        promoter_strings = [promoter[::-1][:max_promoter_length] if len(promoter) > max_promoter_length else promoter[::-1] for promoter in promoter_strings]    
    else:
        promoter_strings = [promoter[-max_promoter_length:] if len(promoter) > max_promoter_length else promoter for promoter in promoter_strings]   

    model.eval()
    all_embeddings = []
    for i in tqdm.tqdm(range(0, len(promoter_strings), batch_size)):
        batch_promoters = promoter_strings[i : i + batch_size]
        batch_inputs = model.promoter_tokenizer(
            batch_promoters,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embeddings = (model.promoter_encoder(batch_inputs.to(device)).cpu().numpy())   
            all_embeddings.append(embeddings)   
    return np.concatenate(all_embeddings, axis=0)   

def compute_protein_embeddings(model, df, batch_size, device):
    protein_strings = df["protein"].to_list()

    protein_inputs = model.protein_tokenizer(
        protein_strings,
        padding=True,
        truncation=True,
        max_length=model.protein_max_len,
        return_tensors="pt",
    ).to(device)

    outputs_raw = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(protein_strings), batch_size):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model.protein_encoder(
                    input_ids=protein_inputs["input_ids"][i : i + batch_size],
                    attention_mask=protein_inputs["attention_mask"][i : i + batch_size],
                ).last_hidden_state
                expanded_mask = protein_inputs["attention_mask"][i : i + batch_size].unsqueeze(-1).expand(outputs.size()).to(outputs.dtype)
                masked_embeddings = outputs * expanded_mask
                sum_embeddings = torch.sum(masked_embeddings, dim=1)
                out_raw = (sum_embeddings / expanded_mask.sum(dim=1)).cpu()                  
                                     
            outputs_raw.append(out_raw)

    protein_embeddings_raw = torch.cat(outputs_raw).numpy()
    return protein_embeddings_raw     