import argparse
import pandas as pd
import os
import numpy as np
import collections


def get_args():
    parser = argparse.ArgumentParser(description="Embed protein sequences")
    
    parser.add_argument('--data_name', type=str, required=True,
                        help="Name of the data download")  
    parser.add_argument('--filter_name', type=str, required=True,
                        help="Name of the filter you are applying here")      
    parser.add_argument('--select_only', type=str, default=None,
                        help="Name of clade to take assemblies from")     
    parser.add_argument('--select_only_accession', type=str, default=None,
                        help="Use if you just want one accession")          
    parser.add_argument('--select_only_taxa_level', type=str, default=None,
                        help="The taxon level of the clade to take assemblies from")          
    parser.add_argument('--subset_diverse_taxa_levels', type=str, default=None,
                        help="taxon levels to subset dataset at, comma separated (ex. 'genus,species')")   
    parser.add_argument('--subset_diverse_taxa_counts', type=str, default=None,
                        help="size of subset dataset at each taxon level, comma separated (ex. '200000,100000')")   
    parser.add_argument('--train_test_split', action='store_true', default=False,
                        help='split data into train and test datasets')         
    parser.add_argument('--test_split_taxa', type=str, default='species',
                        help="the taxon level to split the test data at")      
    parser.add_argument('--test_fraction', type=float, default=0.1,
                        help="The fraction of taxa at test_split_taxa to split off into test data") 
    parser.add_argument('--specific_species_to_exclude', type=str, default=None,
                        help="Species to ensure are in the test set and not the train set, comma separated (e.g, 'Escherichia coli,Klebsiella pneumoniae,Salmonella enterica)")  
    parser.add_argument('--output_path', type=str, default=None,
                        help="File to save output file in, only used if not performing train test split")                                            
    return parser.parse_args()

def subset_diverse_taxa(df, output_size, unique_level):
    counts = collections.defaultdict(int)
    taxa_indices = collections.defaultdict(list)
    taxa = df[unique_level].values.tolist()
    for i, val in enumerate(taxa):
        counts[val] += 1
        taxa_indices[val].append(i)
    selected_counts = {taxon: 0 for taxon in list(counts.keys())}

    total = 0
    level = 1
    max_count = max(counts.values())
    while level <= max_count and total < output_size:
        for taxon in list(counts.keys()):
            if (
                selected_counts[taxon] < counts[taxon]
                and selected_counts[taxon] < level
            ):
                selected_counts[taxon] += 1
                total += 1
                if total >= output_size:
                    break
        level += 1
    selected_indices = []
    for taxon in list(counts.keys()):
        indices = taxa_indices[taxon][: selected_counts[taxon]]
        selected_indices.extend(indices)
    return df.iloc[selected_indices]




def main():
    args = get_args()
    
    df = pd.read_csv(os.path.join('data', args.data_name,args.data_name +'_metadata.csv'),index_col=0)    
    print(df.shape)
    if args.select_only_accession is not None:
        df = df.loc[[args.select_only_accession]]   
    if args.select_only is not None:
        df = df[df[args.select_only_taxa_level] == args.select_only]        
    print(df.shape)
    df = df.sample(frac=1)

    if args.subset_diverse_taxa_levels is not None:
        subset_diverse_taxa_levels = args.subset_diverse_taxa_levels.split(',')
        subset_diverse_taxa_counts = args.subset_diverse_taxa_counts.split(',')
        
        for taxa,count in zip(subset_diverse_taxa_levels, subset_diverse_taxa_counts):
            print(df[taxa].value_counts())        
            df = subset_diverse_taxa(df, int(count), taxa)
            print(df.shape)
            print(df[taxa].value_counts())
            
    if args.train_test_split:
        unique_values = df[args.test_split_taxa].unique()
        np.random.shuffle(unique_values)
        split_index = round(len(unique_values) * args.test_fraction)       
        
        unique_values = unique_values.tolist()
        if args.specific_species_to_exclude is not None:
            for specific_species_to_exclude in args.specific_species_to_exclude.split(','):
                if specific_species_to_exclude in unique_values:
                    unique_values.remove(specific_species_to_exclude)
                    unique_values.insert(0, specific_species_to_exclude) 
                else:
                    print(f"No {specific_species_to_exclude}")
                
            print(unique_values[:10])
            
        unique_values_test = unique_values[:split_index]
        unique_values_train = unique_values[split_index:]
        df_test = df[df[args.test_split_taxa].isin(unique_values_test)]
        df_train = df[df[args.test_split_taxa].isin(unique_values_train)]      
            
        df_train.to_csv(os.path.join('data', args.data_name, f"{args.filter_name if args.filter_name is not None else ""}" +'_train_metadata.csv'), header=True)
        df_test.to_csv(os.path.join('data', args.data_name, f"{args.filter_name if args.filter_name is not None else ""}" +'_test_metadata.csv'), header=True)  
    else:
        if args.output_path is None:
            df.to_csv(os.path.join('data', args.data_name, f"{args.filter_name}" +'_metadata.csv'), header=True)
        else:
            df.to_csv(args.output_path, header=True)
        

if __name__ == "__main__":
    main() 
