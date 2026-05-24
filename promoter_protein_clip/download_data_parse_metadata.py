import argparse
import pandas as pd
import os
import json
import taxopy
import numpy as np
from promoter_protein_clip.utils import get_project_root
import subprocess


def get_args():
    parser = argparse.ArgumentParser(description="Embed protein sequences")
    
    parser.add_argument('--data_name', type=str, required=True,
                        help="Name of this data download")
    parser.add_argument('--assembly_level', type=str, default='complete,chromosome,contig,scaffold',
                        help="quality of assemblies to download")                     
    parser.add_argument('--limit', type=int, default=None,
                        help="Max number of assemblies to download")            
    return parser.parse_args()

def main():
    args = get_args()
    data_folder = os.path.join(get_project_root(), 'data', args.data_name)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)    
    
    # Downloading each genomes metadata
    # Only getting well assembled genomes for now

    metadata_file = os.path.join(data_folder, f"{args.data_name}_metadata.json")
    cmd = f"datasets summary genome taxon 2 \
    --annotated \
    --report 'genome' \
    --exclude-atypical \
    --assembly-source 'RefSeq' \
    {"--limit="+str(args.limit) if args.limit is not None else ""} \
    --assembly-level {args.assembly_level} \
    > {metadata_file}"
    output_dir = os.path.join(get_project_root())
    subprocess.call(cmd, cwd=output_dir, shell=True)    
    
    # Putting the accessions for our genomes into a separate file
    # We need to do this so it can be passed to the download command
    accession_list_file = os.path.join(data_folder, f"{args.data_name}_accessions.csv")
    with open(os.path.join(data_folder, metadata_file)) as f:
        data = json.load(f)
    accessions_list = [x["accession"] for x in data["reports"]]
    df = pd.DataFrame(accessions_list)
    df.set_index(0, inplace=True)
    df.to_csv(accession_list_file, header=False)    
    
    
    # Download gff and protein files for each accession
    # This involves downloading a reference to the data (dehydrated), unzipping it, and then rehydration where the referenced data is actually downloaded
    # We need the gff file with the genome annotations of where are proteins are,
    #  the protein file with all the predicted proteins in each genome,
    #  and the genome fasta file we will pull the promoters from
    cmd = f"datasets download genome accession \
    --inputfile {accession_list_file} \
    --dehydrated \
    --include gff3,protein,genome"  
    print(cmd)
    subprocess.call(cmd, cwd=data_folder, shell=True, stdout =subprocess.DEVNULL,stderr=subprocess.DEVNULL)    
    
    cmd = f"unzip {os.path.join(data_folder, 'ncbi_dataset.zip')}"
    print(cmd)
    subprocess.call(cmd, cwd=data_folder, shell=True) 
    
    
    cmd = f"datasets rehydrate --gzip \
    --directory {data_folder}"
    print(cmd)
    subprocess.call(cmd, cwd=data_folder, shell=True,stdout =subprocess.DEVNULL,stderr=subprocess.DEVNULL)  
    
    print("Downloading finished, now parsing metadata json into csv")     
    
    
    with open(metadata_file, "r") as f:
        data = json.load(f)
    df = pd.DataFrame([x for x in data["reports"]])
    df.sort_index(inplace=True)
    
    
    taxdb = taxopy.TaxDb(taxdb_dir=data_folder)
    df["num_pseudogenes"] = df["annotation_info"].apply(lambda x: x['stats']['gene_counts'].get('pseudogene', None))
    df["num_coding_genes"] = df["annotation_info"].apply(lambda x: x['stats']['gene_counts']['protein_coding'])
    df["num_non_coding_genes"] = df["annotation_info"].apply(lambda x: x['stats']['gene_counts']['non_coding'])
    df["release_date"] = df["annotation_info"].apply(lambda x: x['release_date'])
    df["pipeline"] = df["annotation_info"].apply(lambda x: x.get('pipeline', None))

    df["assembly_level"] = df["assembly_info"].apply(lambda x: x['assembly_level'])

    df["contig_l50"] = df["assembly_stats"].apply(lambda x: x['contig_l50'])
    df["contig_n50"] = df["assembly_stats"].apply(lambda x: x['contig_n50'])
    df["scaffold_l50"] = df["assembly_stats"].apply(lambda x: x['scaffold_l50'])
    df["scaffold_n50"] = df["assembly_stats"].apply(lambda x: x['scaffold_n50'])
    df["gc_percent"] = df["assembly_stats"].apply(lambda x: x.get('gc_percent', None))
    df["number_of_component_sequences"] = df["assembly_stats"].apply(lambda x: x['number_of_component_sequences'])
    df["number_of_contigs"] = df["assembly_stats"].apply(lambda x: x['number_of_contigs'])
    df["number_of_scaffolds"] = df["assembly_stats"].apply(lambda x: x['number_of_scaffolds'])
    df["total_sequence_length"] = df["assembly_stats"].apply(lambda x: x['total_sequence_length'])

    df["best_ani_match_ani"] =  df["average_nucleotide_identity"].apply(lambda x: x.get('best_ani_match', None) if x is not np.nan else None)
    df["best_ani_match_ani"] =  df["best_ani_match_ani"].apply(lambda x: x.get('ani', None) if x is not None else None)
    df["best_ani_match_coverage"] =  df["average_nucleotide_identity"].apply(lambda x: x.get('best_ani_match', None) if x is not np.nan else None)
    df["best_ani_match_coverage"] =  df["best_ani_match_coverage"].apply(lambda x: x.get('assembly_coverage', None) if x is not None else None)

    df["checkm_completeness"] =  df["checkm_info"].apply(lambda x: x.get('completeness', None) if x is not np.nan else None)
    df["checkm_contamination"] = df["checkm_info"].apply(lambda x: x.get('contamination', None) if x is not np.nan else None)

    df["tax_id"] = df["organism"].apply(lambda x: x['tax_id'])
    df["strain"] = df["organism"].apply(lambda x: x.get('infraspecific_names', None))
    df["strain"] = df["strain"].apply(lambda x: x.get('strain', None) if x is not None else None)

    df['is_type'] = df["type_material"].apply(lambda x: x['type_label'] == "TYPE_MATERIAL" if x is not np.nan else False)    
    
    taxonomies = [
        taxopy.Taxon(
            df.loc[i]["organism"]["tax_id"],
            taxdb,
        ).rank_name_dictionary
        for i in list(df.index)
    ]
    df["species"] = [taxonomies[i].get("species") for i in range(len(taxonomies))]
    df["genus"] = [taxonomies[i].get("genus") for i in range(len(taxonomies))]
    df["family"] = [taxonomies[i].get("family") for i in range(len(taxonomies))]
    df["order"] = [taxonomies[i].get("order") for i in range(len(taxonomies))]
    df["class"] = [taxonomies[i].get("class") for i in range(len(taxonomies))]
    df["phylum"] = [taxonomies[i].get("phylum") for i in range(len(taxonomies))]    
    df = df.drop([
    "annotation_info",
    "assembly_info",
    "assembly_stats",
    "average_nucleotide_identity",
    "checkm_info",
    "current_accession",
    "organism",
    "paired_accession",
    "source_database",
    "wgs_info",
    "type_material"
    ],axis=1)
    df.set_index("accession",inplace=True)
    df.to_csv(metadata_file.replace(".json",'.csv'), header=True)

if __name__ == "__main__":
    main() 
