import os
import tqdm
import gzip
import pathlib
from BCBio import GFF
from Bio import SeqIO
import argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_args():
    parser = argparse.ArgumentParser(description="extract promoter-protein pairs")
    
    parser.add_argument('--metadata_path', type=str, required=True,
                        help="path to metadata file")    
    parser.add_argument('--output_file', type=str, required=True,
                        help="Output file path")      
    parser.add_argument('--assemblies_folder', type=str, required=True,
                        help="path to folder where each subfolder corresponds to an assembly in the metadata file")                 
    parser.add_argument('--max_promoter_length', type=int, default=512,
                        help="max promoter length")  
    parser.add_argument('--min_promoter_length', type=int, default=100,
                        help="min promoter length")     
    parser.add_argument('--operon_window', type=int, default=20,
                        help="max distance between genes to be considered part of same strand")
    parser.add_argument('--num_workers', type=int, default=8,
                        help="number of parallel processes")     
    parser.add_argument('--use_old_locus', action='store_true', default=False,
                        help='')                             
    return parser.parse_args()


def process_accession(accession_path, max_promoter_length, min_promoter_length, operon_window, use_old_locus=False):
    
    accession_path = pathlib.Path(accession_path)
    accession = accession_path.name    
    
    files = list(accession_path.glob("*"))
    if len(files) > 3:
        print(f"{accession} has more than 3 files, skipping" )
        return []
    
    genome_file = [file for file in files if ".fna" in str(file)][0]
    gff_file = [file for file in files if ".gff" in str(file)][0]
    proteins_file = [file for file in files if ".faa" in str(file)][0]
    
    in_handle_gff = gzip.open(gff_file , mode='rt')
    in_handle_genome = gzip.open(genome_file, mode='rt')
    in_handle_proteins = gzip.open(proteins_file, mode='rt')
        
    seq_record = SeqIO.to_dict(SeqIO.parse(in_handle_genome, "fasta"))
    protein_record = SeqIO.to_dict(SeqIO.parse(in_handle_proteins, "fasta"))
    
    limit_info = dict(gff_type=[
        "pseudogene",
        "gene",
        "region",
        'mRNA',
        'CDS',
        'tRNA',
        'rRNA',
        'tmRNA',
        'ncRNA',
        'SRP_RNA',
        'RNase_P_RNA'
        
    ])
        
    entries = [] # Will hold all our promoter entries    
        

    for rec in GFF.parse(in_handle_gff, limit_info=limit_info):
        in_forward_operon = False  
        current_is_operon_member=False                  
        starts = [0]
        ends = [0]
               
        for i in range(1,len(rec.features)): # Starting from index 1 because first feature is region               
            if rec.features[i].type == 'gene':
                if len(rec.features[i].sub_features) > 0: 
                    if accession == 'GCF_000006945.2':
                        starts.append(rec.features[i].sub_features[0].location.start)
                        ends.append(rec.features[i].sub_features[0].location.end) 
                    else:                     
                        starts.append(rec.features[i].location.start)
                        ends.append(rec.features[i].location.end) 
        starts.append(len(rec))
        reverse_operon_entries = []
        
        
        # Now that we know what locations are covered, we go through and identify what genes
        # have upstream regions which can be at least 100bp long without reaching a covered index
        # For these genes, we extract this upstream sequence from the genome sequence file, up to the max promoter length
        # For mRNA encoding genes, we also get the sequence of the protein they encode            
        valid_gene_count = 1
        for i in range(1,len(rec.features)): # Starting from index 1 because first feature is region                                          
            strand = rec.features[i].location.strand          
            promoter = ""
            gene_type = None
            protein_id = None
            protein_sequence= None
            gene_name = None
            
            if rec.features[i].type == 'gene': # I need to do this in order to skip pseudogenes
                if len(rec.features[i].sub_features) > 0:
                    if rec.features[i].sub_features[0].type == 'CDS':             
                        gene_type = rec.features[i].sub_features[0].type 

                        if strand == 1:                            
                            reverse_operon_entries = [] # If we were accruing operon genes on the negative strand, get rid of them
                            
                            js = []
                            for idx in range(1,4):
                                past_gene_index = valid_gene_count-idx
                                if past_gene_index < 0:
                                    break
                                else:
                                    js.append(min(starts[valid_gene_count] - ends[valid_gene_count-idx], max_promoter_length))
                            j = min(js)                                  
                            
                            if j >= min_promoter_length: # The area before the gene is long enough to be a promoter
                                if accession == 'GCF_000006945.2':
                                    promoter_start = max(rec.features[i].sub_features[0].location.start-j,0)
                                    promoter_end = rec.features[i].sub_features[0].location.start                                    
                                else:
                                    promoter_start = max(rec.features[i].location.start-j,0)
                                    promoter_end = rec.features[i].location.start
                                promoter = str(seq_record[rec.id][promoter_start: promoter_end].seq)#[::-1] #Reversing so that start of the promoter sequence is part closest to the gene                        
                                in_forward_operon = True # Starting an operon
                                operon_promoter = promoter # This genes promoter is the operon promoter
                                operon_strand = 1 # the operon is on the positive strand
                            elif j <= operon_window and j > -operon_window and in_forward_operon and operon_strand == 1: # If distance to last gene is in operon window, we are currently in an operon, and that operon is on the correct strand
                                promoter = operon_promoter # We use the promter for the first gene in the operon
                                current_is_operon_member=True
                            else: #Distance between genes too long for operon, too short for promoter (no mans land)
                                in_forward_operon = False # If we were in an operon, we no longer are 
                                                        
                        elif strand == -1:
                            in_forward_operon = False
                            operon_strand = -1  # Were on the negative strand
                                                    
                            j = min(starts[valid_gene_count+1] - ends[valid_gene_count], max_promoter_length)
                            for idx in range(1,4):
                                past_gene_index = valid_gene_count-idx
                                if past_gene_index < 0:
                                    break
                                else:
                                    if ends[valid_gene_count]-ends[valid_gene_count-idx] < 0:
                                        j = -1000                                                

                            if j >= min_promoter_length: #If the area after the gene is long enough to be a promoter
                                if accession == 'GCF_000006945.2':
                                    promoter_start = rec.features[i].sub_features[0].location.end
                                    promoter_end = rec.features[i].sub_features[0].location.end + j                                    
                                else:
                                    promoter_start = rec.features[i].location.end
                                    promoter_end = rec.features[i].location.end + j
                                promoter = str(seq_record[rec.id][promoter_start: promoter_end].seq.reverse_complement())#[::-1] #Reversing so that start of the promoter sequence is part closest to the gene                                                        
                                for entry in reverse_operon_entries: # Add all genes before this in the same operon
                                    entries.append((entry[0], entry[1], entry[2], entry[3], entry[4], entry[5], promoter, entry[6], entry[7], True, promoter_start, promoter_end))  
                                reverse_operon_entries = [] # End operon
                            elif j <= operon_window and j > -operon_window:                             
                                protein_id = rec.features[i].sub_features[0].qualifiers['protein_id'][0]
                                protein_sequence = str(protein_record[protein_id].seq)
                                gene_name = rec.features[i].sub_features[0].qualifiers.get('gene', rec.features[i].sub_features[0].qualifiers.get('locus_tag'))[0]  
                                if use_old_locus:
                                    locus_tag = rec.features[i].qualifiers.get('old_locus_tag', rec.features[i].sub_features[0].qualifiers.get('locus_tag'))[0]
                                else:    
                                    locus_tag = rec.features[i].sub_features[0].qualifiers.get('locus_tag')[0]                                
                                reverse_operon_entries.append([accession, rec.id, gene_name, protein_id, gene_type, strand, protein_sequence, locus_tag]) # Accruing genes in the operon                                 
                                
                            else: #Distance between genes too long for operon, too short for promoter
                                reverse_operon_entries = []  # Ending the operon if we were in one, discarding accrued genes
                    else:
                        in_forward_operon = False # If we were in an operon, we no longer are 
                        reverse_operon_entries = []                        
                        
                    valid_gene_count += 1
                else:
                    continue
            else:
                continue

                            
            if len(promoter) > 0: #If we successfully found a promoter of adequate length
                protein_id = rec.features[i].sub_features[0].qualifiers['protein_id'][0]
                protein_sequence = str(protein_record[protein_id].seq)
                gene_name = rec.features[i].sub_features[0].qualifiers.get('gene', rec.features[i].sub_features[0].qualifiers.get('locus_tag'))[0]
                if use_old_locus:
                    locus_tag = rec.features[i].qualifiers.get('old_locus_tag', rec.features[i].sub_features[0].qualifiers.get('locus_tag'))[0]
                else:    
                    locus_tag = rec.features[i].sub_features[0].qualifiers.get('locus_tag')[0]                                   
                entries.append((accession, rec.id, gene_name, protein_id, gene_type, strand, promoter, protein_sequence, locus_tag,current_is_operon_member, promoter_start, promoter_end))
            current_is_operon_member=False  
            
    in_handle_genome.close()
    in_handle_proteins.close()
    in_handle_gff.close()        
          
    return entries         
           
def main():
    args = get_args()
    metadata_df = pd.read_csv(args.metadata_path)
    accession_paths = [pathlib.Path(os.path.join(args.assemblies_folder, acc)) for acc in metadata_df["accession"].tolist()]
    all_entries = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_accession,
                str(p),
                args.max_promoter_length,
                args.min_promoter_length,
                args.operon_window,
                args.use_old_locus,
            )
            for p in accession_paths
        ]

        for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Assemblies"):
            all_entries.extend(fut.result())
    csv_columns = ["accession",'trancript_id','gene','protein_id','gene_type','strand', 'promoter', 'protein', 'locus_tag',"operon_member", "promoter_start", "promoter_end"]
    df= pd.DataFrame(all_entries, columns = csv_columns)

    df.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()                 