# C3P: Contrastive promoter-protein pretraining yields representations capturing bacterial gene regulation
Our paper: https://arxiv.org/abs/2605.25242

Better instructions to come!

# Embedding with pretrained C3P model

Prior to running the below, ensure pytorch is installed, then clone this repository and run `pip install -r requirements.txt` and `pip install -e .`. Pretrained models downloadable from https://huggingface.co/dufaultc/contrastive-promoter-protein-pretraining

Example
```
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
```


# Training your own C3P model instructions/example
### Installation
We recommend first installing miniconda then following the below steps

- `conda create -n "c3p_env" python=3.12`
- `conda activate c3p_env`
- `conda install -c conda-forge ncbi-datasets-cli`
- `conda install -c conda-forge -c bioconda  mmseqs2`
- Install pytorch for your system[https://pytorch.org/get-started/locally/]
- Clone this repository and navigate to base 
- `pip install -r requirements.txt`
- `pip install -e .`


### Download and extract promoter-protein pairs

* Download gff, protein fasta, and genome fasta files for many bacterial genomes, as well as metadata for each assembly from RefSeq using
`promoter_protein_clip/download_data_parse_metadata.py`. Promoter-protein pairs will be extracted using this data.
* Example: download data for 1000 genomes into `data/download_1000` by running `python ./promoter_protein_clip/download_data_parse_metadata.py --data_name=download_1000 --limit=1000`

### Filter by taxonomy and split into train and validation datasets
* Filter the downloaded genomes by taxonomy (such that they are taxonomically diverse) and split the genomes into train and validation sets by taxonomy using `promoter_protein_clip/filter_taxonomy.py`
* Example: From the 1000 downloaded, select a set of 200 genomes with maximal genus level diversity (the least number of genomes sharing a genus), then from that 100 with maximal species level diversity. Of the species represented in the 100, move genomes from 10% into the test set `python ./promoter_protein_clip/filter_taxonomy.py --data_name=download_1000 --filter_name=diverse_100 --subset_diverse_taxa_levels=genus,species --subset_diverse_taxa_counts=200,100 --train_test_split --test_split_taxa=species --test_fraction=0.1`


### Extract promoter-protein pairs
* Extract promoter-protein pairs from each genome in the train and test datasets using `promoter_protein_clip/extract_promoters_prokaryote.py`
* Example: From train dataset genomes, create a file with all valid promoter-protein pairs (minimum upstream non-coding region of 100 bp not overlapping any other annotated genes unless in operon, further details in paper) `python ./promoter_protein_clip/extract_promoters_prokaryote.py --metadata_path="data/download_1000/diverse_100_train_metadata.csv" --assemblies_folder="data/download_1000/ncbi_dataset/data" --output_file="data/download_1000/diverse_100_train_data.csv"`. Repeat for the validation dataset genomes.

### Cluster and pre-embed protein sequences
* Cluster unique proteins in the train and test datasets with mmseqs2, and then embed each cluster representative with an ESM2 model. During training, rather than each protein sequence being embedded on-the-fly, each will be mapped to its cluster, and then the embedding of the cluster representative retrieved. A file containing the cluster mapping and a file containing the embeddings is saved.
* Example: Get all unique proteins in the previously created train dataset of promoter-protein pairs, cluster by 90% identity and 90% coverage, then embed with the ESM2 8M parameter model. `CUDA_VISIBLE_DEVICES=0 python ./promoter_protein_clip/pre_embed.py --data_path='data/download_1000/diverse_100_train_data.csv' --save_dir='data/download_1000' --save_name='diverse_100_train_embeds.pickle' --prot_to_rep_map_save_name='mapping_diverse_100_train_embeds_clustered.pickle' --cluster_proteins --n_gpus=1 --protein_model='facebook/esm2_t6_8M_UR50D'`. Repeat for the validation dataset.

### Run training

* Train a C3P model with `promoter_protein_clip/train.py`.
* Example based on the above: 
```
CUDA_VISIBLE_DEVICES=0 python ./promoter_protein_clip/train.py \
--data_path='data/download_1000/diverse_100_train_data.csv' \
--val_data_path='data/download_1000/diverse_100_test_data.csv' \
--max_promoter_length=300 \
--train_embed_path='data/download_1000/diverse_100_train_embeds.pickle' \
--train_mapping_path='data/download_1000/mapping_diverse_100_train_embeds_clustered.pickle' \
--val_embed_path='data/download_1000/diverse_100_test_embeds.pickle' \
--val_mapping_path='data/download_1000/mapping_diverse_100_test_embeds_clustered.pickle' \
--random_shift_min=99 \
--protein_model='facebook/esm2_t6_8M_UR50D' \
--projection_dim=128 \
--transformer_depth=4 \
--transformer_num_heads=4 \
--promoter_embedding_dim=128 \
--epochs=10 \
--lr=1e-4 \
--save_dir=./checkpoints/diverse_100 \
--wandb_run_name=diverse_100
```
