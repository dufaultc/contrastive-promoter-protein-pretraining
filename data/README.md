running `python promoter_protein_clip/download_data_parse_metadata.py` with appropriate arguments will download data to subdirectories in this directory


`data/imodulondb` contains the data for performing the zero-shot co-regulated gene retrieval evaluation on the 25 iModulonDB datasets. Each subfolder contains the extracted promoter-protein pairs for a given dataset, along with the iModulon annotations.

`data/RefSeq_all_feb_18_2026` contains the csv files which indicate what specific RefSeq genome assemblies were used to build the train and test datasets we used for model training in our work.

`data/ecoli_pairs_with_annotations.csv` Contains the extracted promoter-protein pairs for the E. coli reference genome, with each gene being annotated with its regulons and sigmulons according to RegulonDB.