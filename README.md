# Pangolin_train

The steps described below are for recreating the training and testing steps in [Zeng and Li, Genome Biology 2022](https://doi.org/10.1186/s13059-022-02664-4). In progress--currently setup to produce top-k and AUPRC metrics for Pangolin as in Figure 1b.

### `preprocessing`

Generate training and test datasets. To generate from intermediate files (`splice_table_{species}.txt` present in the repository for `{species}` = `Human`,`Macaque`,`Mouse`,`Rat`) follow Step 2. To run the whole pipeline (starting from RNA-seq reads), follow Step 1 and Step 2.

#### Step 1

Run `snakemake -s Snakefile1 --config SPECIES={species}` and `snakemake -s Snakefile2 --config SPECIES={species}` for `{species}` = `Human`,`Macaque`,`Mouse`,`Rat`. You will probably need to adjust file paths. This will map RNA-seq reads for each species and tissue, quantify usage of splice sites, and output tables of splice sites for each gene. 

Dependencies: [Snakemake](https://snakemake.readthedocs.io/en/stable/), [Samtools](http://www.htslib.org/), [fastp](https://github.com/OpenGene/fastp), [STAR](https://github.com/alexdobin/STAR), [RSEM](https://github.com/deweylab/RSEM), [MMR](https://github.com/ratschlab/mmr), [Sambamba](https://lomereiter.github.io/sambamba/), [RegTools](https://regtools.readthedocs.io/en/latest/), [SpliSER](https://github.com/CraigIDent/SpliSER), [pybedtools](https://daler.github.io/pybedtools/)

Inputs: 
- Reference genomes and annotations from [GENCODE](https://www.gencodegenes.org/) and [Ensembl](https://uswest.ensembl.org/index.html)
- RNA-seq reads from ArrayExpress ([mouse](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-6798), [rat](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-6811), [macaque](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-6813), [human](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-6814))

Outputs:
- `splice_table_{species}.txt` for each species, used to generate training datasets, and `splice_table_Human.test.txt`, used to generate test datasets. Each line is formatted as:
```
gene_id paralog_marker  chromosome  strand  gene_start  gene_end  splice_site_pos:heart_usage,liver_usage,brain_usage,testis_usage,...
# Note: paralog_marker is unused and set to 0 for all genes
# Note: See utils_multi.py for how genes with usage < 0 is interpreted
```

#### Step 2

Run `./create_files.sh`. This will generate `dataset*.h5` files, which are the training and test datasets (requires ~500GB of space). These can be used in the `train` and `evaluate` steps below.

Dependencies:
- `conda create -c bioconda -n create_files_env python=2.7 h5py bedtools` or equivalent

Inputs: 
- `splice_table_{species}.txt` for each species and `splice_table_Human.test.txt` (included in the repository or generated from Step 1)
- Reference genomes for each species from [GENCODE](https://www.gencodegenes.org/) and [Ensembl](https://uswest.ensembl.org/index.html)

Outputs: `dataset_train_all.h5` (all species) and `dataset_test_1.h5` (human test sequences)


### `training`

Run `train.sh` to train all models for the evaluations used in Figure 1b. Depending on your GPU, this may take a few weeks! I have uploaded models from just running the first two lines of `train.sh` to `train/models` for reference. (TODO: Add fine tuning steps for models used in later figures.)

Dependencies:
- `conda create -c pytorch -n train_test_env python=3.8 pytorch torchvision torchaudio cudatoolkit=11.3 h5py` or equivalent

Inputs: 
- `dataset_train_all.h5` from `preprocessing` steps

Outputs:
- Model checkpoints in `train/models`


### `evaluate`

Run `test.sh` to get top-k and AUPRC statistics for test datasets. (TODO: Add additional evaluation metrics.)

Dependencies
- Same as those for `training` + `sklearn` 

Inputs:
- `dataset_test_1.h5` from `preprocessing` steps
- Follow `training` steps or clone https://github.com/tkzeng/Pangolin.git to get models

