#!/bin/sh
#SBATCH --job-name=pangolin
#SBATCH --time=24:00:00
#SBATCH --partition=broadwl-lc
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=1
#SBATCH --account=pi-yangili1
#SBATCH -o log

./create_files.sh
