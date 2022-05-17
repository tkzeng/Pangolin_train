#!/bin/sh
#SBATCH --job-name=pangolin
#SBATCH --time=36:00:00
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --mem=55G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --account=pi-yangili1
#SBATCH -o train_log

module load cuda/11.2

python test_categorical_multi.py ../train/models/model.1.5


