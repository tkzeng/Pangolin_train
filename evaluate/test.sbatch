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

# test
#python test_categorical_multi.py final.1.0.3.v2
#python test_categorical_single.py 0

# conda env to use: /project2/yangili1/tonyzeng/software/miniconda3/envs/pytorch
# dataset_train_human.h5 = human training data, dataset_train_all.h5 = all species data
#python train_multiple.py 1 dataset_train_human.h5
#python train_multiple.py 2 dataset_train_human.h5
#python train_multiple.py 3 dataset_train_human.h5
#python train_multiple.py 4 dataset_train_human.h5
#python train_multiple.py 5 dataset_train_human.h5

#python train_single.py 1 ../train/models/model.1.5 1
#python train_single.py 2 ../train/models/model.2.5 1
#python train_single.py 3 ../train/models/model.3.5 1
#python train_single.py 4 ../train/models/model.4.5 1
#python train_single.py 5 ../train/models/model.5.5 1

#python train_single.py 1 ../train/models/model.1.5 3
#python train_single.py 2 ../train/models/model.2.5 3
#python train_single.py 3 ../train/models/model.3.5 3
#python train_single.py 4 ../train/models/model.4.5 3
#python train_single.py 5 ../train/models/model.5.5 3

#python train_single.py 1 ../train/models/model.1.5 5
#python train_single.py 2 ../train/models/model.2.5 5
#python train_single.py 3 ../train/models/model.3.5 5
#python train_single.py 4 ../train/models/model.4.5 5
#python train_single.py 5 ../train/models/model.5.5 5

#python train_single.py 1 ../train/models/model.1.5 7
#python train_single.py 2 ../train/models/model.2.5 7
#python train_single.py 3 ../train/models/model.3.5 7
#python train_single.py 4 ../train/models/model.4.5 7
#python train_single.py 5 ../train/models/model.5.5 7

#python train_single.py 1 ../train/models/model.1.5 0
#python train_single.py 2 ../train/models/model.2.5 0
#python train_single.py 3 ../train/models/model.3.5 0
#python train_single.py 4 ../train/models/model.4.5 0
#python train_single.py 5 ../train/models/model.5.5 0

#python train_single.py 1 ../train/models/model.1.5 2
#python train_single.py 2 ../train/models/model.2.5 2
#python train_single.py 3 ../train/models/model.3.5 2
#python train_single.py 4 ../train/models/model.4.5 2
#python train_single.py 5 ../train/models/model.5.5 2

#python train_single.py 1 ../train/models/model.1.5 4
#python train_single.py 2 ../train/models/model.2.5 4
#python train_single.py 3 ../train/models/model.3.5 4
#python train_single.py 4 ../train/models/model.4.5 4
#python train_single.py 5 ../train/models/model.5.5 4

#python train_single.py 1 ../train/models/model.1.5 6
#python train_single.py 2 ../train/models/model.2.5 6
#python train_single.py 3 ../train/models/model.3.5 6
#python train_single.py 4 ../train/models/model.4.5 6
#python train_single.py 5 ../train/models/model.5.5 6


