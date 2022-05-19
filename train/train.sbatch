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

# conda env to use: /project2/yangili1/tonyzeng/software/miniconda3/envs/pytorch
module load cuda/11.2

# dataset_train_human.h5 = human training data, dataset_train_all.h5 = all species data
python train_multiple.py 1 ../preprocessing/dataset_train_all.h5
#python train_multiple.py 2 ../preprocessing/dataset_train_all.h5
#python train_multiple.py 3 ../preprocessing/dataset_train_all.h5
#python train_multiple.py 4 ../preprocessing/dataset_train_all.h5
#python train_multiple.py 5 ../preprocessing/dataset_train_all.h5

#python train_single.py 1 models/model.1.5 1 ../preprocessing/dataset_train_all.h5
#python train_single.py 2 models/model.2.5 1 ../preprocessing/dataset_train_all.h5
#python train_single.py 3 models/model.3.5 1 ../preprocessing/dataset_train_all.h5
#python train_single.py 4 models/model.4.5 1 ../preprocessing/dataset_train_all.h5
#python train_single.py 5 models/model.5.5 1 ../preprocessing/dataset_train_all.h5

#python train_single.py 1 models/model.1.5 3 ../preprocessing/dataset_train_all.h5
#python train_single.py 2 models/model.2.5 3 ../preprocessing/dataset_train_all.h5
#python train_single.py 3 models/model.3.5 3 ../preprocessing/dataset_train_all.h5
#python train_single.py 4 models/model.4.5 3 ../preprocessing/dataset_train_all.h5
#python train_single.py 5 models/model.5.5 3 ../preprocessing/dataset_train_all.h5

#python train_single.py 1 models/model.1.5 5 ../preprocessing/dataset_train_all.h5
#python train_single.py 2 models/model.2.5 5 ../preprocessing/dataset_train_all.h5
#python train_single.py 3 models/model.3.5 5 ../preprocessing/dataset_train_all.h5
#python train_single.py 4 models/model.4.5 5 ../preprocessing/dataset_train_all.h5
#python train_single.py 5 models/model.5.5 5 ../preprocessing/dataset_train_all.h5

#python train_single.py 1 models/model.1.5 7 ../preprocessing/dataset_train_all.h5
#python train_single.py 2 models/model.2.5 7 ../preprocessing/dataset_train_all.h5
#python train_single.py 3 models/model.3.5 7 ../preprocessing/dataset_train_all.h5
#python train_single.py 4 models/model.4.5 7 ../preprocessing/dataset_train_all.h5
#python train_single.py 5 models/model.5.5 7 ../preprocessing/dataset_train_all.h5

#python train_single.py 1 models/model.1.5 0 ../preprocessing/dataset_train_all.h5
#python train_single.py 2 models/model.2.5 0 ../preprocessing/dataset_train_all.h5
#python train_single.py 3 models/model.3.5 0 ../preprocessing/dataset_train_all.h5
#python train_single.py 4 models/model.4.5 0 ../preprocessing/dataset_train_all.h5
#python train_single.py 5 models/model.5.5 0 ../preprocessing/dataset_train_all.h5

#python train_single.py 1 models/model.1.5 2 ../preprocessing/dataset_train_all.h5
#python train_single.py 2 models/model.2.5 2 ../preprocessing/dataset_train_all.h5
#python train_single.py 3 models/model.3.5 2 ../preprocessing/dataset_train_all.h5
#python train_single.py 4 models/model.4.5 2 ../preprocessing/dataset_train_all.h5
#python train_single.py 5 models/model.5.5 2 ../preprocessing/dataset_train_all.h5

#python train_single.py 1 models/model.1.5 4 ../preprocessing/dataset_train_all.h5
#python train_single.py 2 models/model.2.5 4 ../preprocessing/dataset_train_all.h5
#python train_single.py 3 models/model.3.5 4 ../preprocessing/dataset_train_all.h5
#python train_single.py 4 models/model.4.5 4 ../preprocessing/dataset_train_all.h5
#python train_single.py 5 models/model.5.5 4 ../preprocessing/dataset_train_all.h5

#python train_single.py 1 models/model.1.5 6 ../preprocessing/dataset_train_all.h5
#python train_single.py 2 models/model.2.5 6 ../preprocessing/dataset_train_all.h5
#python train_single.py 3 models/model.3.5 6 ../preprocessing/dataset_train_all.h5
#python train_single.py 4 models/model.4.5 6 ../preprocessing/dataset_train_all.h5
#python train_single.py 5 models/model.5.5 6 ../preprocessing/dataset_train_all.h5


