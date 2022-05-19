#!/bin/sh

python train_multiple.py 1 ../preprocessing/dataset_train_all.h5
python train_multiple.py 2 ../preprocessing/dataset_train_all.h5
python train_multiple.py 3 ../preprocessing/dataset_train_all.h5
python train_multiple.py 4 ../preprocessing/dataset_train_all.h5
python train_multiple.py 5 ../preprocessing/dataset_train_all.h5

python train_single.py 1 models/model.1.5 1 ../preprocessing/dataset_train_all.h5
python train_single.py 2 models/model.2.5 1 ../preprocessing/dataset_train_all.h5
python train_single.py 3 models/model.3.5 1 ../preprocessing/dataset_train_all.h5
python train_single.py 4 models/model.4.5 1 ../preprocessing/dataset_train_all.h5
python train_single.py 5 models/model.5.5 1 ../preprocessing/dataset_train_all.h5

python train_single.py 1 models/model.1.5 3 ../preprocessing/dataset_train_all.h5
python train_single.py 2 models/model.2.5 3 ../preprocessing/dataset_train_all.h5
python train_single.py 3 models/model.3.5 3 ../preprocessing/dataset_train_all.h5
python train_single.py 4 models/model.4.5 3 ../preprocessing/dataset_train_all.h5
python train_single.py 5 models/model.5.5 3 ../preprocessing/dataset_train_all.h5

python train_single.py 1 models/model.1.5 5 ../preprocessing/dataset_train_all.h5
python train_single.py 2 models/model.2.5 5 ../preprocessing/dataset_train_all.h5
python train_single.py 3 models/model.3.5 5 ../preprocessing/dataset_train_all.h5
python train_single.py 4 models/model.4.5 5 ../preprocessing/dataset_train_all.h5
python train_single.py 5 models/model.5.5 5 ../preprocessing/dataset_train_all.h5

python train_single.py 1 models/model.1.5 7 ../preprocessing/dataset_train_all.h5
python train_single.py 2 models/model.2.5 7 ../preprocessing/dataset_train_all.h5
python train_single.py 3 models/model.3.5 7 ../preprocessing/dataset_train_all.h5
python train_single.py 4 models/model.4.5 7 ../preprocessing/dataset_train_all.h5
python train_single.py 5 models/model.5.5 7 ../preprocessing/dataset_train_all.h5

python train_single.py 1 models/model.1.5 0 ../preprocessing/dataset_train_all.h5
python train_single.py 2 models/model.2.5 0 ../preprocessing/dataset_train_all.h5
python train_single.py 3 models/model.3.5 0 ../preprocessing/dataset_train_all.h5
python train_single.py 4 models/model.4.5 0 ../preprocessing/dataset_train_all.h5
python train_single.py 5 models/model.5.5 0 ../preprocessing/dataset_train_all.h5

python train_single.py 1 models/model.1.5 2 ../preprocessing/dataset_train_all.h5
python train_single.py 2 models/model.2.5 2 ../preprocessing/dataset_train_all.h5
python train_single.py 3 models/model.3.5 2 ../preprocessing/dataset_train_all.h5
python train_single.py 4 models/model.4.5 2 ../preprocessing/dataset_train_all.h5
python train_single.py 5 models/model.5.5 2 ../preprocessing/dataset_train_all.h5

python train_single.py 1 models/model.1.5 4 ../preprocessing/dataset_train_all.h5
python train_single.py 2 models/model.2.5 4 ../preprocessing/dataset_train_all.h5
python train_single.py 3 models/model.3.5 4 ../preprocessing/dataset_train_all.h5
python train_single.py 4 models/model.4.5 4 ../preprocessing/dataset_train_all.h5
python train_single.py 5 models/model.5.5 4 ../preprocessing/dataset_train_all.h5

python train_single.py 1 models/model.1.5 6 ../preprocessing/dataset_train_all.h5
python train_single.py 2 models/model.2.5 6 ../preprocessing/dataset_train_all.h5
python train_single.py 3 models/model.3.5 6 ../preprocessing/dataset_train_all.h5
python train_single.py 4 models/model.4.5 6 ../preprocessing/dataset_train_all.h5
python train_single.py 5 models/model.5.5 6 ../preprocessing/dataset_train_all.h5


