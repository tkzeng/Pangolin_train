#!/bin/sh

# heart
python test_categorical_single.py 0 ../preprocessing/dataset_test_1.h5 ../Pangolin/pangolin/models final.1.0.3,final.2.0.3,final.3.0.3,final.4.0.3,final.5.0.3
# liver
python test_categorical_single.py 2 ../preprocessing/dataset_test_1.h5 ../Pangolin/pangolin/models final.1.2.3,final.2.2.3,final.3.2.3,final.4.2.3,final.5.2.3
# brain
python test_categorical_single.py 4 ../preprocessing/dataset_test_1.h5 ../Pangolin/pangolin/models final.1.4.3,final.2.4.3,final.3.4.3,final.4.4.3,final.5.4.3
# testis
python test_categorical_single.py 6 ../preprocessing/dataset_test_1.h5 ../Pangolin/pangolin/models final.1.6.3,final.2.6.3,final.3.6.3,final.4.6.3,final.5.6.3


