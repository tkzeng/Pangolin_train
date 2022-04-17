###############################################################################
'''This parser takes as input the .h5 file produced by create_datafile.py and
outputs a .h5 file with datapoints of the form (X, Y), which can be understood
by Keras models.'''
###############################################################################

import h5py
import numpy as np
import sys
import time
from utils_multi import *
from constants import *

start_time = time.time()

assert sys.argv[1] in ['train', 'test', 'all']
assert sys.argv[2] in ['0', '1', 'all']
species = sys.argv[3]
start = int(sys.argv[4]) # 0 if creating new file, 1 if appending to existing file
fnum = int(sys.argv[5]) 
data_dir='./'

h5f = h5py.File(data_dir + 'datafile_%s_%s' % (species, fnum)
                + '_' + sys.argv[1] + '_' + "all"
                + '.h5', 'r')

print(len(h5f['SEQ']))
print(len(h5f['TX_START']))
NAME = h5f['NAME'][:]
CHROM = h5f['CHROM'][:]
SEQ = h5f['SEQ'][:]
STRAND = h5f['STRAND'][:]
TX_START = h5f['TX_START'][:]
TX_END = h5f['TX_END'][:]
JN_START = h5f['JN_START'][:]
h5f.close()
print("data loaded")

if start == 0:
    h5f2 = h5py.File(data_dir + 'dataset'
                    + '_' + sys.argv[1] + '_' + sys.argv[2]
                    + '.h5', 'w')
else:
    h5f2 = h5py.File(data_dir + 'dataset'
                    + '_' + sys.argv[1] + '_' + sys.argv[2]
                    + '.h5', 'a')
    assert len(h5f2) % 3 == 0
    start = len(h5f2)//3

# only consider genes without paralogs in the training set
if sys.argv[2] == '1':
    genes = [gene.strip() for gene in open("paralogs.txt").readlines()]
    genes = set(genes)

orthologs = set([gene.strip() for gene in open("orthologs.txt").readlines()])
ctr = start

for idx in range(SEQ.shape[0]):
    if sys.argv[2] == '1':
        if NAME[idx].split('.')[0] in genes:
            continue
    #exclude orthologs from macaque/mouse training sets
    if NAME[idx].split('.')[0] in orthologs and sys.argv[1] != "all":
        continue

    X, A0, A1, B0, B1, C0, C1, D0, D1 = create_datapoints(SEQ[idx], STRAND[idx], TX_START[idx], TX_END[idx], JN_START[idx])

    for i, y in enumerate(A0[0]):
        # continue if there are no spliced sites in region
        if (np.sum(A0[0][i][:,1]) + np.sum(B0[0][i][:,1]) + np.sum(C0[0][i][:,1]) + np.sum(D0[0][i][:,1])) < 1:
            continue

        # sequence
        h5f2.create_dataset('X' + str(ctr), data=X[i])

        # A0, B0, C0, D0: spliced or unspliced for 4 tissues
        # A1, B1, C1, D1: splice site usage
        Y = np.concatenate([A0[0][i],np.expand_dims(A1[0][i],1),
                            B0[0][i],np.expand_dims(B1[0][i],1),
                            C0[0][i],np.expand_dims(C1[0][i],1),
                            D0[0][i],np.expand_dims(D1[0][i],1)], axis=1)
        h5f2.create_dataset('Y' + str(ctr), data=Y)

        # coordinates
        if STRAND[idx] == '+':
            h5f2.create_dataset('Z' + str(ctr), data=np.array([CHROM[idx], str(int(TX_START[idx])-SL+SL*i-1), str(int(TX_START[idx])-SL+SL*i+SL+CL_max-1), STRAND[idx]]))
        elif STRAND[idx] == '-':
            h5f2.create_dataset('Z' + str(ctr), data=np.array([CHROM[idx], str(int(TX_END[idx])+SL-SL*i), str(int(TX_END[idx])+SL-SL*i-SL-CL_max), STRAND[idx]]))
        else:
            print("error")
            exit()
            
        ctr += 1

h5f2.close()
print(ctr)
print "--- %s seconds ---" % (time.time() - start_time)

###############################################################################         
