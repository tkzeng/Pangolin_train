###############################################################################
'''This code has functions which process the information in the .h5 files
datafile_{}_{}.h5 and convert them into a format usable by Keras.'''
###############################################################################

import numpy as np
import re
from math import ceil
from constants import *

assert CL_max % 2 == 0

IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
# One-hot encoding of the inputs: 0 is for padding, and 1, 2, 3, 4 correspond
# to A, C, G, T respectively.

OUT_MAP = np.asarray([[1, 0],
                      [0, 1],
                      [0, 0]])
# 0 = no splice, 1 = spliced

def reformat_data_X(X0, Y0):
    # This function converts X0, Y0 of the create_datapoints function into
    # blocks such that the data is broken down into data points where the
    # input is a sequence of length SL+CL_max corresponding to SL nucleotides
    # of interest and CL_max context nucleotides, the output is a sequence of
    # length SL corresponding to the splicing information of the nucleotides
    # of interest. The CL_max context nucleotides are such that they are
    # CL_max/2 on either side of the SL nucleotides of interest.

    assert len(X0) == len(Y0[0]) + CL_max
    num_points = ceil_div(len(Y0[0]), SL)

    Xd = np.zeros((num_points, SL+CL_max))
    X0 = np.pad(X0, [0, SL], 'constant', constant_values=0)

    for i in range(num_points):
        Xd[i] = X0[SL*i:CL_max+SL*(i+1)]

    return Xd

def reformat_data_Y(Y0):
    num_points = ceil_div(len(Y0[0]), SL)
    Yd = [-np.ones((num_points, SL)) for t in range(1)]

    Y0 = [np.pad(Y0[t], [0, SL], 'constant', constant_values=-1) for t in range(1)]
 
    for t in range(1):
        for i in range(num_points):
            Yd[t][i] = Y0[t][SL*i:SL*(i+1)]

    return Yd

def one_hot_encode_X(Xd):
    return IN_MAP[Xd.astype('int8')]

def one_hot_encode_Y(Yd):
    return [OUT_MAP[Yd[t].astype('int8')] for t in range(1)]

def ceil_div(x, y):
    return int(ceil(float(x)/y))

# -3 = unexpressed, -2 = # of reads >0 but <threshold, -1 = spliced but no usage est
def get_usage(cov):
    if cov == "":
        return 0
    cov = float(cov)
    if cov < -2.99: # -3
        return -1
    elif cov < -1.99 or cov == 0: # -2 or 0
        return 0
    elif cov > -1.01 and cov < -0.99: # -1
        return -1
    else:
        return cov

def get_bin(cov):
    if cov == "":
        return 0
    cov = float(cov)
    if cov < -2.99:
        return 3
    elif cov < -1.99:
        return 0
    else:
        return 1

def create_datapoints(seq, strand, tx_start, tx_end, jn_start):
    # This function first converts the sequence into an integer array, where
    # A, C, G, T, N are mapped to 1, 2, 3, 4, 0 respectively. If the strand is
    # negative, then reverse complementing is done. The splice junctions 
    # are also converted into an array of integers, where 0, 1, 2, -1 
    # correspond to no splicing, acceptor, donor and missing information
    # respectively. It then calls reformat_data and one_hot_encode
    # and returns X, Y which can be used by Keras models.

    seq = 'N'*(CL_max//2) + seq[CL_max//2:-CL_max//2] + 'N'*(CL_max//2)
    # Context being provided on the RNA and not the DNA

    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')

    tx_start = int(tx_start)
    tx_end = int(tx_end) 

    jn_start = map(lambda x: re.split(';', x)[:-1], [jn_start])

    A0 = [-np.ones(tx_end-tx_start+1) for t in range(1)]
    A1 = [-np.ones(tx_end-tx_start+1) for t in range(1)]
    B0 = [-np.ones(tx_end-tx_start+1) for t in range(1)]
    B1 = [-np.ones(tx_end-tx_start+1) for t in range(1)]
    C0 = [-np.ones(tx_end-tx_start+1) for t in range(1)]
    C1 = [-np.ones(tx_end-tx_start+1) for t in range(1)]
    D0 = [-np.ones(tx_end-tx_start+1) for t in range(1)]
    D1 = [-np.ones(tx_end-tx_start+1) for t in range(1)]

    if strand == '+':
        X0 = np.asarray(map(int, list(seq)))

        for t in range(1):
            if len(jn_start[t]) > 0:
                A0[t] = np.zeros(tx_end-tx_start+1)
                A1[t] = np.zeros(tx_end-tx_start+1)
                B0[t] = np.zeros(tx_end-tx_start+1)
                B1[t] = np.zeros(tx_end-tx_start+1)
                C0[t] = np.zeros(tx_end-tx_start+1)
                C1[t] = np.zeros(tx_end-tx_start+1)
                D0[t] = np.zeros(tx_end-tx_start+1)
                D1[t] = np.zeros(tx_end-tx_start+1)
                for c in jn_start[t]:
                    coord, cov = c.split(':')
                    cov = cov.split(',')
                    assert(len(cov)==4)
                    if tx_start <= int(coord) <= tx_end:
                        A0[t][int(coord)-tx_start] = get_bin(cov[0])
                        A1[t][int(coord)-tx_start] = get_usage(cov[0])
                        B0[t][int(coord)-tx_start] = get_bin(cov[1])
                        B1[t][int(coord)-tx_start] = get_usage(cov[1])
                        C0[t][int(coord)-tx_start] = get_bin(cov[2])
                        C1[t][int(coord)-tx_start] = get_usage(cov[2])
                        D0[t][int(coord)-tx_start] = get_bin(cov[3])
                        D1[t][int(coord)-tx_start] = get_usage(cov[3])
                    # Ignoring junctions outside annotated tx start/end sites
                     
    elif strand == '-':
        X0 = (5-np.asarray(map(int, list(seq[::-1])))) % 5  # Reverse complement

        for t in range(1):
            if len(jn_start[t]) > 0:
                A0[t] = np.zeros(tx_end-tx_start+1)
                A1[t] = np.zeros(tx_end-tx_start+1)
                B0[t] = np.zeros(tx_end-tx_start+1)
                B1[t] = np.zeros(tx_end-tx_start+1)
                C0[t] = np.zeros(tx_end-tx_start+1)
                C1[t] = np.zeros(tx_end-tx_start+1)
                D0[t] = np.zeros(tx_end-tx_start+1)
                D1[t] = np.zeros(tx_end-tx_start+1)
                for c in jn_start[t]:
                    coord, cov = c.split(':')
                    cov = cov.split(',')
                    if tx_start <= int(coord) <= tx_end:
                        A0[t][tx_end-int(coord)] = get_bin(cov[0])
                        A1[t][tx_end-int(coord)] = get_usage(cov[0])
                        B0[t][tx_end-int(coord)] = get_bin(cov[1])
                        B1[t][tx_end-int(coord)] = get_usage(cov[1])
                        C0[t][tx_end-int(coord)] = get_bin(cov[2])
                        C1[t][tx_end-int(coord)] = get_usage(cov[2])
                        D0[t][tx_end-int(coord)] = get_bin(cov[3])
                        D1[t][tx_end-int(coord)] = get_usage(cov[3])

    if np.sum(A0[t]==3) != 0:
        A0[t] = -np.ones(tx_end-tx_start+1)
    if np.sum(B0[t]==3) != 0:
        B0[t] = -np.ones(tx_end-tx_start+1)
    if np.sum(C0[t]==3) != 0:
        C0[t] = -np.ones(tx_end-tx_start+1)
    if np.sum(D0[t]==3) != 0:
        D0[t] = -np.ones(tx_end-tx_start+1)

    X0 = reformat_data_X(X0, A0)
    A0 = reformat_data_Y(A0)
    A1 = reformat_data_Y(A1)
    B0 = reformat_data_Y(B0)
    B1 = reformat_data_Y(B1)
    C0 = reformat_data_Y(C0)
    C1 = reformat_data_Y(C1)
    D0 = reformat_data_Y(D0)
    D1 = reformat_data_Y(D1)

    X0 = one_hot_encode_X(X0)
    A0 = one_hot_encode_Y(A0)
    B0 = one_hot_encode_Y(B0)
    C0 = one_hot_encode_Y(C0)
    D0 = one_hot_encode_Y(D0)

    return [X0, A0, A1, B0, B1, C0, C1, D0, D1]

def clip_datapoints(X, Y, CL, N_GPUS):
    # This function is necessary to make sure of the following:
    # (i) Each time model_m.fit is called, the number of datapoints is a
    # multiple of N_GPUS. Failure to ensure this often results in crashes.
    # (ii) If the required context length is less than CL_max, then
    # appropriate clipping is done below.
    # Additionally, Y is also converted to a list (the .h5 files store 
    # them as an array).

    rem = X.shape[0]%N_GPUS
    clip = (CL_max-CL)//2

    if rem != 0 and clip != 0:
        return X[:-rem, clip:-clip], [Y[t][:-rem] for t in range(1)]
    elif rem == 0 and clip != 0:
        return X[:, clip:-clip], [Y[t] for t in range(1)]
    elif rem != 0 and clip == 0:
        return X[:-rem], [Y[t][:-rem] for t in range(1)]
    else:
        return X, [Y[t] for t in range(1)]
