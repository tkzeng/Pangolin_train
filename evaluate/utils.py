import numpy as np
from sklearn.metrics import average_precision_score

IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def one_hot_encode(seq, strand):
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
    if strand == '+':
        seq = np.asarray(list(map(int, list(seq))))
    elif strand == '-':
        seq = np.asarray(list(map(int, list(seq[::-1]))))
        seq = (5 - seq) % 5  # Reverse complement
    return IN_MAP[seq.astype('int8')]


def print_metrics(target, output, spliceai=False):
    is_expr = (target.sum(axis=(1, 2)) >= 1)
    target1 = target[is_expr, 1, :].flatten()
    output1 = output[is_expr, 1, :].flatten()
    if spliceai:
        output2 = output[is_expr, 2, :].flatten()
        output = np.maximum(output1, output2)
    else:
        output = output1

    print_topl_statistics(np.asarray(target1),
                          np.asarray(output))


def print_topl_statistics(y_true, y_pred):
    # Prints the following information: top-kL statistics for k=0.5,1,2,4,
    # auprc, thresholds for k=0.5,1,2,4, number of true splice sites.

    idx_true = np.nonzero(y_true == 1)[0]
    argsorted_y_pred = np.argsort(y_pred)
    sorted_y_pred = np.sort(y_pred)

    topkl_accuracy = []
    threshold = []

    for top_length in [0.5, 1, 2, 4]:
        idx_pred = argsorted_y_pred[-int(top_length * len(idx_true)):]
        topkl_accuracy += [np.size(np.intersect1d(idx_true, idx_pred)) \
                           / float(min(len(idx_pred), len(idx_true)))]
        threshold += [sorted_y_pred[-int(top_length * len(idx_true))]]

    auprc = average_precision_score(y_true, y_pred)

    print("topkl accuracy")
    print(topkl_accuracy)
    print("auprc:")
    print(auprc)
    print("threshold:")
    print(threshold)