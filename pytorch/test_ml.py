import numpy as np

import sys
sys.path.append('./')
from classifier import run

if __name__ == '__main__':
    res_corr = np.load('../data/res_corr.npy', allow_pickle = True)
    res_wrong = np.load('../data/res_wrong.npy', allow_pickle = True)

    n_corr = res_corr.shape[0]
    n_wrong = res_wrong.shape[0]
    n = n_corr + n_wrong
    boxsize = res_corr.shape[1]

    images = np.concatenate((res_corr, res_wrong), axis = 0)
    images = images.reshape((n, 1, boxsize, boxsize))
    labels = np.zeros(n, np.int64)
    labels[:n_corr] = 1

    run(images, labels)
