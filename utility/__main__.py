#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from time import time
from utility import mrcread, mrcsave, fourier_lowpass

def test_fourier_lowpass_3d():
    (f, voxel_size) = mrcread('./data/cng_Reference_000_B_Final.mrc', True)
    time1 = time()
    box_size = f.shape[0]
    f = fourier_lowpass(f, box_size * voxel_size / 10)
    time2 = time()
    print('Fourier lowpass of CNG done in %.5fs.' % (time2 - time1))
    mrcsave('./output/cng_lowpass_10A.mrc', f, voxel_size)

if __name__ == '__main__':
    np.set_printoptions(precision = 3, suppress = True)

    test_fourier_lowpass_3d()
