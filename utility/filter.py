#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.fft import fftshift, ifftshift, rfftn, irfftn
from math import sqrt

def fourier_lowpass(x, r):
    '''
    Fourier lowpass of a 3-D volume

    Parameters
    ==========
    x : numpy.ndarray
        x.ndim == 3
    r : double
        Cut-off frequency for Fourier lowpass.
        The unit of r is the number of voxels.
        The resolution is box_size * voxel_size / r.
    '''
    fx = rfftn(fftshift(x), norm = 'ortho')
    for (i, j, k) in np.ndindex(fx.shape):
        ii = i if i < fx.shape[0] // 2 else i - fx.shape[0]
        jj = j if j < fx.shape[1] // 2 else j - fx.shape[1]
        kk = k
        rho = sqrt(ii ** 2 + jj ** 2 + kk ** 2)
        if rho > r:
            fx[i, j, k] = 0
    return ifftshift(irfftn(fx, norm = 'ortho'))
