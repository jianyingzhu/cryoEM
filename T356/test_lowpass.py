#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import sys
sys.path.append('../')
from utility import mrcread, mrcsave, fourier_lowpass

if __name__ == '__main__':
    data = mrcread('../data/CNG_Reference_000_B_Final.mrc')
    data = fourier_lowpass(data, 160 * 1.32 / 10.)
    mrcsave('../output/cng_lowpass_10A.mrc', data)
