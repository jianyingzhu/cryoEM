#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mrcfile

def mrcread(fname, ret_voxel_size = False):
    with mrcfile.open(fname, mode = 'r', permissive = True) as mrc:
        data = np.array(mrc.data, np.float64, order = 'C')
        voxel_size = mrc.voxel_size
        assert(voxel_size.x == voxel_size.y == voxel_size.z)
    return (data, voxel_size.x) if ret_voxel_size else data

def mrcsave(fname, data, voxel_size = 0.0):
    with mrcfile.new(fname, overwrite = True) as mrc:
        mrc.set_data(np.array(data, np.float32, order = 'C'))
        mrc.voxel_size = voxel_size
