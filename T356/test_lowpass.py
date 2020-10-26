from math import sqrt
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import mrcfile

def mrcread(fname, ret_voxel_size = False):
    with mrcfile.open(fname, mode = 'r', permissive = True) as mrc:
        data = np.array(mrc.data, np.float64, order = 'C')
        voxel_size = mrc.voxel_size
        assert(voxel_size.x == voxel_size.y == voxel_size.z)
    return (data, voxel_size.x) if ret_voxel_size else data

def mrcsave(fname, data, voxel_size = 0.):
    with mrcfile.new(fname, overwrite = True) as mrc:
        mrc.set_data(np.array(data, np.float32, order = 'C'))
        mrc.voxel_size = voxel_size

def lowpass(x, freq, voxel_size):
    fx = fftn(fftshift(x), norm = 'ortho')

    n = x.shape[0]
    assert(x.shape == (n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                ii = i if i < n // 2 else i - n
                jj = j if j < n // 2 else j - n
                kk = k if k < n // 2 else k - n
                radius = sqrt(ii ** 2 + jj ** 2 + kk ** 2)
                if n * voxel_size < freq * radius:
                    fx[i, j, k] = 0

    x = ifftshift(ifftn(fx, norm = 'ortho'))
    assert(np.linalg.norm(x.imag) < 1e-6)
    return x.real

if __name__ == '__main__':
    (data, voxel_size) = mrcread('../data/CNG_Reference_000_B_Final.mrc', True)
    print(data.shape)
    data = lowpass(data, 10.0, 1.32)
    mrcsave('cng_lowpass.mrc', data)
