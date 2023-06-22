import h5py
import numpy as np

f = h5py.File('./charge-density.hdf5','r')
for attr in f.keys():
    print(attr)

rhotot_g = f['rhotot_g'][:]
print(rhotot_g)
print(rhotot_g.shape)
print(np.sum(rhotot_g))

f.close()

