import numpy as np

ref = np.loadtxt('./vxc.1.ref')
vxc = np.loadtxt('./vxc.1.dat')
nbnd = len(ref)
for ibnd in range(nbnd):
    print(ref[ibnd]/vxc[ibnd])
