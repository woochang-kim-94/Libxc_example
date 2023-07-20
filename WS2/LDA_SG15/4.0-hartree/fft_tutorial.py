import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift


sym_zero_cent = np.arange(-5, 6)
n1 = len(sym_zero_cent)
print('sym_zero_cent', sym_zero_cent)
sym_zero_init = ifftshift(sym_zero_cent)
print('sym_zero_init', sym_zero_init)

pos_zero_init = np.arange(0,n1)
print('pos_zero_init', pos_zero_init)
pos_zero_cent = fftshift(pos_zero_init)
print('pos_zero_cent', pos_zero_cent)
shifted_pos_zero_init = (pos_zero_init - n1//2) # becomes sym zero_centered
print(np.allclose(shifted_pos_zero_init, sym_zero_cent))
print(n1//2)
print('shifted_pos_zero_init', shifted_pos_zero_init)
print('ifftshift(shifted_pos_zero_init)', ifftshift(shifted_pos_zero_init))
print('pos_zero_init', pos_zero_init)

n1, n2, n3 = [4, 4, 4]
Nt = n1*n2*n3
h_max, k_max, l_max = n1//2, n2//2, n3//2
#generate zero initial positive index
hs, ls, ks = np.unravel_index(np.array(range(Nt)),(n1,n2,n3))
#generate zero centered positive index
print(hs)
hs[hs>=h_max] -= n1
print(hs)
hs = (hs-h_max)
ls = (ls-l_max)
ks = (ks-k_max)
hkls = np.concatenate([[hs],[ls],[ks]], axis = 0).T

