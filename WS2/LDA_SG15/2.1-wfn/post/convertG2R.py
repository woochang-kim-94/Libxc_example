import numpy as np
from fft_mod import put_FFTbox2, ifft_g2r

data_g = np.load('./vxc_pw2bgw.npy')
gvec   = np.loadtxt('./vxc_gvec.txt',dtype=np.int32)
fn = 'vxc_pw2bgw_FFTbox'
n1, n2, n3 = 18, 18, 135
data_g_FFTbox = put_FFTbox2(data_g, gvec, [n1,n2,n3], noncolin=False)
data_r_FFTbox = ifft_g2r(data_g_FFTbox)
np.save(fn, data_r_FFTbox)
