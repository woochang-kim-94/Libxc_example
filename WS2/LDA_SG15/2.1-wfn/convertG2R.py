import numpy as np
from fft_mod import put_FFTbox2, ifft_g2r
vol = 1449.7341

data_g = np.load('./vxc_pw2bgw.npy')
gvec   = np.load('./vxc_gvec.npy')
fn = 'rho_pw2bgw_FFTbox'
n1, n2, n3 = 18, 18, 135
nfft = n1*n2*n3
data_g_FFTbox = put_FFTbox2(data_g, gvec, [n1,n2,n3], noncolin=False)
data_r_FFTbox = ifft_g2r(data_g_FFTbox)#*nfft
value = np.sum(data_r_FFTbox[0,:,:,:])#*(1/nfft)
print(' np.sum(Rho)*vol/nfft = ', value)
np.save('vxcr_pw2bgw', data_r_FFTbox*(nfft/vol))
