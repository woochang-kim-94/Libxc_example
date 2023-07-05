import numpy as np
from fft_mod import put_FFTbox2, ifft_g2r
vol = 1449.7341

data_g = np.load('./rho_pw2bgw.npy')
gvec   = np.load('./rho_gvec.npy')
fn = 'rho_pw2bgw_FFTbox'
n1, n2, n3 = 18, 18, 135
nfft = n1*n2*n3
data_g_FFTbox = put_FFTbox2(data_g, gvec, [n1,n2,n3], noncolin=False)
data_r_FFTbox = ifft_g2r(data_g_FFTbox)
print(' np.sum(Rho)*vol/nfft = ', np.sum(data_r_FFTbox[0,:,:,:])*vol/nfft)
#np.save(fn, data_r_FFTbox)
