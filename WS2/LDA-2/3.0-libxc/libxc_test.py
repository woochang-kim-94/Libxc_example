import pylibxc
import numpy as np
from fft_mod import put_FFTbox2, ifft_g2r

# Load data

n1, n2, n3 = [18, 18, 135]
gvec_rho = np.loadtxt('../2.1-wfn/rhotot_gvec.txt',dtype=np.int32)
rho_pw2bgw = np.load('../2.1-wfn/rhotot_pw2bgw.npy')
rho_FFTbox = put_FFTbox2(rho_pw2bgw, gvec_rho, [n1,n2,n3], noncolin=False)
rho        = ifft_g2r(rho_FFTbox)

# Create input
inp = {}
rho_flatten = rho.flatten()
#print(rho  - np.reshape(rho_flatten, (18,18, 135)))
inp["rho"] = rho_flatten

# Build functional
func_c = pylibxc.LibXCFunctional("LDA_C_PW", "unpolarized")
func_x = pylibxc.LibXCFunctional("LDA_X", "unpolarized")

# Compute exchange part
ret_x = func_x.compute(inp)
print(ret_x['vrho'])
for k, v in ret_x.items():
    print(k, v)
vrho_x = ret_x['vrho']
zk_x   = ret_x['zk']

# Compute correlation part
ret_c = func_c.compute(inp)
print(ret_c['vrho'])
for k, v in ret_c.items():
    print(k, v)
vrho_c = ret_c['vrho']
zk_c   = ret_c['zk']
vxc = vrho_c*zk_c + vrho_c*zk_x + vrho_x*zk_c + vrho_x*vrho_x

vxc_FFTbox = np.reshape(vxc, (18,18,135))
np.save('vxc_FFTbox', vxc_FFTbox)
