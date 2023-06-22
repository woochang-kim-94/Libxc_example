import pylibxc
import numpy as np
from fft_mod import put_FFTbox2, ifft_g2r
# Build functional
func = pylibxc.LibXCFunctional("LDA_C_PW", "unpolarized")

n1, n2, n3 = [18, 18, 135]
gvec_rho = np.loadtxt('../2.1-wfn/rhotot_gvec.txt',dtype=np.int32)
rho_pw2bgw = np.load('../2.1-wfn/rhotot_pw2bgw.npy')
rho_FFTbox = put_FFTbox2(rho_pw2bgw, gvec_rho, [n1,n2,n3], noncolin=False)
rho        = ifft_g2r(rho_FFTbox)
rho_flatten = rho.flatten()*10
print(rho  - np.reshape(rho_flatten, (18,18, 135)))


# Create input
inp = {}
inp["rho"] = rho_flatten

# Compute
ret = func.compute(inp)
print(ret['vrho'])
for k, v in ret.items():
    print(k, v)
vrho = ret['vrho']
zk   = ret['zk']
vxc = vrho*zk
vxc_FFTbox = np.reshape(vxc, (18,18,135))
np.save('vxc_FFTbox', vxc_FFTbox)
