import pylibxc
import numpy as np
from netCDF4 import Dataset
from fft_mod import put_FFTbox2, ifft_g2r

bohr2ang = 0.529177
vol  = 214.828427 *((1/bohr2ang)**3)
hartree  = 27.2114 #eV
# Load data
def vxc_libxc():
    rootgrp = Dataset("WS2.nc", "r", format="NETCDF4")
    print(rootgrp['GRID'])
    RhoXC = rootgrp['GRID']['RhoXC'][:]
    nspin, n3, n2, n1 = RhoXC.shape
    print(nspin, n1, n2, n3)
    nfft = n1*n2*n3
    print('np.sum(RhoXC)*vol/nfft')
    print(np.sum(RhoXC[0,:,:,:])*vol/nfft)
    #print(np.sum(RhoXC[1,:,:,:])*vol/nfft)
    rho = RhoXC[0,:,:,:].T
    # Create input
    const = 1
    inp = {}
    rho_flatten = rho.flatten()
    #print(rho  - np.reshape(rho_flatten, (18,18, 135)))
    inp["rho"] = const*rho_flatten

    # Build functional
    func_c = pylibxc.LibXCFunctional("LDA_C_PZ", "unpolarized")
    func_x = pylibxc.LibXCFunctional("LDA_X", "unpolarized")

    # Compute exchange part
    ret_x = func_x.compute(inp)
    #print(ret_x['vrho'])
    for k, v in ret_x.items():
        print(k, v)
    vrho_x = ret_x['vrho'][:,0]
    zk_x   = ret_x['zk'][:,0]

    # Compute correlation part
    ret_c = func_c.compute(inp)
    #print(ret_c['vrho'])
    for k, v in ret_c.items():
        print(k, v)
    vrho_c = ret_c['vrho'][:,0]
    zk_c   = ret_c['zk'][:,0]

    #vxc = zk_x + zk_c + inp["rho"]*vrho_x + inp["rho"]*vrho_c
    vxc = vrho_x + vrho_c

    vxc_FFTbox = np.reshape(vxc,(n1,n2,n3))
    np.save('vxc_FFTbox', vxc_FFTbox)
    return vxc_FFTbox

vxc_gen = vxc_libxc()
rootgrp = Dataset("WS2.nc", "r", format="NETCDF4")
Vt = rootgrp['GRID']['Vt'][:]
Vt = Vt[0,:,:,:]
Vh = rootgrp['GRID']['Vh'][:]
Vxc = Vt-Vh
Vxc = Vxc.T
diff = vxc_gen-Vxc
print(Vxc.shape)
print(vxc_gen.shape)
print('max',np.max(diff))
print('min',np.min(diff))
print('avg',np.average((diff)))
Vxc /= 2
diff = vxc_gen-Vxc
print('max',np.max(diff))
print('min',np.min(diff))
print('avg',np.average((diff)))
