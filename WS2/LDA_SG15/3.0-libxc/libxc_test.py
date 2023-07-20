import pylibxc
import numpy as np
from fft_mod import put_FFTbox2, ifft_g2r
from pymatgen.io.cube import Cube
from write_cube_mod import write_cube
bohr2ang = 0.529177210903

# Load data
def vxc_libxc():
    n1, n2, n3 = [18, 18, 135]
    #gvec_rho = np.loadtxt('../2.1-wfn/rho_gvec.txt',dtype=np.int32)
    #rho_pw2bgw = np.load('../2.1-wfn/rho_pw2bgw.npy')
    gvec_rho = np.load('../1-scf/igvec.npy')
    rho_pw2bgw = np.load('../1-scf/rhog.npy')*18*18*135
    rho_FFTbox = put_FFTbox2(rho_pw2bgw, gvec_rho, [n1,n2,n3], noncolin=False)
    rho        = ifft_g2r(rho_FFTbox)
    #rho_pp = Cube("../1.1-pp/Rho.cube")
    #rho = rho_pp.data
    #rho_pw2bgw = np.load("../2.1-wfn/rhor_pw2bgw.npy")
    #rho = rho_pw2bgw[0,:,:,:]

    # Create input
    #const = 214.82791025888773*18*18*135
    #const = 18*18*135
    inp = {}
    rho_flatten = rho.flatten()
    #print(rho  - np.reshape(rho_flatten, (18,18, 135)))
    inp["rho"] = rho_flatten

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
    zk  = zk_x   + zk_c

    vxc_FFTbox = np.reshape(vxc,(18,18,135))
    zk_FFTbox  = np.reshape(zk,(18,18,135))
    Exc = np.sum(zk_FFTbox*rho)
    np.save('vxc_FFTbox', vxc_FFTbox)
    return vxc_FFTbox, Exc

vxc_gen, Exc = vxc_libxc()
print(vxc_gen.shape)
n1, n2, n3 = [18, 18, 135]
nfft = n1*n2*n3
vxc_pp  = Cube("../1.1-pp/Vxc.cube")
vol     = vxc_pp.volume/(bohr2ang**3)   # in Bohr^3
vol     = 1449.7341
vxc_pp  = vxc_pp.data
#vxc_pp  = np.load('../2.1-wfn/vxcr_pw2bgw.npy')
#vxc_pp  = vxc_pp[0,:,:,:]
print(vxc_pp.shape)
#struct  = vxc_pp.structure
#write_cube('vxc_libxc.cube',vxc_gen,struct)
#print(np.max(vxc_pp))
ratio = vxc_gen/vxc_pp
#ratio *= vxc_pp/np.max(vxc_pp)
#write_cube('ratio.cube',ratio,struct)
#print(ratio)
#print('\n')
diff = np.abs(2*vxc_gen - vxc_pp)
print(np.max(diff))
print(np.min(diff))
print(np.average(diff))
#print(np.sort(ratio.flat)[-1000:])
print('Exc*2, ',2*Exc*(vol/nfft))
