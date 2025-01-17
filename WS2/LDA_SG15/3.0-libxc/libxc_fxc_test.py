import pylibxc
import numpy as np
from fft_mod import put_FFTbox2, ifft_g2r
from pymatgen.io.cube import Cube
from write_cube_mod import write_cube


def main():
    rho_pw2bgw = np.load("../2.1-wfn/rhor_pw2bgw.npy")
    rho_t = np.real(rho_pw2bgw[0,:,:,:])

    frac = 1e-4
    rho_f = rho_t * frac
    rho_r = rho_t - rho_f

    vxc_rho_t, fxc_rho_t = get_vxc(rho_t)
    vxc_rho_r, fxc_rho_r = get_vxc(rho_r)
    #vxc_approx  = vxc_rho_t - fxc_rho_t*rho_f
    #vxc_approx2 = vxc_rho_t - vxc_rho_f
    vsub        =  vxc_rho_t - vxc_rho_r
    vsub_aprox  =  fxc_rho_t*rho_f
    #vsub_aprox2 =  fxc_rho_r*rho_f

    diff = np.abs(vsub - vsub_aprox)
    print('Error = abs(vsub - vsub_aprox) in Hartree')
    print('max(Error)',np.max(diff))
    print('min(Error)',np.min(diff))
    print('avg(Error)',np.average(diff))

    fn_vsub = 'vxc_sub'
    print('Save ',fn_vsub)
    np.save(fn_vsub, vsub)

    fn_vsub_aprox = 'vxc_sub_linear'
    print('Save ',fn_vsub_aprox)
    np.save(fn_vsub_aprox, vsub_aprox)
    #diff = np.abs(vsub - vsub_aprox2)
    #print()
    #print('diff = abs(vsub - vsub_aprox) in Hartree')
    #print('max(diff)',np.max(diff))
    #print('min(diff)',np.min(diff))
    #print('avg(diff)',np.average(diff))
    #print(np.sort(ratio.flat)[-1000:])
    return

def get_vxc(rho):
    """
    --Input--
    rho : float(:,:,:)
        real space charge density in Hartree Atomic Unit

    --Output--
    vxc : float(:,:,:)

    """
    inp = {}
    n1, n2, n3 = rho.shape
    rho_flatten = rho.flatten()
    inp["rho"]  = rho_flatten

    # Build functional
    func_c = pylibxc.LibXCFunctional("LDA_C_PZ", "unpolarized")
    func_x = pylibxc.LibXCFunctional("LDA_X", "unpolarized")

    # Compute exchange part
    ret_x    = func_x.compute(inp, do_fxc=True)
    v2rho2_x = ret_x['v2rho2'][:,0]
    vrho_x   = ret_x['vrho'][:,0]
    zk_x     = ret_x['zk'][:,0]

    # Compute correlation part
    ret_c    = func_c.compute(inp, do_fxc=True)
    v2rho2_c = ret_x['v2rho2'][:,0]
    vrho_c   = ret_c['vrho'][:,0]
    zk_c     = ret_c['zk'][:,0]

    vxc = vrho_x + vrho_c
    fxc = v2rho2_x + v2rho2_c

    vxc_FFTbox = np.reshape(vxc,(n1,n2,n3))
    fxc_FFTbox = np.reshape(fxc,(n1,n2,n3))

    return vxc_FFTbox, fxc_FFTbox

def vxc_libxc():
    n1, n2, n3 = [18, 18, 135]
    #gvec_rho = np.loadtxt('../2.1-wfn/rho_gvec.txt',dtype=np.int32)
    #rho_pw2bgw = np.load('../2.1-wfn/rho_pw2bgw.npy')
    #rho_FFTbox = put_FFTbox2(rho_pw2bgw, gvec_rho, [n1,n2,n3], noncolin=False)
    #rho        = ifft_g2r(rho_FFTbox)
    #rho_pp = Cube("../1.1-pp/Rho.cube")
    #rho = rho_pp.data
    rho_pw2bgw = np.load("../2.1-wfn/rhor_pw2bgw.npy")
    rho = rho_pw2bgw[0,:,:,:]
    frac = 0.0001
    rho_f = rho * frac
    rho_r = rho - rho_f

    # Create input
    const = 1
    inp = {}
    rho_flatten = rho.flatten()
    rho_r_flatten = rho_r.flatten()
    rho_f_flatten = rho_f.flatten()
    inp["rho"] = const*rho_flatten

    # Build functional
    func_c = pylibxc.LibXCFunctional("LDA_C_PZ", "unpolarized")
    func_x = pylibxc.LibXCFunctional("LDA_X", "unpolarized")

    # Compute exchange part
    ret_x    = func_x.compute(inp, do_fxc=True)
    v2rho2_x = ret_x['v2rho2'][:,0]
    vrho_x   = ret_x['vrho'][:,0]
    zk_x     = ret_x['zk'][:,0]

    # Compute correlation part
    ret_c    = func_c.compute(inp, do_fxc=True)
    v2rho2_c = ret_x['v2rho2'][:,0]
    vrho_c   = ret_c['vrho'][:,0]
    zk_c     = ret_c['zk'][:,0]

    vxc = vrho_x + vrho_c
    fxc = v2rho2_x + v2rho2_c

    vxc_FFTbox = np.reshape(vxc,(18,18,135))
    fxc_FFTbox = np.reshape(fxc,(18,18,135))
    return vxc_FFTbox, fxc_FFTbox

main()
