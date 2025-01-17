import numpy as np
from fft_mod import put_FFTbox2, ifft_g2r, fft_r2g, flat_FFTbox
from pymatgen.io.cube import Cube
bohr2ang = 0.529177249


def main():
    n1, n2, n3 = [18, 18, 135]
    nfft = n1*n2*n3

    # vh
    vbh = Cube("../1.1-pp/Vh.cube") #just for struct
    struct = vbh.structure
    vol    = vbh.volume/(bohr2ang**3)   # in Bohr^3
    vol     = 1449.7341
    cell   = struct.lattice._matrix*(1/bohr2ang) # in Bohr
    bcell  = 2*np.pi*np.linalg.inv(cell).T # in Bohr^-1

    # rho_r
    rho_r_FFTbox = np.zeros((1,n1,n2,n3), dtype=np.cdouble)
    rho = Cube("../1.1-pp/Rho.cube") #just for struct
    rho_r_FFTbox[0,:,:,:] = rho.data
    print('sum(rho_r)*(vol/nfft)',np.sum(rho_r_FFTbox)*(vol/nfft))
    # debug
    rho_g_FFTbox  = fft_r2g(rho_r_FFTbox)
    rho_g, igvec  = flat_FFTbox(rho_g_FFTbox)
    vh_r   = get_vh(rho_g, igvec, bcell)
    rho_r  = rho_r_FFTbox
    Eh     = np.real(0.5*np.sum(vh_r*rho_r*(vol/nfft)))

    vbh_r = vbh.data/2 # from Ry to Hartree
    Eh_ref = np.real(0.5*np.sum(vbh_r*rho_r*(vol/nfft)))

    diff = np.abs(vbh_r-vh_r)
    print(f'Eh_ref = {2*Eh_ref} in Ry')
    print(f'Eh_ref-Eh = {2*(Eh_ref-Eh)} in Ry')
    print(f'Eh_ref-Eh/Eh_ref = {(Eh_ref-Eh)/Eh_ref} ')
    print('Error = abs(vsub - vsub_aprox) in Hartree')
    print('max(Error)',np.max(diff))
    print('min(Error)',np.min(diff))
    print('avg(Error)',np.average(diff))
    diff = np.abs(vbh_r/(vh_r))
    print('Ratio = abs(vbh_r/vh_r)')
    print('max(Error)',np.max(diff))
    print('min(Error)',np.min(diff))
    print('avg(Error)',np.average(diff))
    #print()
    #print('diff = abs(vsub - vsub_aprox) in Hartree')
    #print('max(diff)',np.max(diff))
    #print('min(diff)',np.min(diff))
    #print('avg(diff)',np.average(diff))
    #print(np.sort(ratio.flat)[-1000:])
    return


def get_vh(rho_g, igvec, bcell):
    """
    --Input--
    rho_g : complex128(ng)
        momentum space charge density in Hartree Atomic Unit

    gvec : int(ng,3)
        momentum space charge density in Hartree Atomic Unit

    bcell_in: float64(3,3)
        reciprocal lattice vectors (bohr^{-1})

    --Output--
    vh : float(:,:,:)

    """
    vh_g      = np.zeros_like(rho_g)
    gvec      = igvec.dot(bcell)
    gvec_sq   = np.einsum('ix,ix->i', gvec, gvec) # bohr^{-1}  or Ry
    print(gvec_sq)
    zero_indices = gvec_sq < 1e-12
    print(zero_indices)
    vh_g[~zero_indices] = 4*np.pi*rho_g[~zero_indices]/gvec_sq[~zero_indices]
    vh_g[zero_indices]  = 0.0
    """
    for ig, g_sq in enumerate(gvec_sq):
        if g_sq < 1e-12:
            vh_g[ig] = 0
        else:
            vh_g[ig] = 4*np.pi*rho_g[ig]/g_sq
    """
    n1, n2, n3 = [18, 18, 135]
    vh_g_FFTbox = put_FFTbox2(vh_g, igvec, [n1, n2, n3], noncolin=False)
    vh_r = ifft_g2r(vh_g_FFTbox)
    return np.real(vh_r[0,:,:,:])



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
