import numpy as np
from fft_mod import put_FFTbox2, ifft_g2r, fft_r2g, flat_FFTbox
from pymatgen.core.structure import Structure
from pymatgen.io.cube import Cube
np.set_printoptions(5,suppress=True,linewidth=2000,threshold=2000)
bohr2ang = 0.529177
hartree  = 27.2116 #eV
Ha2eV    = hartree
vol      = 1449.7341#321798.0168 *((1/bohr2ang)**3)


def main():
    prefix    = 'WS2'
    savedir   = '../2.1-wfn/npy.save/'
    fn_poscar = './struct.POSCAR' # POSCAR file
    FFTgrid   = [ 18, 18, 135] # coarse grid
    noncolin  = False
    struct = Structure.from_file(fn_poscar)
    #vol    = struct.volume/(bohr2ang**3)   # in Bohr^3
    cell   = struct.lattice._matrix*(1/bohr2ang) # in Bohr
    bcell  = 2*np.pi*np.linalg.inv(cell).T # in Bohr^-1

    n1, n2, n3 = FFTgrid
    nfft = n1*n2*n3
    nk =   36
    nbnd = 20

    fn_vh_r_FFTbox = f'./vxc_sub.npy'
    vh_r_FFTbox   = np.load(fn_vh_r_FFTbox)

    #print('sum(rho_r)*(vol/nfft)',np.sum(rho_r_FFTbox)*(vol/nfft))
    #Eh = np.real(0.5*np.sum(vh_r_FFTbox*rho_r_FFTbox*(vol/nfft)))
    #print(f'Eh = {2*Eh} in Ry')

    #unkr_FFTbox_set = get_unkr_FFTbox_set(nk, nbnd, FFTgrid,
    unkr_FFTbox_set = np.load('./unkr_FFTbox_set.npy')
    vmat = np.zeros((nk,nbnd,nbnd),dtype=np.cdouble)
    for ik in range(nk):
        for i in range(nbnd):
            for j in range(nbnd):
                print('ik,i,j',f'{ik},{i},{j}')
                unkr_FFTbox_i = unkr_FFTbox_set[ik,i,0,:,:,:]
                unkr_FFTbox_j = unkr_FFTbox_set[ik,j,0,:,:,:]
                matel = np.sum(np.conjugate(unkr_FFTbox_i)*vh_r_FFTbox
                               *unkr_FFTbox_j*(vol/nfft))
                vmat[ik,i,j] = matel

        print(vmat[ik]*Ha2eV)

    fn_vmat = 'vxc_sub_matel'
    np.save(fn_vmat, vmat)


    return

def get_unkr_FFTbox_set(nk, nbnd, FFTgrid, noncolin, vol, savedir):
    """
    get set of unkr in FFTbox

    --output--
    unkr_FFTbox_set :  cdouble(nk, nbnd, nspin, n1, n2, n3)

    """
    print()
    print('Start IFFT')
    if noncolin:
        nspin = 2
    else:
        nspin = 1

    n1,n2,n3 = FFTgrid
    nfft = np.prod(FFTgrid)

    unkr_FFTbox_set = np.zeros((nk, nbnd, nspin, n1, n2, n3), dtype=np.cdouble)

    for ik in range(nk):
        gvecs = np.load(savedir+f'Gvecs_k{ik+1}.npy')
        unkg  = np.load(savedir+f'Unkg_k{ik+1}.npy')
        for ibnd in range(nbnd):
            print(f'State ibnd = {ibnd}, ik={ik}')
            unkg_FFTbox = put_FFTbox2(unkg[ibnd], gvecs, \
                FFTgrid, noncolin, savedir=savedir)
            unkr_FFTbox = ifft_g2r(unkg_FFTbox)*np.sqrt(nfft)
            norm_r = np.linalg.norm(unkr_FFTbox)
            print(f'Norm in R: {norm_r}')
            print(f'Density in Hartree Atomic Unit')
            unkr_FFTbox *= np.sqrt(nfft/vol)/norm_r
            unkr_FFTbox_set[ik,ibnd,:,:,:,:] = unkr_FFTbox
            #rho_r_FFTbox = get_density(unkr_FFTbox)*(nfft/vol)

    print('End of IFFT')
    return unkr_FFTbox_set




def get_unkr(ik, ibnd, nfft,vol, savedir):
    fn_unkr =savedir + f'unkr.K{ik+1}.B{ibnd+1}.npy'
    unkr_FFTbox = np.load(fn_unkr)*np.sqrt(nfft)
    norm_r = np.linalg.norm(unkr_FFTbox)
    unkr_FFTbox /= norm_r
    unkr_FFTbox *= np.sqrt(nfft/vol)
    #rho_r_FFTbox = np.abs(unkr_FFTbox)**2
    #print('sum(rho_r)*(vol/nfft)',np.sum(rho_r_FFTbox)*(vol/nfft))
    return unkr_FFTbox


def get_vh(rho_g, igvec, bcell, FFTgrid, savedir):
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
    for ig, g_sq in enumerate(gvec_sq):
        if g_sq < 1e-12:
            vh_g[ig] = 0
        else:
            vh_g[ig] = 4*np.pi*rho_g[ig]/g_sq

    vh_g_FFTbox = put_FFTbox2(vh_g, igvec, FFTgrid, noncolin=False)
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
