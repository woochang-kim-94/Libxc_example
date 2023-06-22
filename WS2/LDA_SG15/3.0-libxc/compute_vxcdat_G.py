import numpy as np
import pylibxc
from pymatgen.io.cube import Cube
from fft_mod import put_FFTbox2, ifft_g2r

def main():
    ik = 1
    ibnd_lst = list(range(1,21))
    unkg_lst = np.load(f'../2.1-wfn/npy.save/Unkg_k{ik}.npy')
    gvec = np.load(f'../2.1-wfn/npy.save/Gvecs_k{ik}.npy')
    #vxcr_temp = np.load('./vxc_FFTbox.npy')
    vxcr_temp = vxc_libxc()
    n1, n2, n3 = vxcr_temp.shape
    vxcr = np.zeros((1,n1,n2,n3),dtype=np.complex128)
    vxcr[0,:,:,:] = vxcr_temp
    Nt = n1*n2*n3
    vxc_diag = np.zeros(len(ibnd_lst),dtype=np.complex128)

    for ibnd in ibnd_lst:
        unkg = unkg_lst[ibnd-1,:]
        unkg_FFTbox = put_FFTbox2(unkg, gvec, [n1,n2,n3],noncolin=False)
        unkr = ifft_g2r(unkg_FFTbox)
        #unkg_FFTbox = unkr
        #unkr = unkr[0,:,:,:]
        #unkr *= np.sqrt(Nt)
        temp_grid = np.conjugate(unkr)*vxcr*unkr
        #temp_grid = np.conjugate(unkg_FFTbox)*unkg_FFTbox*vxc_FFTbox#*unkg_FFTbox
        print(temp_grid.shape)
        vxc_el    = np.sum(temp_grid)
        vxc_diag[ibnd-1] = vxc_el
    const = 1
    #const = 10.8464158868
    #const = 39758.758151405
    #const = 2845.5096357348
    #const = 3170.9975366561
    #const = 292.3544117929
    #const = 2775.8059027528
    #const = 5.480218527
    #const = 9612938.992198247
    const = 2267412.514292088
    const *= 0.5
    for ibnd in ibnd_lst:
        print(f'<{ibnd}|Vxc|{ibnd}> =',const*vxc_diag[ibnd-1])

    np.savetxt(f'vxc.{ik}.dat',const*vxc_diag)

    return

# Load data

def vxc_libxc():
    n1, n2, n3 = [18, 18, 135]
    #gvec_rho = np.loadtxt('../2.1-wfn/rho_gvec.txt',dtype=np.int32)
    #rho_pw2bgw = np.load('../2.1-wfn/rho_pw2bgw.npy')
    #rho_FFTbox = put_FFTbox2(rho_pw2bgw, gvec_rho, [n1,n2,n3], noncolin=False)
    #rho        = ifft_g2r(rho_FFTbox)
    rho_pp = Cube("../1.1-pp/Rho.cube")
    rho = rho_pp.data

    # Create input
    const = 0.5
    #const = 214.82791025888773*18*18*135
    #const = 18*18*135
    inp = {}
    rho_flatten = rho.flatten()
    #print(rho  - np.reshape(rho_flatten, (18,18, 135)))
    inp["rho"] = const*rho_flatten

    # Build functional
    func_c = pylibxc.LibXCFunctional("LDA_C_PW", "unpolarized")
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

    vxc = zk_x + zk_c + inp["rho"]*vrho_x + inp["rho"]*vrho_c

    vxc_FFTbox = np.reshape(vxc,(18,18,135))
    np.save('vxc_FFTbox', vxc_FFTbox)
    return vxc_FFTbox


if __name__=="__main__":
    main()
