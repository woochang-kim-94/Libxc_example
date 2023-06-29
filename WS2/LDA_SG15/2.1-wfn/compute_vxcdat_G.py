import numpy as np
from pymatgen.io.cube import Cube
from fft_mod import put_FFTbox2, ifft_g2r

def main():
    ik = 1
    ibnd_lst = list(range(1,21))
    rho_pp = Cube("../1.1-pp/Rho.cube")
    rho_pp_data = rho_pp.data

    gvec_rho = np.loadtxt('./rho_gvec.txt',dtype=np.int32)
    rho_pw2bgw = np.load('./rho_pw2bgw.npy')
    print(np.sum(rho_pw2bgw))
    #n1, n2, n3 = vxc.data.shape
    n1, n2, n3 = [18, 18, 135]
    rho_g_FFTbox = put_FFTbox2(rho_pw2bgw, gvec_rho, [n1,n2,n3], noncolin=False)
    rho_r        = ifft_g2r(rho_g_FFTbox)[0,:,:,:]
    #print(rho_r/rho_pp_data)
    print(np.min(np.abs(rho_r/rho_pp_data)))
    print(np.average(np.abs(rho_r/rho_pp_data)))
    print(np.max(np.abs(rho_r/rho_pp_data)))
    #vxc_diag = np.zeros(len(ibnd_lst),dtype=np.complex128)

    #for ibnd in ibnd_lst:
    #    unkg = unkg_lst[ibnd-1,:]
    #    unkg_FFTbox = put_FFTbox2(unkg, gvec, [n1,n2,n3],noncolin=False)
    #    unkr = ifft_g2r(unkg_FFTbox)
    #    #unkg_FFTbox = unkr
    #    #unkr = unkr[0,:,:,:]
    #    #unkr *= np.sqrt(Nt)
    #    temp_grid = np.conjugate(unkr)*vxcr*unkr
    #    #temp_grid = np.conjugate(unkg_FFTbox)*unkg_FFTbox*vxc_FFTbox#*unkg_FFTbox
    #    print(temp_grid.shape)
    #    vxc_el    = np.sum(temp_grid)
    #    vxc_diag[ibnd-1] = vxc_el
    #const = 1
    #const = 10.8464158868
    ##const = 39758.758151405
    ##const = 2845.5096357348
    #for ibnd in ibnd_lst:
    #    print(f'<{ibnd}|Vxc|{ibnd}> =',const*vxc_diag[ibnd-1])

    #np.savetxt(f'vxc.{ik}.dat',const*vxc_diag)

    return



if __name__=="__main__":
    main()
