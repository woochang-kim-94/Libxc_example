import numpy as np
from pymatgen.io.cube import Cube
from fft_mod import put_FFTbox2, ifft_g2r

def main():
    vxc_pp = Cube("./Vxc.cube")
    vxc_pp_data = vxc_pp.data
    gvec_vxc = np.loadtxt('./vxc_gvec.txt',dtype=np.int32)
    vxc_pw2bgw = np.load('./vxc_pw2bgw.npy')
    #n1, n2, n3 = vxc.data.shape
    n1, n2, n3 = [18, 18, 135]
    #vxcr          = np.zeros((1,n1,n2,n3),dtype=np.complex128)
    #vxcr[0,:,:,:] = vxc_data
    #Nt = n1*n2*n3
    vxc_FFTbox = put_FFTbox2(vxc_pw2bgw, gvec_vxc, [n1,n2,n3], noncolin=False)
    vxcr       = ifft_g2r(vxc_FFTbox)
    print(vxcr/vxc_pp_data)
    print(np.max(vxcr/vxc_pp_data))
    print(np.min(vxcr/vxc_pp_data))
    print(np.average(vxcr/vxc_pp_data))
    np.save('vxcr',vxcr)

    #
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
