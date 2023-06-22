import numpy as np
from pymatgen.io.cube import Cube
from fft_mod import put_FFTbox2, ifft_g2r

def main():
    ik = 1
    ibnd_lst = list(range(1,21))
    unkg_lst = np.load(f'../2.1-wfn/npy.save/Unkg_k{ik}.npy')
    gvec = np.load(f'../2.1-wfn/npy.save/Gvecs_k{ik}.npy')
    vxcr_temp = np.load('./vxc_FFTbox.npy')
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
    for ibnd in ibnd_lst:
        print(f'<{ibnd}|Vxc|{ibnd}> =',const*vxc_diag[ibnd-1])

    np.savetxt(f'vxc.{ik}.dat',const*vxc_diag)

    return



if __name__=="__main__":
    main()
