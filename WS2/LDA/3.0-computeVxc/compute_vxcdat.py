import numpy as np
from pymatgen.io.cube import Cube
from fft_mod import put_FFTbox2, ifft_g2r

def main():
    ik = 1
    ibnd_lst = list(range(1,21))
    unkg_lst = np.load(f'./npy.save/Unkg_k{ik}.npy')
    gvec = np.load(f'./npy.save/Gvecs_k{ik}.npy')
    vxc = Cube("./Vxc.cube")
    n1, n2, n3 = vxc.data.shape
    vxc_data = vxc.data
    Nt = n1*n2*n3
    vxc_diag = np.zeros(len(ibnd_lst))

    for ibnd in ibnd_lst:
        unkg = unkg_lst[ibnd-1,:]
        unkg_FFTbox = put_FFTbox2(unkg, gvec, [n1,n2,n3],noncolin=False)
        unkr = ifft_g2r(unkg_FFTbox)
        unkr = unkr[0,:,:,:]
        #unkr *= np.sqrt(Nt)
        temp_grid = np.conjugate(unkr)*vxc_data*unkr
        vxc_el    = np.sum(temp_grid)
        vxc_diag[ibnd-1] = vxc_el
    #const = 10.8464158868
    const = 1.0
    for ibnd in ibnd_lst:
        print(f'<{ibnd}|Vxc|{ibnd}> =',const*vxc_diag[ibnd-1])

    np.savetxt(f'vxc.{ik}.dat',vxc_diag)

    return



if __name__=="__main__":
    main()
