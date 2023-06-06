import numpy as np
from pymatgen.io.cube import Cube
from fft_mod import put_FFTbox2, ifft_g2r

def main():
    ik = 1
    ibnd_lst = list(range(1,21))
    unkg_lst = np.load(f'./npy.save/Unkg_k{ik}.npy')
    gvec = np.load(f'./npy.save/Gvecs_k{ik}.npy')
    vxc = Cube("./Vxc.cube")
    #n1, n2, n3 = vxc.data.shape
    n1, n2, n3 = [18, 18, 135]
    vxc_data = vxc.data
    Nt = n1*n2*n3
    #gvec_vxc = np.loadtxt('./vxc_gvec.txt',dtype=np.int32)
    #vxc_pw2bgw = np.load('./vxc_pw2bgw.npy')
    #vxc_FFTbox = put_FFTbox2(vxc_pw2bgw, gvec_vxc, [n1,n2,n3], noncolin=False)
    vxc_diag = np.zeros(len(ibnd_lst),dtype=np.complex128)

    for ibnd in ibnd_lst:
        unkg = unkg_lst[ibnd-1,:]
        unkg_FFTbox = put_FFTbox2(unkg, gvec, [n1,n2,n3],noncolin=False)
        unkr = ifft_g2r(unkg_FFTbox)
        #unkg_FFTbox = unkr
        unkr = unkr[0,:,:,:]

        #unkr *= np.sqrt(Nt)
        temp_grid = np.conjugate(unkr)*vxc_data*unkr
        #temp_grid = np.conjugate(unkg_FFTbox)*unkg_FFTbox*vxc_FFTbox#*unkg_FFTbox
        print(temp_grid.shape)
        vxc_el    = np.sum(temp_grid)
        vxc_diag[ibnd-1] = vxc_el
    #const = 10.8464158868
    #const = 39758.758151405
    const = 12.9503243987
    for ibnd in ibnd_lst:
        print(f'<{ibnd}|Vxc|{ibnd}> =',const*vxc_diag[ibnd-1])

    np.savetxt(f'vxc.{ik}.dat',vxc_diag)

    return



if __name__=="__main__":
    main()
