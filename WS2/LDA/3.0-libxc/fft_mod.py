import os
import time
import numpy as np
from numpy.fft import fftn, ifftn, ifftshift

def ifft_g2r(unkg_FFTbox):
    """ ifft from G space to R space

    ---Input---
    unkg_FFTbox : cdouble(ispin,N1,N2,N3)
        unkg in FFTbox

    --Output--
    unkr_FFTbox : cdouble(ispin,N1,N2,N3)
        unkr in FFTbox
    """
    # FFTgrid
    nspin, n1, n2, n3 = unkg_FFTbox.shape
    Nt = n1*n2*n3
    unkr_FFTbox = np.zeros_like(unkg_FFTbox)
    start = time.time()
    for ispin in range(nspin):
        unkr_FFTbox[ispin,:,:,:] = np.sqrt(Nt)*ifftn(unkg_FFTbox[ispin,:,:,:])
    end   = time.time()
    print(f'\nIFFT time', end-start)

    return unkr_FFTbox

def put_FFTbox2(unkg, gvecs, FFTgrid, noncolin, savedir='./', savefn=False):
    """put unkg(igvec) in a FFTbox of (ispin, ix, iy, iz)
    and save it in a npy file.

    ---Input---
    unkg : cdouble(:)
        unkg for ik and ibnd

    FFTgrid : int(3)
        3D FFTgrid N1, N2, N3

    noncolin : bool
        True for non-colinear calc.

    savedir : path for save directories

    ---Output---
    unkg_FFTbox : cdouble(nspin,N1,N2,N3)
        unkg in FFTbox
        for colinear case, nspin = 1
        for non-colinear case, nspin = 2

    """
    #fn = savedir+f'./Unkg_FFTbox_k{ik}.npy'

    if False:
#    if os.path.isfile(fn):
        print('Found ', fn)
        return np.load(fn)

    else:
        if noncolin:
            nspin = 2
        else:
            nspin = 1

        N1, N2, N3 = FFTgrid
        ngvec = len(gvecs)

        unkg_FFTbox = np.zeros((nspin, N1, N2, N3),dtype=np.cdouble)
        h_max, k_max, l_max = N1//2, N2//2, N3//2
        gvecs_shifted = gvecs + np.array([h_max,k_max,l_max])
        # mapping from (h,k,l) to flatten array index in C-ordering
        hkl2flat = ( N3 * (N2 * gvecs_shifted[:, 0] + gvecs_shifted[:, 1]) \
                                                    + gvecs_shifted[:, 2])

        start = time.time()
        for ispin in range(nspin):
            # First half of unkg for spin-up
            # Second half of unkg for spin-dw
            igstart =       ispin*ngvec
            igend   = (ispin + 1)*ngvec
            unkg_FFTbox[ispin,:,:,:].flat[hkl2flat] = unkg[igstart:igend]
            # For numpy.fft unkg(h=0,k=0,l=0) should be the first one
            unkg_FFTbox[ispin,:,:,:] = ifftshift(unkg_FFTbox[ispin,:,:,:])

        end = time.time()

        #Save it for later calculation
        #np.save(savedir+f'./Unkg_FFTbox_k{ik}.npy',unkg_FFTbox)
        print(f'\nFFTBOX time', end-start)

        return unkg_FFTbox

def put_FFTbox(ik, FFTgrid, savedir='./'):
    """put unkg(ibnd,igvec) in a FFTbox of (ibnd, ix, iy, iz)
    and save it in a npy file.

    ---Input---
    ik : int
        k-point index. starting from 1

    FFTgrid : int(3)
        3D FFTgrid N1, N2, N3

    savedir : path for save directories

    noncolin : bool
        True for non-colinear calc.

    ---Output---
    unkg_FFTbox : cdouble(nbnd,N1,N2,N3)
        unkg in FFTbox

    """
    fn = savedir+f'./Unkg_FFTbox_k{ik}.npy'

    if os.path.isfile(fn):
        print('Found ', fn)
        return np.load(fn)

    else:
        gvecs = np.load(savedir+f'./Gvecs_k{ik}.npy')
        unkg  = np.load(savedir+f'./Unkg_k{ik}.npy')
        N1, N2, N3 = FFTgrid
        nbnd  = len(unkg)

        unkg_FFTbox = np.zeros((nbnd, N1, N2, N3),dtype=np.cdouble)

        start = time.time()
        for igvec, gvec in enumerate(gvecs):
            h, k, l = gvec
            if h < 0:
                h +=N1
            if k < 0:
                k +=N2
            if l < 0:
                l +=N3
            unkg_FFTbox[:,h,k,l] = unkg[:, igvec]
        end = time.time()

        # Save it for later calculation
        np.save(savedir+f'./Unkg_FFTbox_k{ik}.npy',unkg_FFTbox)
        print(f'\nFFTBOX time', end-start)

        return unkg_FFTbox

if __name__=='__main__':
    ik = 1
    FFTgrid = [495, 495, 83]
    unkg_FFTbox1 =  put_FFTbox(ik, FFTgrid, savedir='../../data/')
    unkg_FFTbox2 =  put_FFTbox2(ik, FFTgrid, savedir='../../data/')
    print(np.allclose(unkg_FFTbox2,unkg_FFTbox1))
