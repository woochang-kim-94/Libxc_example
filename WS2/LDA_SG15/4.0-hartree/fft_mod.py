import os
import time
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift


def flat_FFTbox(unkg_FFTbox):
    """
    Generate a flattened 1d array 'unkg' and corresponding igvecs from unkg_FFTbox.
    Assume unkg_FFTbox has its zero-frequency component in unkg_FFTbox[ispin,0,0,0].
    This functions is a kind of the inverse function of 'put_FFTbox'.
    Note, because of the zero padding in unkg_FFTbox, unkg may have lots of zeros.

    --Input--
    unkg_FFTbox : cdouble(nspin,N1,N2,N3)
        unkg in FFTbox
        for colinear case, nspin = 1
        for non-colinear case, nspin = 2

    --Output--
    unkg : cdouble(nspin*N1*N2*N3)
        for non-colinear case, the first half is for spin-up,
                               the second half is for spin-dw.
    gvecs_sym : int(N1*N2*N3,3)
        array of miller index (h,k,l)
    """
    nspin, n1, n2, n3 = unkg_FFTbox.shape
    ng = n1*n2*n3
    h_max, k_max, l_max = n1//2, n2//2, n3//2

    hs, ks, ls = np.unravel_index(np.array(range(ng)),(n1,n2,n3))

    #generate zero starting positive index. it is just for reshaping
    gvecs_pos = np.concatenate([[hs],[ks],[ls]], axis = 0).T
    hkl2flat = ( n3 * (n2 * gvecs_pos[:, 0] + gvecs_pos[:, 1]) \
                                                    + gvecs_pos[:, 2])
    hs[hs>=h_max] -= n1
    ks[ks>=k_max] -= n2
    ls[ls>=l_max] -= n3

    #generate zero starting symmetric index. it will be returned
    gvecs_sym = np.concatenate([[hs],[ks],[ls]], axis = 0).T

    unkg = np.zeros(nspin*ng,dtype=np.cdouble)
    for ispin in range(nspin):
        gstart =       ispin*ng
        gend   = (ispin + 1)*ng
        unkg[gstart:gend] = unkg_FFTbox[ispin,:,:,:].flat[hkl2flat]

    return unkg, gvecs_sym


def fft_r2g(unkr_FFTbox):
    """ fft from R space to G space

    ---Input---
    unkr_FFTbox : cdouble(ispin,N1,N2,N3)
        unkr in FFTbox

    --Output--
    unkg_FFTbox : cdouble(ispin,N1,N2,N3)
        unkg in FFTbox
    """
    # FFTgrid
    nspin, n1, n2, n3 = unkr_FFTbox.shape
    Nt = n1*n2*n3
    unkg_FFTbox = np.zeros_like(unkr_FFTbox)
    start = time.time()
    for ispin in range(nspin):
        unkg_FFTbox[ispin,:,:,:] = fftn(unkr_FFTbox[ispin,:,:,:])
    end   = time.time()
    print(f'\nFFT time', end-start)

    return unkg_FFTbox

def ifft_g2r(unkg_FFTbox):
    """
    ifft from G space to R space

    with an input array x[l], the output array y[k] satisfy
    y[k] = np.sum(x * np.exp(2j * np.pi * k * np.arange(n)/n)) / len(x)

    Note unkg_FFTbox have the unk(G=0) component as the first element
    in the unkg_FFTbox[ispin,0,0,0].


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
        unkr_FFTbox[ispin,:,:,:] = ifftn(unkg_FFTbox[ispin,:,:,:])
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

        unkg_FFTbox[ispin,0,0,0] = unk_ispin(G=0),
        which is a zero frequency start array.


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
