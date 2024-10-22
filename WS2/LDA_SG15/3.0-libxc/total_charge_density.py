import os
import time
import numpy as np
from pymatgen.core.structure import Structure
from fft_mod import put_FFTbox2, ifft_g2r
np.set_printoptions(5,suppress=True,linewidth=2000,threshold=2000)

bohr2ang = 0.529177
hartree  = 27.2116 #eV
vol      = 1449.7341#321798.0168 *((1/bohr2ang)**3)

def main():
    prefix    = 'WS2'
    savedir   = '../2.1-wfn/npy.save/'
    fn_poscar = './struct.POSCAR' # POSCAR file
    #mode      =  6   #6 is density
    ik_lst    = np.array(range(1,37))  #Start from 1
    highest_band = 20
    lowest_band  = 1
    ibnd_lst  = np.array(range(1,21))  #Start from 1 within unkg file
    FFTgrid   = [ 18, 18, 135] # coarse grid
    noncolin  = False
    save_unkr = True

    #### construct half filling occ        ####
    #### we do not consider spin deg. here ####
    nk   = len(ik_lst)
    nbnd = len(ibnd_lst)
    occ  = np.zeros((nk,nbnd))
    occ[:,:]  += 1
    #occ[5,:]   = 1/2  # K+
    #occ[7,:]   = 1/2  # K-
    #### construct half filling occ        ####

    struct = Structure.from_file(fn_poscar)
    total_density(prefix, savedir, struct, ik_lst, highest_band, \
                lowest_band, ibnd_lst, occ, FFTgrid, noncolin, save_unkr)

def total_density(prefix, savedir, struct, ik_lst, highest_band, \
        lowest_band, ibnd_lst, occ, FFTgrid, noncolin, save_unkr):
    """ Plot density

    ---Input---
    prefix  : str
        system prefix

    savedir : str
        path to save directory contains Unkg, Gvecs

    struct  : Structure

    ik      : int
        k-point id. Start from 1

    highest_band : int

    lowest_band : int

    ibnd_lst: int(:)
        Band index of the system. Start from 1

    FFTgrid : int(3)
        3D FFTgrid N1, N2, N3

    noncolin : bool
        True for non-colinear calc.

    save_unkr : bool

    """
    print('\n#############################################')
    print('#############################################')
    print('\n')
    ibnd_lst = np.array(ibnd_lst, dtype=np.int32)
    nbnd = len(ibnd_lst)
    ibnd_in_wfsx_lst = ibnd_lst - lowest_band
    #print(f'Plot charge density ik = {ik}')
    ##### Normalization
    n1, n2, n3 = FFTgrid
    if noncolin:
        nspin = 2
    else:
        nspin = 1
    rhotot_r_FFTbox = np.zeros((nspin,n1,n2,n3), dtype=np.cdouble)

    nfft   = np.prod(FFTgrid)
    nk   = len(ik_lst)
    nbnd = len(ibnd_lst)
    wk   = 1/nk # we assume a symmetric uniform grid
    #vol  = struct.volume/(bohr2ang**3) # in Bohr**3

    print('FFTgrid: ', FFTgrid)
    print('   nfft: ', nfft)
    print('     nk: ', nk)
    print('    vol: ', vol, ' Bohr^3')
    #####

    #### construct half filling occ ####
    occ *= (2/nspin)
    print('occupation with spin deg.')
    print(occ)
    ####


    #### save unkr_FFTbox
    unkr_FFTbox_set = np.zeros((nk, nbnd, nspin, n1, n2, n3), dtype=np.cdouble)
    #### save unkr_FFTbox

    for ik in ik_lst:
        gvecs = np.load(savedir+f'Gvecs_k{ik}.npy')
        unkg  = np.load(savedir+f'Unkg_k{ik}.npy')
        for i in range(nbnd):
            ibnd   = ibnd_lst[i]
            ibnd_wfsx = ibnd_in_wfsx_lst[i]
            # We put FFTbox and ifft on-the-fly. We can save it for later.
            print()
            print(f'#'*50)
            print(f'State ibnd = {ibnd}, ik={ik}')
            unkg_FFTbox = put_FFTbox2(unkg[ibnd_wfsx], gvecs, \
                FFTgrid, noncolin, savedir=savedir)
            unkr_FFTbox = ifft_g2r(unkg_FFTbox)*np.sqrt(nfft)
            norm_r = np.linalg.norm(unkr_FFTbox)
            print(f'Norm in R: {norm_r}')
            print(f'Density in Hartree Atomic Unit')
            unkr_FFTbox *= np.sqrt(nfft/vol)#/norm_r
            unkr_FFTbox_set[ik-1,i,:,:,:,:] = unkr_FFTbox
            rho_r_FFTbox = get_density(unkr_FFTbox)
            print('sum(rho_r)*(vol/nfft)',np.sum(rho_r_FFTbox)*(vol/nfft))
            rhotot_r_FFTbox[:,:,:,:] += occ[ik-1,i]*wk*rho_r_FFTbox

    print()
    print()
    fn_unkr = f'unkr_FFTbox_set'
    print('Save ',fn_unkr)
    np.save(fn_unkr, unkr_FFTbox_set)
    print('sum(rhotot_r)*(vol/nfft)',np.sum(rhotot_r_FFTbox)*(vol/nfft))
    fn_rhotot_r = f'rhoflat_r_FFTbox'
    print('Save ',fn_rhotot_r)
    np.save(fn_rhotot_r,rhotot_r_FFTbox)

    return None

def get_density(wfc):
    """
    ---Input---
    wfc : cdouble(ispin,N1,N2,N3)
        wfc in FFTbox

    --Output--
    density : float64(:,:,:)
        density in FFTbox
    """
    nspin, N1, N2, N3 = wfc.shape
    density = np.zeros((N1,N2,N3), dtype=np.float64)
    for ispin in range(nspin):
        density += np.abs(wfc[ispin,:,:,:])**2

    return density


if __name__=='__main__':
    main()
