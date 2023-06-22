import os
import time
import numpy as np
from pymatgen.core.structure import Structure
bohr2ang = 0.529177249

def write_cube(fn, data, struct):
    """
    write a gaussian cube file from a 3D data

    ---Input---
    fn   : name of output file

    data : real(:,:,:)
        data for plotting

    struct : pymatgen Structure instance

    """
    print('\nStart writing cube file ', fn)
    natom      = len(struct.sites)
    latt       = (1/bohr2ang)*struct.lattice._matrix
    at_n_lst    = struct.atomic_numbers
    origin     = np.array([0.0,0.0,0.0])
    n1, n2, n3 = data.shape
    a1 = latt[0]/n1
    a2 = latt[1]/n2
    a3 = latt[2]/n3

    f = open(fn, 'w')
    f.write('Cubefile created from Siesta. By W. Kim\n')
    f.write(fn+'\n')
    f.write(f'{natom:6d} {origin[0]:04.6f} {origin[1]:4.6f} {origin[2]:4.6f}\n')
    f.write(f'{n1:6d} {a1[0]:4.6f} {a1[1]:4.6f} {a1[2]:4.6f}\n')
    f.write(f'{n2:6d} {a2[0]:4.6f} {a2[1]:4.6f} {a2[2]:4.6f}\n')
    f.write(f'{n3:6d} {a3[0]:4.6f} {a3[1]:4.6f} {a3[2]:4.6f}\n')
    # Atomic structure
    print('Start writing hearder ', fn)
    start = time.time()
    for isite, site in enumerate(struct.sites):
        at_n     = at_n_lst[isite]
        #at_n     = 6 #Hard coded for C
        charge   = 0.0
        x, y, z  = (1/bohr2ang)*site.coords
        f.write(f'{at_n:6d} {charge:4.6f} {x:4.6f} {y:4.6f} {z:4.6f}\n')


    print('End writing hearder ', fn)

    print('Start writing grid data ', fn)
    # Data grid
    count = 0
    for ix in range(n1):
        for iy in range(n2):
            for iz in range(n3):
                f.write(f'  {data[ix][iy][iz]:.5E}')
                count += 1
                if count%6 == 0:
                    f.write('\n')

    end = time.time()
    f.close()
    print('End writing grid data ', fn)
    print('Writing time:', end-start)

    return None
