from matplotlib import pyplot as plt
import numpy as np
#from plot_point import plot_kpts_sc

def write_kpts_sc(fn=None, grid_spec=None):
    fn = 'kpts.txt'
    grid_spec = [6, 6, 1]
    nkpts = int(grid_spec[0]*grid_spec[1]*grid_spec[2])

    kpt_sc = np.zeros((nkpts,3))
    kx_points = np.linspace(0, 1, endpoint=False, num=grid_spec[0])
    ky_points = np.linspace(0, 1, endpoint=False, num=grid_spec[1])
    kz_points = np.linspace(0, 1, endpoint=False, num=grid_spec[2])

    f = open(fn,'w')
    f.write('K_POINTS crystal\n')
    f.write('{0:d}\n'.format(nkpts))

    for ikx, kx in enumerate(kx_points):
        for iky, ky in enumerate(ky_points):
            for ikz, kz in enumerate(kz_points):
                kpt_sc[ikx+iky+ikz,:] = np.array([kx,ky,kz])
                f.write('{0:1.10f}  {1:1.10f}  {2:1.10f}  1.0\n'.format(kx, ky, kz))
    f.close()

    return kpt_sc

write_kpts_sc()
