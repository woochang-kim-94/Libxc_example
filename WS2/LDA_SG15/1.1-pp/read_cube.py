import numpy as np
from pymatgen.io.cube import Cube
bohr2ang = 0.529177

def main():
    vtot = Cube("./Rho.cube")
    print(vtot.natoms)
    print(vtot.origin)
    print(vtot.NX)
    print(vtot.X)
    print(vtot.dX)
    print(vtot.voxel_volume)
    print(vtot.volume)
    print(vtot.structure)
    print(vtot.structure.volume)
    print(vtot.data.shape)
    print(np.sum(vtot.data)*vtot.voxel_volume/(bohr2ang**3))


    return



if __name__=="__main__":
    main()
