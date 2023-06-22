import numpy as np
from pymatgen.io.cube import Cube

def main():
    vtot = Cube("./Vtot.cube")
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


    return



if __name__=="__main__":
    main()
