import numpy as np
from pymatgen.io.cube import Cube
from write_cube_mod import write_cube

def main():
    vtot = Cube("./Vtot.cube")
    vbh  = Cube("./Vbh.cube")
    vb   = Cube("./Vb.cube")
    vxc_data  = vtot.data - vbh.data
    vh_data   = vbh.data  - vb.data
    #print(vtot.natoms)
    #print(vtot.origin)
    #print(vtot.NX)
    #print(vtot.X)
    #print(vtot.dX)
    #print(vtot.voxel_volume)
    #print(vtot.volume)
    #print(vtot.structure)
    #print(vtot.structure.volume)
    #print(vtot.data.shape)
    print(vxc_data.shape)
    write_cube('Vxc.cube',vxc_data,vtot.structure)
    write_cube('Vh.cube',vh_data,vtot.structure)


    return



if __name__=="__main__":
    main()
