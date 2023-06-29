from netCDF4 import Dataset
import numpy as np
bohr2ang = 0.529177

def read_nc():
    def walktree(top):
        yield top.groups.values()
        for value in top.groups.values():
            yield from walktree(value)


    rootgrp = Dataset("WS2.nc", "r", format="NETCDF4")
    rootgrp
    for children in walktree(rootgrp['GRID']):
        for child in children:
            print(child)

    #print(rootgrp['GRID'])
    #Vt = rootgrp['GRID']['Vt'][:]
    #print('Vt')
    #print(Vt)
    #Vh = rootgrp['GRID']['Vh'][:]
    #print('Vh')
    #print(Vh)
    #print('Vh-Vt')
    #print(Vh-Vt)
    Rho = rootgrp['GRID']['Rho'][:]
    print('Rho')
    vol = 214.828427          # in Ang**3
    vol *= (1/bohr2ang)**3  # in Bohr**3
    nspin, nz, ny, nx = Rho.shape
    nfft = nx*ny*nz
    print(np.sum(Rho)*vol/nfft)
    RhoXC = rootgrp['GRID']['RhoXC'][:]
    print('RhoXC')
    print(np.sum(RhoXC)*vol/nfft)

#Rhogrid = Dataset("./GridFunc.nc","r", format="NETCDF4")
#print(Rhogrid)
#Rhogrid["gridfunc"]
#print(Rhogrid["gridfunc"][:])
#breakpoint()
read_nc()
