from netCDF4 import Dataset
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

    print(rootgrp['GRID'])
    Vt = rootgrp['GRID']['Vt'][:]
    print('Vt')
    print(Vt)
    Vh = rootgrp['GRID']['Vh'][:]
    print('Vh')
    print(Vh)
    print('Vh-Vt')
    print(Vh-Vt)
    Rho = rootgrp['GRID']['Rho'][:]
    print('Rho')
    print(Rho)

#Rhogrid = Dataset("./GridFunc.nc","r", format="NETCDF4")
#print(Rhogrid)
#Rhogrid["gridfunc"]
#print(Rhogrid["gridfunc"][:])
read_nc()
breakpoint()
