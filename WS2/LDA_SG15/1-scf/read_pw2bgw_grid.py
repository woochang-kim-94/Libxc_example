import h5py
import numpy as np

f = h5py.File('./WS2.save/charge-density.hdf5','r')
print(list(f.keys()))
print()
['MillerIndices', 'rhotot_g']
igvec = f['MillerIndices'][:]
rhog  = f['rhotot_g'][:]
ngm_g = f.attrs['ngm_g']
print(ngm_g)
print(igvec.dtype)
print(igvec.shape)
print(rhog.dtype)
print(rhog.shape)
print(rhog[ngm_g])

print(rhog[0]*1449.7341)
print(rhog[1]*1449.7341)
rhog = rhog[::2] + 1j*rhog[1::2]
print(rhog.shape)
np.save('rhog', rhog)
np.save('igvec', igvec)
##data = np.loadtxt('./VXC.txt', skiprows=13, dtype=np.complex128)
##print(data)
#f = open('./pw2bgw.save/VXC.txt', 'r')
#f.readline()
##nsf, ng_g, ntran, cell_symmetry, nat, ecutrho = np.int32(f.readline().split())
#f.readline()
##n1, n2, n3 = no.int32(f.readline().split())
#f.readline()
#f.readline()
#f.readline()
#f.readline()
#f.readline()
#f.readline()
#f.readline()
#ng = np.int32(f.readline().split()[0])
#gvec = np.int32(f.readline().split()).reshape((ng,3))
#print(gvec[:5])
#f.readline()
#ng = np.int32(f.readline().split()[0])
#data = f.readline().split()
#f.close()
#data = np.array([np.float64(eval(str)) for str in data]).view(np.complex128).reshape(ng)
#print(data[0:5])
#print(data.shape)
#np.save('vxc_gvec',gvec)
#np.save('vxc_pw2bgw',data)
