import numpy as np
np.set_printoptions(5,suppress=True,linewidth=2000,threshold=2000)
ha  = 27.2116 #eV

matel     = np.load('vxc_sub_matel.npy')*ha
matel_lin = np.load('vxc_sub_linear_matel.npy')*ha


print('np.max(np.abs(matel-matel_lin)): ', np.max(np.abs(matel-0.5*matel_lin)), ' eV')
ik = 0
print(matel[ik])
print()
print(matel_lin[ik])
print()
print(matel[ik]-0.5*matel_lin[ik])

mask =  (np.abs(matel_lin) < 1e-8)
ratio = np.abs(matel[~mask] / matel_lin[~mask])
print(np.max(ratio))
print(np.average(ratio))
print(np.min(ratio))

