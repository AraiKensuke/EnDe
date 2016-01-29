import scipy.io as _sio

A = _sio.loadmat("../Spksrtd/bond_data_day4/Bon/bonripples04.mat")
A = _sio.loadmat("../Spksrtd/bond_data_day4/bond04/01-149.eeg")

A["ripples"][0, 3][0, 9]


#  ripple times in 30 electrodes
#  etrd 
A["ripples"][0, 3][0, etrd][0,0][0,0][0][:, 0]
