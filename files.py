import numpy as np
import h5py as h5

def hdf5Save(arr, filename, name="object"):
    f = h5.File(filename, "w")
    dset = f.create_dataset(name, data=arr)
    dset[...] = arr
    f.close()
    
def hdf5Load(filename, name="object"):
    f = h5.File(filename, "r")
    arr = f[name][...]
    f.close()
    return arr
