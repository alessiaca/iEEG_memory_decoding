from nex5file.reader import Reader
from nex5file.filedata import FileData
import h5py

filename = r"..\data_nix\Data_Subject_01_Session_01.h5"
h5 = h5py.File(filename,'r')

print("h")
