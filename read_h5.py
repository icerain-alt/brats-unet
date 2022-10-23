import h5py
import numpy as np


<<<<<<< HEAD
p = '/data/omnisky/postgraduate/Yb/data_set/BraTS2021/all/BraTS2021_00000_mri_norm2.h5'
=======
p = '/***/BraTS2021/dataset/BraTS2021_00000_mri_norm2.h5'
>>>>>>> 465a5c0d4bc21b38d6085bff23b53bda8dcf9a9a
h5f = h5py.File(p, 'r')
image = h5f['image'][:]
label = h5f['label'][:]

print('image shape:',image.shape,'\t','label shape',label.shape)
print('label set:',np.unique(label))