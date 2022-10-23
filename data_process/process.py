import os
import numpy as np
import h5py
import nibabel as nib
from tqdm import tqdm


modalities = ('flair', 't1ce', 't1', 't2')

# train
train_set = {
<<<<<<< HEAD
        'root': '/data/omnisky/postgraduate/Yb/data_set/BraTS2021/data',  # 四个模态数据所在地址
        'out': '/data/omnisky/postgraduate/Yb/data_set/BraTS2021/dataset',  # 预处理输出地址
=======
        'root': '/***/dataset/BraTS2021/data',  # 四个模态数据所在地址
        'out': '/***/dataset/BraTS2021/dataset',  # 预处理输出地址
>>>>>>> 465a5c0d4bc21b38d6085bff23b53bda8dcf9a9a
        'flist': 'train.txt',  # 训练集名单
        }


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


def process_h5(path, out_path):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')
    images = np.stack([np.array(nib_load(path + modal + '.nii.gz'), dtype='float32', order='C') for modal in modalities], 0)  # [240,240,155]
    case_name = path.split('/')[-1]
    # case_name = os.path.split(path)[-1]  # windows路径与linux不同
<<<<<<< HEAD
    
=======

>>>>>>> 465a5c0d4bc21b38d6085bff23b53bda8dcf9a9a
    path = os.path.join(out_path,case_name)  # 输出地址
    output = path + 'mri_norm2.h5'
    mask = images.sum(0) > 0
    for k in range(4):

        x = images[k,...]  #
        y = x[mask]

        # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[k,...] = x
    print(case_name,images.shape,label.shape)
<<<<<<< HEAD
    # f = h5py.File(output, 'w')
    # f.create_dataset('image', data=images, compression="gzip")
    # f.create_dataset('label', data=label, compression="gzip")
    # f.close()
=======
    f = h5py.File(output, 'w')
    f.create_dataset('image', data=images, compression="gzip")
    f.create_dataset('label', data=label, compression="gzip")
    f.close()
>>>>>>> 465a5c0d4bc21b38d6085bff23b53bda8dcf9a9a


def doit(dset):
    root, out_path = dset['root'], dset['out']
    file_list = os.path.join(root, dset['flist'])
    subjects = open(file_list).read().splitlines()
    names = ['BraTS2021_' + sub for sub in subjects]
    paths = [os.path.join(root, name, name + '_') for name in names]

    for path in tqdm(paths):
        process_h5(path, out_path)
<<<<<<< HEAD
        break
=======
>>>>>>> 465a5c0d4bc21b38d6085bff23b53bda8dcf9a9a


if __name__ == '__main__':
    doit(train_set)

