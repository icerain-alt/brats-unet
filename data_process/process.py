import pickle
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm


modalities = ('flair', 't1ce', 't1', 't2')

# train
train_set = {
        'root': '../../dataset/BraTS2021/data',
        'flist': 'train.txt',
        'has_label': True
        }


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


def process_f32b0(path, has_label=True):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    if has_label:
        label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')
    images = np.stack([np.array(nib_load(path + modal + '.nii.gz'), dtype='float32', order='C') for modal in modalities], -1)  # [240,240,155]

    # print(path.split('/')[-1])
    path = '../../dataset/BraTS2021/all/'+ path.split('/')[-1]
    output = path + 'data_f32b0.pkl'
    mask = images.sum(-1) > 0
    for k in range(4):

        x = images[..., k]  #
        y = x[mask]

        # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[..., k] = x

    with open(output, 'wb') as f:
        print(output)

        if has_label:
            pickle.dump((images, label), f)
        else:
            pickle.dump(images, f)

    if not has_label:
        return


def doit(dset):
    root, has_label = dset['root'], dset['has_label']
    file_list = os.path.join(root, dset['flist'])
    subjects = open(file_list).read().splitlines()
    # print(subjects)
    names = ['BraTS2021_' + sub for sub in subjects]
    paths = [os.path.join(root, name, name + '_') for name in names]

    for path in tqdm(paths):
        process_f32b0(path, has_label)
        # break


if __name__ == '__main__':
    doit(train_set)
    # doit(valid_set)
    # doit(test_set)

