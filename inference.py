import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import nibabel as nib
from medpy import metric
from networks.Unet import UNet


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    print(image.shape)
    c, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape[1:]).astype(np.float32)
    cnt = np.zeros(image.shape[1:]).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[:,xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(test_patch,axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    return label_map, score_map

def test_all_case(net, image_list, num_classes=2, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None):
    total_metric = 0.0
    for ith,image_path in enumerate(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label[label==4] =3  # 标签4变为3
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        print(np.unique(prediction),np.unique(label))

        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))
        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + "%02d_pred.nii.gz"%(ith))
            # image只保留一个模态
            nib.save(nib.Nifti1Image(image[0].astype(np.float32), np.eye(4)), test_save_path + "%02d_img.nii.gz"%(ith))
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz"%(ith))
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    return avg_metric


if __name__ == '__main__':
    data_path = '/root/data/dataset/BraTS2021/dataset'
    test_save_path = 'predictions/unet/'
    save_mode_path = 'results/UNet.pth'
    net = UNet(in_channels=4,num_classes=4).cuda()
    net.load_state_dict(torch.load(save_mode_path)['model'])
    print("init weight from {}".format(save_mode_path))
    net.eval()
    with open(data_path + '/../test.txt', 'r') as f:
        image_list = [os.path.join(data_path, x.strip()) for x in f.readlines()]
    print(len(image_list))
    # print(image_list[0])
    # 滑动窗口法
    avg_metric = test_all_case(net, image_list, num_classes=4,
                                patch_size=(160,160,128), stride_xy=32, stride_z=16,
                                save_result=True,test_save_path=test_save_path)   
