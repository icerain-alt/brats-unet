import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 3)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class Random_Crop(object):
    """
    纵向随机裁剪
    """
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        D = random.randint(0, 155 - 128)

        # print(image.shape)
        # print(label.shape)
        # [240,240,155] -> [160,160,128]
        image = image[:,40:200,40:200, D: D + 128]
        label = label[40:200,40:200, D: D + 128]

        return {'image': image, 'label': label}


class Random_rotate(object):
    # 简单起见，随机旋转90,180,270度
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        angles = [90,180,270]
        index = random.randint(0,2)  # 0,1,2

        # angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angles[index], axes=(1, 2), reshape=False)
        label = ndimage.rotate(label, angles[index], axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}


def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample


class GaussianNoise(object):
    def __init__(self, noise_variance=(0, 0.1), p=0.5):
        self.prob = p
        self.noise_variance = noise_variance

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if np.random.uniform() < self.prob:
            image = augment_gaussian_noise(image, self.noise_variance)
        return {'image': image, 'label': label}


def augment_contrast(data_sample, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True):
    if not per_channel:
        mn = data_sample.mean()
        if preserve_range:
            minm = data_sample.min()
            maxm = data_sample.max()
        if np.random.random() < 0.5 and contrast_range[0] < 1:
            factor = np.random.uniform(contrast_range[0], 1)
        else:
            factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
        data_sample = (data_sample - mn) * factor + mn
        if preserve_range:
            data_sample[data_sample < minm] = minm
            data_sample[data_sample > maxm] = maxm
    else:
        for c in range(data_sample.shape[0]):
            mn = data_sample[c].mean()
            if preserve_range:
                minm = data_sample[c].min()
                maxm = data_sample[c].max()
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
            data_sample[c] = (data_sample[c] - mn) * factor + mn
            if preserve_range:
                data_sample[c][data_sample[c] < minm] = minm
                data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample


class ContrastAugmentationTransform(object):
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True,p_per_sample=1.):
        """
        Augments the contrast of data
        :param contrast_range: range from which to sample a random contrast that is applied to the data. If
        one value is smaller and one is larger than 1, half of the contrast modifiers will be >1 and the other half <1
        (in the inverval that was specified)
        :param preserve_range: if True then the intensity values after contrast augmentation will be cropped to min and
        max values of the data before augmentation.
        :param per_channel: whether to use the same contrast modifier for all color channels or a separate one for each
        channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        for b in range(len(image)):
            if np.random.uniform() < self.p_per_sample:
                image[b] = augment_contrast(image[b], contrast_range=self.contrast_range,
                                            preserve_range=self.preserve_range, per_channel=self.per_channel)
        return {'image': image, 'label': label}


def augment_brightness_additive(data_sample, mu:float, sigma:float , per_channel:bool=True, p_per_channel:float=1.):
    """
    data_sample must have shape (c, x, y(, z)))
    :param data_sample:
    :param mu:
    :param sigma:
    :param per_channel:
    :param p_per_channel:
    :return:
    """
    if not per_channel:
        rnd_nb = np.random.normal(mu, sigma)
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                data_sample[c] += rnd_nb
    else:
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                rnd_nb = np.random.normal(mu, sigma)
                data_sample[c] += rnd_nb
    return data_sample


class BrightnessTransform(object):
    def __init__(self, mu, sigma, per_channel=True, p_per_sample=1., p_per_channel=1.):
        """
        Augments the brightness of data. Additive brightness is sampled from Gaussian distribution with mu and sigma
        :param mu: mean of the Gaussian distribution to sample the added brightness from
        :param sigma: standard deviation of the Gaussian distribution to sample the added brightness from
        :param per_channel: whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.mu = mu
        self.sigma = sigma
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, sample):
        data, label = sample['image'], sample['label']

        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                data[b] = augment_brightness_additive(data[b], self.mu, self.sigma, self.per_channel,
                                                      p_per_channel=self.p_per_channel)

        return {'image': data, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}


def transform(sample):
    trans = transforms.Compose([
        # Pad(),
        Random_rotate(),
        Random_Crop(),
        Random_Flip(),
        GaussianNoise(p=0.1),
        ContrastAugmentationTransform(p_per_sample=0.15),
        BrightnessTransform(0,0.1,True,0.15,0.5),
        ToTensor()
    ])

    return trans(sample)


def transform_valid(sample):
    trans = transforms.Compose([
        Random_Crop(),  # transformer训练的时候使用，训练集和验证集尺寸要一样
        # Pad(),
        # Crop(),
        # MaxMinNormalization(),
        ToTensor()
    ])

    return trans(sample)


class BraTS(Dataset):
    def __init__(self,file_path,data_path="../BraTS2021/data", mode='train'):
        with open(file_path, 'r') as f:
            self.paths = [os.path.join(data_path, x.strip()) for x in f.readlines()]
        self.mode = mode

    def __getitem__(self, item):
        path = self.paths[item]
        if self.mode == 'train':
            image, label = pkload(path)
            # [h,w,s,d] -> [d,h,w,s]
            image = image.transpose(3, 0, 1, 2)
            sample = {'image': image, 'label': label}
            sample = transform(sample)
            return sample['image'], sample['label']
        elif self.mode == 'valid':
            image, label = pkload(path)
            image = image.transpose(3, 0, 1, 2)
            sample = {'image': image, 'label': label}
            sample = transform_valid(sample)
            return sample['image'], sample['label']
        else:
            image = pkload(path)
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
            image = torch.from_numpy(image).float()
            return image

    def __len__(self):
        return len(self.paths)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


if __name__ == '__main__':
    data_path = "../../dataset/BraTS2021/dataset"
    test_txt = "../../dataset/BraTS2021/test.txt"
    test_set = BraTS(test_txt,data_path,'train')
    # print(test_set.paths)
    d1 = test_set[0]
    image,label = d1
    print(image.shape)
    print(label.shape)
    print(np.unique(label))
