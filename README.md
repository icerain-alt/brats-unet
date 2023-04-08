# brats-unet
**UNet for brain tumor segmentation**

![brats](C:\Users\Yubin\Desktop\brats.jpg)

BraTS是**MICCAI**所有比赛中历史最悠久的，到2021年已经连续举办了10年，参赛人数众多，是学习医学图像分割最前沿的平台之一。

![image-20230408093340483](C:\Users\Yubin\AppData\Roaming\Typora\typora-user-images\image-20230408093340483.png)

## 数据准备

**数据集下载地址**：

1.官网：[BraTS 2021 Challenge](https://www.synapse.org/#!Synapse:syn25829067/wiki/)   需要注册和申请（包括训练集和验证集）

2.Kaggle：[BRaTS 2021 Task 1 Dataset ](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)  建议在kaggle上下载，数据集与官网一致（不包括验证集）

下载数据集，解压后如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/a526dd654e9b461da4ab79d96be8b8e9.png#pic_center)


每个病例包含四种模态的MRI图像和分割标签，结构如下：

```
BraTS2021_00000
├── BraTS2021_00000_flair.nii.gz
├── BraTS2021_00000_seg.nii.gz
├── BraTS2021_00000_t1ce.nii.gz
├── BraTS2021_00000_t1.nii.gz
└── BraTS2021_00000_t2.nii.gz
```

四种模态数据：**flair, t1ce, t1, t2**，每个模态的数据大小都为 240 x 240 x 155，且共享分割标签。

​	分割标签：[0, 1, 2, 4] 

- label0：背景（background）
- label1：坏疽（NT, necrotic tumor core）
- label2：浮肿区域（ED,peritumoral edema）
- label4：增强肿瘤区域（ET,enhancing tumor）

> 建议使用3D Slicer查看图像和标签，直观的了解一下自己要用的数据集。

## 数据处理

```bash
python data/process.py
```

将四种模态的图像合并为一个4D图像（C x H x W x D , C=4），并且和分割标签一起保存为一个`.h5`文件。数据保存在 mri_norm2.h5 文件中，每个 `.h5`文件 是一个字典，字典的键为 image 和 label ，值为对应的数组。

![在这里插入图片描述](https://img-blog.csdnimg.cn/c6785e4987ef4b1f93a618f7cfd7186d.png#pic_center)

***

将数据集按照 8:1:1随机划分为训练集、验证集和测试集，将划分后的数据名保存为`.txt`文件

```bash
python data/split_data.py
```

## 开始训练

记得修改路径

```bash
python train.py
```

损失曲线：

![在这里插入图片描述](https://img-blog.csdnimg.cn/574e3913f02e4a548f22ee2c32445dbc.png#pic_center)

## 滑动推理

原始数据的尺寸与输入网络的尺寸不同，常用的方法是用一个滑动窗口，遍历原始图像的全部区域进行推理

`inference.py`

```python
def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    # print(image.shape)
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
```

设置权重路径和测试集路径后，执行命令

```bash
python inference.py
```

***

> batch_size=1是可以在`2080Ti`上运行的，超过2就需要更大内存的显卡了。
