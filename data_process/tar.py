import os
import tarfile
def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

# 解压文件
path1 = "../../dataset/BraTS2021/data"
mk_dir(path1)
tar = tarfile.open("../../dataset/BraTS2021/BraTS2021_Training_Data.tar")
tar.extractall(path1)
tar.close()

path2 = "../../dataset/BraTS2021/data/BraTS2021_00495"
mk_dir(path2)
tar = tarfile.open("../../dataset/BraTS2021/BraTS2021_00495.tar")
tar.extractall(path2)
tar.close()

path3 = "../../dataset/BraTS2021/data/BraTS2021_00621"
mk_dir(path3)
tar = tarfile.open("../../dataset/BraTS2021/BraTS2021_00621.tar")
tar.extractall(path3)
tar.close()

