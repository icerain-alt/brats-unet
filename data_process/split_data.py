import os
from sklearn.model_selection import train_test_split

# 预处理输出地址
<<<<<<< HEAD
data_path = "/data/omnisky/postgraduate/Yb/data_set/BraTS2021/dataset"
=======
data_path = "/***/data_set/BraTS2021/dataset"
>>>>>>> 465a5c0d4bc21b38d6085bff23b53bda8dcf9a9a
train_and_test_ids = os.listdir(data_path)

train_ids, val_test_ids = train_test_split(train_and_test_ids, test_size=0.2,random_state=21)
val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5,random_state=21)
print("Using {} images for training, {} images for validation, {} images for testing.".format(len(train_ids),len(val_ids),len(test_ids)))

train_ids.sort()
val_ids.sort()
test_ids.sort()
<<<<<<< HEAD
with open('/data/omnisky/postgraduate/Yb/data_set/BraTS2021/train.txt','w') as f:
    f.write('\n'.join(train_ids))

with open('/data/omnisky/postgraduate/Yb/data_set/BraTS2021/valid.txt','w') as f:
    f.write('\n'.join(val_ids))

with open('/data/omnisky/postgraduate/Yb/data_set/BraTS2021/test.txt','w') as f:
=======
with open('/***/data_set/BraTS2021/train.txt','w') as f:
    f.write('\n'.join(train_ids))

with open('/***/data_set/BraTS2021/valid.txt','w') as f:
    f.write('\n'.join(val_ids))

with open('/***/data_set/BraTS2021/test.txt','w') as f:
>>>>>>> 465a5c0d4bc21b38d6085bff23b53bda8dcf9a9a
    f.write('\n'.join(test_ids))