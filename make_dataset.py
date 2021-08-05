import os

import random
import shutil

fileDir =  'D:\图像分类\kagglecatsanddogs_3367a\PetImages\Cat/'   #设置原有文件路径
dog_train_Dir = 'dataset/train/cat/'   #设置数据集路径文件
dog_test_Dir = 'dataset/test/cat/'
rate = 1  #设置复制比例

pathDir = os.listdir(fileDir)  #scan
filenumber = len(pathDir)
picknumber = int(filenumber * rate)
print('total {} pictures'.format(filenumber))
print('moved {} pictures to {}'.format(picknumber, dog_train_Dir))#
#
sample = random.sample(pathDir, picknumber)#进行移动
for name in sample:
    shutil.move(fileDir + name, dog_train_Dir + name)#设置目标文件夹
    print(name)
print('succeed moved {} pictures from {} to {}'.format(picknumber, fileDir, dog_train_Dir))
