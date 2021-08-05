#用于提取数据集数据

#ImageFolder用法，可以直接从分类的文件夹里获取标签和对应的图片
# import torchvision.datasets as dset
# dataset = dset.ImageFolder('./dataset/train/') #没有transform，先看看取得的原始图像数据
# print(dataset.classes)  #根据分的文件夹的名字来确定的类别
# print(dataset.class_to_idx) #按顺序为文件夹中得到的图片的路径以及其类别


##建立图像和标签读取通道
#1.将所有图像进行同等大小尺度处理。
#2.利用数据集的均值和标准差把数据集归一化处理。
#3.转换成可输入网络的tensors
from torchvision import transforms
from torchvision import  datasets
#首先，设置转换函数
train_path ="dataset/train"
test_path ="dataset/test"
simple_transform = transforms.Compose([transforms.Scale((200,200)),transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
train_set = datasets.ImageFolder(train_path,simple_transform)
test_set = datasets.ImageFolder(test_path,simple_transform)
# print(train.classes)
# print(train.class_to_idx)
