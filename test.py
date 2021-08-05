from get_data import simple_transform
from net import Net
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

import cv2
from PIL import Image



def test(imge):

    PIL_image = Image.fromarray(imge)
    img_data = simple_transform(PIL_image)

    model_file = 'model.pth'  # 模型保存路径
    model = Net().cuda()                                          # 实例化一个网络
    model.load_state_dict(torch.load(model_file))       # 加载训练好的模型参数
    model.eval()                                        # 设定为评估模式，即计算过程中不要dropout


    img = Variable(img_data).cuda()                                  # 将数据放置在PyTorch的Variable节点中，并送入GPU中作为网络计算起点
    img = img.unsqueeze(0)
    out = model(img)                                          # 网路前向计算，输出图片属于猫或狗的概率，第一列维猫的概率，第二列为狗的概率
    #print(out)
    out = F.softmax(out, dim=1)       # 采用SoftMax方法将输出的2个输出值调整至[0.0, 1.0],两者和为1
    print(out)                      # 输出该图像属于猫或狗的概率
    if out[0, 0] > out[0, 1]:                   # 猫的概率大于狗
        print('the image is a cat')
        imge=cv2.putText(imge, 'cat', (100, 100), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, (100, 255, 255), 2)
        cv2.imshow("2",imge)
        cv2.waitKey(0)
    else:                                       # 猫的概率小于狗
        print('the image is a dog')
        imge=cv2.putText(imge, 'dog', (100, 100), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, (100, 255, 255), 2)
        cv2.imshow("2",imge)
        cv2.waitKey(0)
if __name__ == '__main__':
    imge = cv2.imread("dataset/test/cat/52.jpg")
    test(imge)
