from get_data import train_set
from torch.utils.data import DataLoader as DataLoader
from net import Net
import torch
from torch.autograd import Variable
import torch.nn as nn

workers = 10                        # PyTorch读取数据线程数量
batch_size = 20                     # batch_size大小
lr = 0.01                         # 学习率
nepoch = 2

def train():
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    model = Net().cuda()
    model.train()

    criterion =nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    cnt = 0
    #time.time()
    for epoch in range(nepoch):
        for img , label in dataloader:
            img,label=Variable(img.cuda()),Variable(label.cuda())
            out  = model(img)
            loss = criterion(out,label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cnt += 1

            print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt * batch_size, loss / batch_size))

    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    train()