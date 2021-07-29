from torch import nn
import numpy as np
import torch
#import matplotlib.pyplot as plt
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        
        y = self.fc(y).view(b, c, 1, 1)
        return y.expand_as(x)
if __name__ == '__main__':
    input = np.random.randint(0,10,size=(1,512,32,32))
    # input = np.random.randn(1,512,32,32)
    input = torch.FloatTensor(input)

    SE = SEModule(512)
    weight = SE(input).squeeze()
    y = weight.reshape(512, -1)[:,0]
    print(y.max(),y.min())
    x_axis = range(len(y))

    # fig = plt.figure()
    # plt.xlabel('pixel index')
    # plt.ylabel('value')
    # plt.scatter(x_axis,y.detach().cpu().numpy(),s=1)
    # plt.show()
