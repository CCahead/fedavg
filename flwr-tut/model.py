
import torch
import torch.nn as nn
import torch.nn.functional as F





class Net(nn.Module):
    def __init__(self):
        # in_channels (int): Number of channels in the input image
        # out_channels (int): Number of channels produced by the convolution
        # kernel_size (int or tuple): Size of the convolving kernel
        # stride (int or tuple, optional): Stride of the convolution. Default: 1
        # padding (int, tuple or str, optional): Padding added to all four sides of
        #     the input. Default: 0
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 根据计算调整维度
        self.fc1 = nn.Linear(16 * 5 * 5, 120) #这里
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # print("After conv1 and pool:", x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print("After conv2 and pool:", x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print("After flatten:", x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    