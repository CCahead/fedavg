
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from threading import Thread 
import json
import socket

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

net = Net()

def clientListen(id,modelData):
    serverIp = ("localhost",20000)

    clientIp = ("localhost",20000+id) # bind expects a tuple!
    # clientPort = 20000+id
    print(f"client:{clientIp}")
    client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    client.bind(clientIp)
    count = 0
    try:
        modelData  = net.state_dict()
        
        client.connect(serverIp)
        # client.sendall(pck)
        with client,client.makefile('wb',buffering=0) as w:
            pickle.dump(modelData,w)
        
        # received = client.recv(1024)
        # print(received)
    except Exception as e:
        print(f"client:{clientIp},err:{e}")
    finally:
        client.close()
    print(f'Server Connection Closed.{clientIp}')

    
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


modelData  = net.state_dict()

# pck = pickle.dumps(modelData)

# client1 = Thread(target=clientListen,args=(1,pck))
# client2 = Thread(target=clientListen,args=(2,pck))
# client2.start()
# client1.start()

clientListen(1,modelData)