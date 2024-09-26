
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

ServerCount = 0
clientPool = dict()
clientBase = 20000
serverIp = ("localhost",20000)
# https://stackoverflow.com/questions/73673011/how-do-you-send-multiple-objects-using-pickle-using-socket
def recvall(clientSocket):
    
    # BUFF_SIZE = 4096
    # data = b""#  'x95x42x55x00usb.' 要再做一次转换？把str转换成hex？
    try:
        with clientSocket,clientSocket.makefile('rb') as r:
            while True:
                try:
                    data = pickle.load(r)
                except EOFError:
                    break
                print(data)
    #     while True:
            
    #         part = clientSocket.recv(BUFF_SIZE)
    #         data += part
    #         if len(part)<BUFF_SIZE:
    #             break # goto the end 
    except Exception as e:
        print(f"err during recv:{e}")
    return data # original is return data
def handle_client(clientSocket,addr,num):
    global ServerCount # [1,totalRound]
    global clientPool
    global clientBase
    try:
# Connection Estabilished with clientIp:('127.0.0.1', 20001)
        
        print(f"Connection Estabilished with clientIp:{addr}")
        port = addr[1] # addr is a tuple: (host,port)
        clientPool[port]=clientPool[port]+1 # update client Status
        Flag = True
        for i in range(0,num):
            clientId = clientBase+i+1
            clientRound = clientPool[clientId]
            
            pckData = recvall(clientSocket)
            try:
                # Error during unzip pickle: source code string cannot contain null bytes
                # modelData = pickle.loads(eval(pckData))
                print(pckData["conv1.weight"].shape+'\n')
            except Exception as e:
                print(f"Error during unzip pickle: {e}")
                # ServerCount+=1 #TODO test use 
            if(clientRound != ServerCount):
                Flag = False  # find out if this round is over. we have a dict to present.
            print(f'Connected with: {port}, round: {ServerCount}')
                            
            
        if(Flag):
            ServerCount+=1 #Update ServerCount! 这里有效嘛？如果我在线程内部更新全局变量的话，全局变量会改变吗？
            # 在server主线程看到的ServerCount会变吗？
    except Exception as e:
        print(f"Error happened during handle request{e}")
    # clientSocket.recv(1024)

def serverListen(num,totalRound):
    global ServerCount
    global clientPool
    global clientBase
    global serverIp
    print(f"server:{serverIp}")
    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.bind(serverIp)
    for i in range(0,num):
        clinetId = clientBase+i+1 
        clientPool[clinetId] = 0
    server.listen(10)

    try:
        while(True):
            print(f"server:{serverIp},round:{ServerCount}")
            #  lock
            if(ServerCount ==totalRound):
                break
            #  1. 发送一次普通文件，
            clientSocket, addr = server.accept()
            
            thread = Thread(target=handle_client, args=(clientSocket, addr,num))
            thread.start()
            
            # for i in range(0,num):
            #     clientId = clientBase+i+1
            #     print(f"clientPool:clientid:{clientId}:round:{clientPool[clientId]}")
            
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # close client socket (connection to the server)
        server.close()
        print("Server closed")


serverListen(1,1)