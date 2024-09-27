
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from threading import Condition, Thread 
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


# https://stackoverflow.com/questions/73673011/how-do-you-send-multiple-objects-using-pickle-using-socket

def serverTrain():
    return 

def recvall(connection):

    try:
        with connection.makefile('rb') as r:
            while True:    
                try:
                    data = pickle.load(r)
                except EOFError:
                    print("End of file reached.") #这个try catch就有点像多线程读写同一个list用except Empty控制条件
                    break
    except Exception as e:
        print(f"err during recv:{e}")
    # original is return data
    return data
def handle_client(connection,addr):
    try:
# Connection Estabilished with clientIp:('127.0.0.1', 20001)
        
        print(f"Connection Estabilished with clientIp:{addr}")
            
        pckData = recvall(connection)
        try:
            print("show the data:\n")
            print(pckData["conv1.weight"].shape)
        except Exception as e:
            print(f"Error during unzip pickle: {e}")

    except Exception as e:
        print(f"Error happened during handle request:{addr},{e}")
    return pckData

def serverListen(clientNum,totalRound):
    clientPool = []
    global serverIp
    print(f"server:{serverIp}")
    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.bind(serverIp)
    server.listen(10)
    round = 0
    step = 0 
    avgpckData = dict()
    try:
        while(True):
            print(step,round)
            #  lock
            if(round ==totalRound): # if reach maxRound, stop and broadcasting again
                
                break
            print(f"server:{serverIp},round:{round}")
            connection, addr = server.accept()
            clientPool.append(connection)
            pckData=handle_client(connection,addr)
            step +=1 # I received a client model!
            print(f"finished:{step}")
            
            # if(step==0): # initial avgModel 
            #     avgpckData = serverTrain()
            if(step==1): # initial avgModel 
                avgpckData = pckData.copy()
            elif(step>1):
                for k,v in avgpckData.items():
                    avgpckData[k] = avgpckData[k] + pckData[k]
            #  1. if a client send a model
            if(step==clientNum):
                step =0 # reset the connection , avg the model and broadcasting
                round +=1
                print(avgpckData["fc3.bias"])
                for k,v in avgpckData.items():
                    
                    avgpckData[k] = avgpckData[k]/clientNum # ave the weights
                print(avgpckData["fc3.bias"])

                for connection in clientPool:
                    with connection.makefile('wb',buffering=0) as w:
                        pickle.dump(avgpckData,w)
                        w.flush()
                    connection.shutdown(socket.SHUT_WR)
                    print(f"Avg model has sent")
                clientPool[:] = [] #reset clientPool
                        
            print(f"finished:{step}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # close client socket (connection to the server)
        server.close()
        print("Server closed")

net = Net()

serverIp = ("localhost",20000)
serverListen(2,2)