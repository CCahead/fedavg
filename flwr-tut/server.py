
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

import torch.optim as optim
from model import Net




# https://stackoverflow.com/questions/73673011/how-do-you-send-multiple-objects-using-pickle-using-socket


    # pre-trained model 
def train(epoches):
    criterion = nn.CrossEntropyLoss()
    net = Net()
    # net.load_state_dict(torch.load("avg.pth")) #test use

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(epoches):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0

    return net.state_dict()

def test(round,modelData):
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    net = Net()
    net.load_state_dict(state_dict=modelData)
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(4)))
    correct = 0
    total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Round:{round},acc: {100 * correct // total} %')
        
        
def recvall(connection):
    modelData = b''
    try:
       
        while True:
            packet = connection.recv(BUFFER)
            if packet!=b'':
                modelData += packet
                if len(packet)<BUFFER:
                    
                    break

    except Exception as e:
        print(f"err during recv:{e}")
    # original is return data
    num,id,modelData = pickle.loads(modelData)
    return num,id,modelData
def handle_client(connection):
    try:
# Connection Estabilished with clientIp:('127.0.0.1', 20001)
        
        print(f"Connection Estabilished") #  with clientIp:{addr}
        try:    
            num,ID,modelData = recvall(connection)
        except Exception as e:
            print(f"Error during unzip pickle: {e}")

    except Exception as e:
        print(f"Error happened during handle request:{e}") #{addr}
    return num,ID,modelData
def broadcast(clientPool,round,modelData):
    msg = []
    for connection in clientPool:
        # in first round, server sent total round to client to control transmission 
        if round == INIT_ROUND:
           msg = [0-NUM_ROUND,epoches,modelData]
        else:
            msg = [round,epoches,modelData]
        msg = pickle.dumps(msg)
        connection.sendall(msg)
        r,_,_ = pickle.loads(msg)
        print(f"Successfully Broadcasting!Round Check:{r}")
    
def recvModel(clientPool):
   
    modelDataList = []
    try:
        for connection in clientPool:
            num,id,modelData = handle_client(connection)
            modelDataList.append(modelData)
            print(f"Model received:clientId:{id},client round:{num}") # this could be parallel! now it's sequential! 
        avgModelData = modelDataList[0]
        for modelData in modelDataList:
            for k in modelData:
                avgModelData[k] += modelData[k] # avg the model
        for k in avgModelData:
            avgModelData[k] /=len(clientPool)
        # finish receive procedure
    except Exception as e:
        print(f"handle model err:{e}")
    return avgModelData
    
def serverListen(clientNum):
    clientPool = []
    global serverIp
    print(f"server:{serverIp}")
    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.bind(serverIp)
    server.listen(10)
    round = INIT_ROUND # 
    Flag = False
    modelData = dict()
    try:
        while(True):
            # init connection

            if(round >NUM_ROUND): # if reach maxRound, stop and broadcasting again
                try:
                    for conn in clientPool:
                        conn.close()
                    break
                except Exception as e:
                    print(f"close conn pool err: {e}")

            if len(clientPool)<clientNum:
                connection, addr = server.accept()
                clientPool.append(connection)
            # init model
            if Flag == False and len(clientPool)==clientNum:
                modelData = train(epoches = epoches)
                Flag = True
            
            print(f"server:{serverIp},round:{round}")
            
            if Flag == True:
                broadcast(clientPool,round,modelData)
                modelData = recvModel(clientPool)
                test(round,modelData)
                round +=1
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # close client socket (connection to the server)
        server.close()
        torch.save(modelData,"avg.pth")
        print("Server closed")


if __name__ == '__main__':
    # Setup DataLoader
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
    serverIp = ("localhost",20000)
    BUFFER = 4096 # recv buffer size
    NUM_ROUND = 3 # stands for total transmission round
    INIT_ROUND = 1 # stands for init round 
    CLIENTPOOL = 2 # how many client could connect
    epoches = 1 # train epoches/ client epoches per transmission
    
    serverListen(CLIENTPOOL)






