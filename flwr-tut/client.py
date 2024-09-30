
import pickle
import random
import time
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
import torch.optim as optim
from model import Net

def pull(client):
    modelData = dict()
    data = b''
    global BUFFER
    
    try:
        
            # https://stackoverflow.com/questions/44637809/python-3-6-socket-pickle-data-was-truncated
            # before reading this I use the same way like server reading data: use makefile
            # but it would cause block and broken file condition. So, I pivot my strategy,use this again.
        while True:
            packet = client.recv(BUFFER)
            if packet != b"":
                data += packet
                if len(packet)<BUFFER:
                    break
            #    ROUND0EPOCHES1
            # roundb'\x00',epochesb'R' error
        round,epoches,modelData = pickle.loads(data)
    except Exception as e:
        print(f"client recv err:{e}")
    
    return round,epoches,modelData
def push(client,num,modelData):
    try:
        print(f"round:{num},Client:{ID} model Sent!Waiting for avg Model!")
        msg = [num,ID,modelData]
        msg = pickle.dumps(msg)
        client.sendall(msg)
    except Exception as e:
        print(f"client:err:{e}")
    return 

def connection():
    serverIp = ("localhost",20000)
    # clientIp = ("localhost",20010+id) # bind expects a tuple!
    client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    client.connect(serverIp)
    return client

def train(modelData,epoches):
    model = Net()
    model.load_state_dict(state_dict=modelData)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(epoches):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0

    return model.state_dict()


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
# *** init ***
if __name__ == '__main__':
    BUFFER = 4096
    CLIENT_HEAD = "CLIENT"
    METHOD_HEAD = "METHOD"
    ROUND_HEAD = "ROUND"
    PUSH = "PUSH"
    PULL = "PULL"
    FINISHED = "FINISHED"
    EPOCHES_HEAD = "EPOCHES"
    
    EPOCHES_LENGTH = 1
    ROUND_LENGTH = 1
    # NUM_ROUND = 2
    INIT_ROUND = 1
    ID = random.randint(1, 5)
    count = 1
    epoches = 1
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # batch_size = 4  # to simulate different data distribution,it could varies
    batch_size = 2**(random.randint(2,5)) # [2,32]

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
    client = connection()
    
    
    
    # *** init ***
    Init = False
    while True:
        Round,Epoches,modelData = pull(client)
        if Init == False:
            Round =  abs(Round)
            NUM_ROUND = Round
            Round = INIT_ROUND
            Init = True
        print(f"Total round:{NUM_ROUND},client round:{Round},Epoches:{Epoches},BatchSize:{batch_size}")
        
        modelData = train(modelData,Epoches) 
        push(client,count,modelData)
        test(round,modelData)
        if NUM_ROUND==Round:
            # msg = "finished"
            # msg = pickle.dumps(msg)
            # client.send(msg)
            client.close()
            break
    print("finished")