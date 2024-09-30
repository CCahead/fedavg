import pickle

import socket
import time
serverIp = ("localhost",20001)
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(serverIp)
server.listen(10)
buf = 1024
modelData = b""
msg_cls = "close"
msg_upload = "upload"
msg_init = "init"

ROUND = "round"
METHOD= "method"
train_and_upload = "T&U"
wait_broadcast = "W&B"
connection, addr = server.accept()
round = 0
while True:
    try:
        data = connection.recv(buf)
        tmp = pickle.loads(data) 
        # two possibles: 1. it's part of dict(model data) 2. it's condition
        if tmp == msg_cls:
            print("closing!")
            connection.close()
            break
        elif tmp ==msg_upload:
            connection.send(pickle.dumps("plz waiting"))
        elif tmp == msg_init:
            connection.send(pickle.dumps("plz waiting Im training data!"))
        elif tmp != "":
            modelData += data 
            
            # time.sleep(3)
            if len(data)<buf:
                round +=1
                print(f"model received:{pickle.loads(modelData)}")
                if round == 1:
                    method = train_and_upload
                    msg = pickle.dumps(f"round{round}method{method}")
                    connection.send(msg)
                if round == 2:
                    method = wait_broadcast
                    msg = pickle.dumps(f"round{round}method{method}")
                    connection.send(msg)
                if round ==3:
                    connection.send(pickle.dumps(f"round{round}")) # 之前没这一条，好像就会死锁？
                    print("reach max round,closing!")
                    connection.close()
                    break
            #  加入之后就好了！好像是的！一边发送完消息他其实是在等待这边的通信？
            # in this condition, only data being fully received, server send response.
            #  Server   Client 
            #   recv<----send
            #   send---->recv 

    except socket.error as e:
        print(e)
        connection.close()
        break
print("finish")
    
print(modelData[4096:])
# model received:1024
# model received:2048
# model received:3072
# model received:4096
# model received:4109
# closing!
# finish
# b'data finished'