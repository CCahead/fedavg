
import socket

serverIp = ("localhost",20001)
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(serverIp)
server.listen(10)
buf = 1024
modelData = b""
msg_cls = b"close"
msg_upload = b"upload"
msg_init = b"init"
connection, addr = server.accept()

while True:
    try:
        # print("wait...\n")
        data = connection.recv(buf)
        # print("recv...\n")
        if data == msg_cls:
            print("closing!")
            connection.close()
            break
        elif data ==msg_upload:
            connection.send(b"plz waiting")
        elif data == msg_init:
            connection.send(b"plz waiting Im training data!")
        elif data != b"":
            modelData += data
            print(f"model received:{len(modelData)}")
            connection.send(b"data received") # 之前没这一条，好像就会死锁？
            #  加入之后就好了！好像是的！一边发送完消息他其实是在等待这边的通信？
        
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