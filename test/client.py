import socket

serverIp = ("localhost",20001)

client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
client.connect(serverIp)
recvData = b''
while True:
    try:
        # user_input = input("Please enter something: ")

        # Print the input received
        with open("text.txt","r") as file:
            a = file.read()
        user_input = a
        print(f'You entered: {len(user_input)}')
        client.sendall(user_input.encode('utf-8')) # 
        if(user_input=="close"):
            client.close()
            break
        user_input = "data finished"
        
        client.sendall(user_input.encode('utf-8')) # 
        # *** sychronized ***
        while True:
            
            data = client.recv(1024)
            if data != b'':
                recvData += data
                if len(data)<1024:
                    print(f"server msg:{recvData}")
                    break
        # *** sychronized ***
        # client.send(b"close")
       
        # client.close()
        
    except socket.error as e:
        print(f"{e}")
        client.close()
        break

print("finish!")


# 4096
# You entered: 4096
# server msg:b''
# finish!