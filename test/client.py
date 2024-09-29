import socket

serverIp = ("localhost",20001)

client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
client.connect(serverIp)

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
        
        data = client.recv(1024)
        client.send(b"close")
        data = client.recv(1024)
        print(f"server msg:{data}")
        client.close()
        break
    except socket.error as e:
        print(f"{e}")
        client.close()
        break

print("finish!")
    