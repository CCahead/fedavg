import socket
import pickle
serverIp = ("localhost",20001)

client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
client.connect(serverIp)
recvData = b""
condition = True
ROUND = "round"
while True:
    try:

        s_dict = dict()
        s_dict["a"] = "abc"
        s_dict["c"] = "abc"
        print(s_dict)
        pck = pickle.dumps(s_dict)
        client.sendall(pck) # 
        
        # *** sychronized ***
        while True:
            
            data = client.recv(1024)
            if data != b'':
                tmp = pickle.loads(data)
                # *** condition control***
                if tmp=="close":
                    client.close()
                    condition = False
                    break 
                pos_round = tmp.find(ROUND)
                if pos_round!= -1:
                    # round = tmp[pos_round+len(ROUND)+1:]
                    round = tmp[pos_round+len(ROUND):]
                    print(tmp,":",round)
                    if round == "3": # reach max round
                        client.close()
                        condition = False
                        break 
                # *** condition control***
                # *** recv data ***
                
                # recvData += data
                recvData += data

                if len(data)<1024:
                    print(f"server msg:{recvData}")
                    recvData = b'' #reset 
                    break
                # *** recv data ***
        if condition == False:
            break # connection switch
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