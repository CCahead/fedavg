import socket
import pickle
serverIp = ("localhost",20001)

client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
client.connect(serverIp)
recvData = b""
condition = True
ROUND = "round"
METHOD= "method"
train_and_upload = "T&U"
wait_broadcast = "W&B"
while True:
    try:

        s_dict = dict()
        s_dict["a"] = "abc"
        s_dict["c"] = "abc"
        print(s_dict)
        pck = pickle.dumps(s_dict)
        client.sendall(pck) # 
        
        # *** synchronized ***
        while True:
            
            data = client.recv(1024)
            if data != b'':
                tmp = pickle.loads(data)
                # *** condition control***
                if tmp=="close":
                    client.close()
                    condition = False
                    break 
                pos_round_head = tmp.find(ROUND)
                if pos_round_head!= -1:
                    # round = tmp[pos_round+len(ROUND)+1:]
                    pos_round= pos_round_head+len(ROUND)
                    round = tmp[pos_round:pos_round+1]
                    print(tmp,":",round)
                    if round == "3": # reach max round
                        client.close()
                        condition = False
                        break 
                    pos_method_head = pos_round+1
                    pos_method = pos_method_head + len(METHOD)
                    method = tmp[pos_method:]
                    if method == wait_broadcast:
                        print(wait_broadcast)
                    elif method == train_and_upload:
                        print(train_and_upload)
                    else:
                        print("default")
            #   round{round}method{method}
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