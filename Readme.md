# RUN GUIDE
in server.py main function:
```python
    NUM_ROUND = 3 # stands for total transmission round
    INIT_ROUND = 1 # stands for init round 
    CLIENTPOOL = 2 # how many client could connect
    epoches = 1 # train epoches/ client epoches per transmission
   # change this variables to set the training procedure.
   # NUM_ROUND: max transmission round 
   # CLIENTPOOL: max client connection
   # epoches: max epoches client would train per transmission
   #
```
## NOTE 
in server side, the default batch size is 32. batch size in client size is random picked from 4 to 32.
This code contains three parts: model,client and server. 
Training Procedure per transmission round:

Client   Server
pull<=====broadcast
push=====>recvModel
--------------------



# tcp block machanism
by default, after call recv, it would block until it receives a response from remote site.
https://www.scottklement.com/rpg/socktut/nonblocking.html#:~:text=The%20solution%20to%20this%20problem%20is



# pickle 
pickle.loads *.dumps => transmit as byte stream

pck.load .dump => transmit as file
? representing different ways of encoding?


https://stackoverflow.com/questions/69262878/what-are-the-differences-between-pickle-dump-load-and-pickle-dumps-loads

The same applies to loading - load "unpickles" from an open (readable) file object, and loads uses a bytes object.

# string find
find: 
    return -1: not found
    return >=0: substring starts from this position 

```python

ROUND = "round"
tmp = b'\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00\x8c\x06round1\x94.'
tmp = pickle.loads(tmp)
pos_round = tmp.find(ROUND)
a = tmp[pos_round+len(ROUND)+1:]
b = tmp[pos_round+len(ROUND):]
print("a:",a,"b:",b,"len_str:",len(tmp),"pos:",pos_round+len(ROUND)+1)
```
this string is: "round1"
it has 6 chars, pos starts from 0!!! which means its index being like:0,1,2,3,4,5
if tmp[pos_round+len(ROUND)+1:]: it means: tmp[0+5+1:]=>tmp[6:], out of range!!!
So, it should be like: tmp[pos_round+len(ROUND):]=>tmp[5:]
a:  b: 1 len_str: 6 pos: 6
fuser -k 20001/tcp