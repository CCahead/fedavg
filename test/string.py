import pickle
string = "data123begin"
begin = string.find("begin")
a = string.find("data")
print(a,string[a:a+len("data")])
print(begin,string[begin:begin+len("begin")]) # string[find_loc:len_words]

string = b"data123begin"
string = str(string) #or a.decode("utf-8")
begin = string.find("begin") # error: bytes string no method called begin
# https://stackoverflow.com/questions/38777714/how-to-check-if-a-specific-string-is-in-a-variable-that-byte-variable
a = string.find("data") 
print(a,string[a:a+len("data")])
print(begin,string[begin:begin+len("begin")]) # string[find_loc:len_words]

# transmit a dict using pickle
s_dict = dict()
s_dict["a"] = "abc"
s_dict["c"] = "abc"
print(s_dict)
pck = pickle.dumps(s_dict)
print(pck)
# {'a': 'abc', 'c': 'abc'}
# b'\x80\x04\x95\x15\x00\x00\x00\x00\x00\x00\x00}\x94(\x8c\x01a\x94\x8c\x03abc\x94\x8c\x01c\x94h\x02u.'

data = f"data{len(pck)}"
print("before:",data)
begin = b"begin"
data = data.encode("utf-8")
print("encode:",data)
data += begin
data += pck
print("after:",data)

# before: data32
# encode: b'data32'
# after: b'data32begin\x80\x04\x95\x15\x00\x00\x00\x00\x00\x00\x00}\x94(\x8c\x01a\x94\x8c\x03abc\x94\x8c\x01c\x94h\x02u.'

# *** decode *** 
# string = data.decode("utf-8")
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 11: invalid start byte
string = str(data)
head = "data"
payload_head = "begin"
print("decodeString:",string)
# payload_start = 
loc_payload_head = string.find(payload_head)
print(string[:len(head)])
print(string[len(head)+1:loc_payload_head-1])
print(string[loc_payload_head:len(payload_head)+loc_payload_head])
print(string[loc_payload_head+len(payload_head)+1:-1]) # A:B range [A:B] -1 stands for ignore last pos


a = b"12"

print(len(a)) # works! return 2

a = "12"
pck = pickle.dumps(a)
print(pck)
a = pickle.loads(pck)
print(a)
print(type(a))
print(a.find("z"))

ROUND = "round"
tmp = b'\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00\x8c\x06round1\x94.'
tmp = pickle.loads(tmp)
pos_round = tmp.find(ROUND)
a = tmp[pos_round+len(ROUND)+1:]
b = tmp[pos_round+len(ROUND):]
print("a:",a,"b:",b,"len_str:",len(tmp),"pos:",pos_round+len(ROUND)+1)
