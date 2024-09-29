a = ""
for i in range(4096+1000):
    a+="x"
print(a)
print(len(a))
with open("text.txt","w") as file:
    file.write(a)