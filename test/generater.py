a = ""
for i in range(4096):
    a+="x"
print(a)
print(len(a))
with open("text.txt","w") as file:
    file.write(a)