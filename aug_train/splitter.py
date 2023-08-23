import random 
 
fin = open("gc_ori.txt", 'rb') 
f70out = open("train_gc.txt", 'wb') 
f15vout = open("val_gc.txt", 'wb') 
f15tout = open("test_gc.txt", 'wb') 

for line in fin: 
    r = random.random() 
    if r < 0.7: 
        f70out.write(line) 
    elif 0.7<=r<0.85: 
        f15vout.write(line)
    else:
        f15tout.write(line) 
fin.close() 
f70out.close() 
f15vout.close()
f15tout.close() 
