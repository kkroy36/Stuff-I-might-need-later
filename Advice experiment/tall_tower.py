import random

#on4 = ['A','B','C','D']
#on3 = ['A','B','C']
#on2 = ['A','B']
noise = 1 #noise percentage

for i in range(100):
    t = random.randint(2,4)
    if t == 2:
        with open("facts.txt","a") as fp:
            fp.write("on(s"+str(i)+",b,a)."+"\n")
        with open("neg.txt","a") as fp:
            fp.write("putdown(s"+str(i)+")."+"\n")
    elif t == 3:
        with open("facts.txt","a") as fp:
            fp.write("on(s"+str(i)+",b,a)."+"\n")
        with open("facts.txt","a") as fp:
            fp.write("on(s"+str(i)+",c,b)."+"\n")
	if random.random() > noise:
            with open("neg.txt","a") as fp:
                fp.write("putdown(s"+str(i)+")."+"\n")
        else:
            with open("pos.txt","a") as fp:
                fp.write("putdown(s"+str(i)+")."+"\n")
    else:
        with open("facts.txt","a") as fp:
            fp.write("on(s"+str(i)+",b,a)."+"\n")
        with open("facts.txt","a") as fp:
            fp.write("on(s"+str(i)+",c,b)."+"\n")
        with open("facts.txt","a") as fp:
            fp.write("on(s"+str(i)+",d,c)."+"\n")
	if random.random() > noise:
            with open("pos.txt","a") as fp:
                fp.write("putdown(s"+str(i)+")."+"\n")
	else:
	    with open("neg.txt","a") as fp:
	        fp.write("putdown(s"+str(i)+")."+"\n")


