import numpy
import random

noise = 1 #noise

persons = ['carl','kurt','keith','kevin','shawn','rhonda','diana','denny','sam','pete','luke']

#americandata
n_persons = len(persons)
chols = []
for i in range(n_persons):
    x = numpy.random.normal(1,0.5)
    chols.append(abs(int(x)))

with open("facts.txt","a") as fp:
    for i in range(n_persons):
        if chols[i] == 0:
            fp.write("chol0("+persons[i]+")."+"\n")
        elif chols[i] == 1:
            fp.write("chol1("+persons[i]+")."+"\n")
        else:
            fp.write("chol2("+persons[i]+")."+"\n")

with open("pos.txt","a") as fp:
    for i in range(n_persons):
        if chols[i] > 0:
            if random.random() < noise:
                fp.write("ha("+persons[i]+")."+"\n")

with open("neg.txt","a") as fp:
    for i in range(n_persons):
        if chols[i] < 1:
            if random.random() < noise:
                fp.write("ha("+persons[i]+")."+"\n")
        
