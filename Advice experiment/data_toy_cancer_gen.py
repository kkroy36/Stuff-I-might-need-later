import numpy
import random

noise = 1

persons = ['carl','kurt','keith','kevin','shawn','rhonda','diana','denny','sam','pete','luke']

n_persons = len(persons)
n_friends = []
for i in range(n_persons):
    x = numpy.random.norma(1,0.5)
    n_friends.append(abs(int(x)))

with open("facts.txt","a") as fp:
    for i in range(n_persons):
        if chols[i] == 0:
            fp.write("friends0("+persons[i]+")."+"\n")
        elif chols[i] == 1:
            fp.write("friends1("+persons[i]+")."+"\n")
        else:
            fp.write("friends2("+persons[i]+")."+"\n")with open("pos.txt","a") as fp:

with open("pos.txt","a") as fp:
    for i in range(n_persons):
        if n_friends[i] > 1:
            if random.random() < noise:
                fp.write("cancer("+persons[i]+")."+"\n")


with open("neg.txt","a") as fp:
    for i in range(n_persons):
        if n_friends[i] < 2:
            if random.random() < noise:
                fp.write("cancer("+persons[i]+")."+"\n")
