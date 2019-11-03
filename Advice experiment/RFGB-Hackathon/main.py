from GradientBoosting import GradientBoosting
import random

def main():
    '''main method that runs boosting'''

    bk = ["chol0(+Person)","chol1(+Person)","ha(Person)"]
    facts = []
    pos = []
    neg = []

    test_facts = []
    test_pos = []
    test_neg = []

    with open("Dem/train/train_facts.txt") as f:
        facts = f.read().splitlines()
        facts = [item[:-1] for item in facts]

    with open("Dem/train/train_pos.txt") as p:
        pos = p.read().splitlines()
        pos = [item[:-1] for item in pos]

    with open("Dem/train/train_neg.txt") as n:
        neg = n.read().splitlines()
        neg = [item[:-1] for item in neg]

    with open("Dem/test/test_facts.txt") as f:
        test_facts = f.read().splitlines()
        test_facts = [item[:-1] for item in test_facts]

    with open("Dem/test/test_pos.txt") as p:
        test_pos = p.read().splitlines()
        test_pos = [item[:-1] for item in test_pos]

    with open("Dem/test/test_neg.txt") as n:
        test_neg = n.read().splitlines()
        test_neg = [item[:-1] for item in test_neg]
        
    '''
    ratio = len(neg)/float(len(pos))

    if ratio > 1:
        prob = 2*len(pos)/float(len(neg))
        neg = [item for item in neg if random.random() < prob]
    '''
    clf = GradientBoosting(treeDepth = 1, trees = 10, expert_advice = True)
    clf.setTargets(["ha"])
    clf.learn_clf(facts,pos,neg,bk)
    clf.infer_clf(test_facts,test_pos,test_neg)

main()
