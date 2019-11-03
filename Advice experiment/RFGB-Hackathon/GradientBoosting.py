from __future__ import print_function

from Utils import Utils
from Tree import node
from Boosting import Boosting
from sys import argv
from os import system

class GradientBoosting(object):

    def __init__(self,regression=False,trees=10,treeDepth=2,loss="LS",sampling_rate=1.0,advice=False,expert_advice=False,multiple_advice=False):
        self.targets = None
        self.regression = regression
        self.sampling_rate = sampling_rate
        self.numberOfTrees = trees
        self.treeDepth = treeDepth
        self.trees = {}
        self.data = None
        self.loss = loss
        self.advice = advice
        self.expert_advice = expert_advice
        self.multiple_advice = multiple_advice
        self.testPos,self.testNeg,self.testExamples = {},{},{}

    def setTargets(self,targets):
        self.targets = targets
    
    def addTarget(self,target):
        self.targets.append(target)
    
    def learn(self,facts,examples,bk):
        '''learns regressor'''
        
        self.trees = {}
        for target in self.targets:
            data = Utils.setTrainingData(target=target,facts=facts,examples=examples,bk=bk,regression=self.regression,sampling_rate = self.sampling_rate,advice=self.advice,expert_advice=self.expert_advice)
            trees = []
            for i in range(self.numberOfTrees):
                print ('='*20,"learning tree",str(i+1),'='*20)
                node.setMaxDepth(self.treeDepth)
                node.learnTree(data)
                trees.append(node.learnedDecisionTree)
                Boosting.updateGradients(data,trees,loss=self.loss)
            self.trees[target] = trees
        for target in self.targets:
            if target in self.trees:
                number_of_trees = len(self.trees[target])
                counter = 1
                for tree in self.trees[target]:
                    print ('='*30,"tree",str(counter),'='*30)
                    for clause in tree:
                        print (clause)
                    counter += 1

    def learn_clf(self,facts,pos,neg,bk):
        '''learns classifier'''
        
        self.trees = {}
        for target in self.targets:
            data = Utils.setTrainingData(target=target,facts=facts,pos=pos,neg=neg,bk=bk,regression=self.regression,sampling_rate = self.sampling_rate,advice=self.advice,expert_advice=self.expert_advice,multiple_advice=self.multiple_advice)
            trees = []
            for i in range(self.numberOfTrees):
                print ('='*20,"learning tree",str(i+1),'='*20)
                node.setMaxDepth(self.treeDepth)
                node.learnTree(data)
                trees.append(node.learnedDecisionTree)
                Boosting.updateGradients(data,trees)
            self.trees[target] = trees
        for target in self.targets:
            if target in self.trees:
                number_of_trees = len(self.trees[target])
                counter = 1
                for tree in self.trees[target]:
                    print ('='*30,"tree",str(counter),'='*30)
                    for clause in tree:
                        print (clause)
                    counter += 1
                    
    def get_tree_clauses(self,tree):
        '''prints clauses in the input tree'''
        clauses = []
        for clause in tree:
            clauses.append(clause)
        return (clauses)

    def infer(self,facts,examples):
        self.testExamples = {}
        for target in self.targets:
            data = Utils.setTestData(target=target,facts=facts,examples=examples,regression=self.regression)
            Boosting.performInference(data,self.trees[target])
            self.testExamples[target] = data.examples
            #print (data.examples)
            
    def infer_clf(self,facts,pos,neg):
        self.testPos = {}
        self.testNeg = {}
        for target in self.targets:
            data = Utils.setTestData(target=target,facts=facts,pos=pos,neg=neg)
            Boosting.performInference(data,self.trees[target])
