from Utils import Utils
from math import log,exp
from Logic import Prover
from copy import deepcopy
import sklearn.metrics as sk
import numpy
import pulp

class Boosting(object):
    '''boosting class'''
    
    logPrior = log(0.5/float(1-0.5))

    @staticmethod
    def computeAdviceGradient(example):
        '''computes the advice gradients as nt-nf'''
        nt,nf = 0,0
        target = Utils.data.target.split('(')[0]
        for clause in Utils.data.adviceClauses:
            if Prover.prove(Utils.data,example,clause.split('.')[1]):
                if target in Utils.data.adviceClauses[clause]['preferred']:
                    nt += 1
                if target in Utils.data.adviceClauses[clause]['nonPreferred']:
                    nf += 1
        return (nt-nf)

    @staticmethod
    def computeExpertAdviceGradient(example,exampleProbability,exampleGradient):
        '''computes the expert advice gradient as I-P where it applies'''
        target = Utils.data.target.split('(')[0]
        for clause in Utils.data.expertAdviceClauses:
            if Prover.prove(Utils.data,example,clause.split('.')[1]):
                if target in Utils.data.expertAdviceClauses[clause]['preferred']:
                    return (1-exampleProbability)
                elif target in Utils.data.expertAdviceClauses[clause]['nonPreferred']:
                    return (0-exampleProbability)
        return (exampleGradient)

    @staticmethod
    def computeAdviceGradients(example,exampleProbability,exampleGradient,clauseIDs):
        '''computes the gradient I-P for each piece of advice'''
        gradients = [0 for i in range(len(Utils.data.multipleAdviceClauses))]
        target = Utils.data.target.split('(')[0]
        for clause in Utils.data.multipleAdviceClauses:
            if Prover.prove(Utils.data,example,clause.split('.')[1]):
                if target in Utils.data.multipleAdviceClauses[clause]['preferred']:
                    gradients[int(clause.split('.')[0])-1] = 1-exampleProbability
                    #gradients.append((clauseIDs[clause],1-exampleProbability))
                elif target in Utils.data.multipleAdviceClauses[clause]['nonPreferred']:
                    gradients[int(clause.split('.')[0])-1] = 0-exampleProbability
                    #gradients.append((clauseIDs[clause],0-exampleProbability))
        return (gradients)

    @staticmethod
    def inferTreeValue(clauses,query,data):
        '''returns probability of query
           given data and clauses learned
        '''
        for clause in clauses: #for every clause in the tree
            clauseCopy = deepcopy(clause)
            clauseValue = float(clauseCopy.split(" ")[1])
            clauseRule = clauseCopy.split(" ")[0].replace(";",",")
            if not clauseRule.split(":-")[1]:
                return clauseValue
            if Prover.prove(data,query,clauseRule): #check if query satisfies clause
                return clauseValue
    
    @staticmethod
    def computeSumOfGradients(example,trees,data):
        '''computes new gradient for example'''
        sumOfGradients = 0
        for tree in trees: #add leaf values satisfied by example in each tree
            gradient = Boosting.inferTreeValue(tree,example,data)
            #print ("value of example: ",gradient)
            if gradient != None:
                sumOfGradients += gradient
        return sumOfGradients #return the sum

    @staticmethod
    def computeAdviceGradientSums(adviceGradients,clauseIDs):
        '''computes batch gradient by summing'''

        ids = list(clauseIDs.values())
        nids = len(ids)
        nexamples = len(adviceGradients)
        sums = [0 for k in range(nids)]
        for example in adviceGradients:
            gradients = adviceGradients[example]
            for gradient in gradients:
                sums[gradient[0]] += gradient[1]
        sums = [x/float(nexamples) for x in sums]
        return (sums)

    @staticmethod
    def computeLP(c):
        '''computes LP solution'''
        
        problem = pulp.LpProblem("max", pulp.LpMaximize)
        lambda1 = pulp.LpVariable("lamda1",0,1)
        lambda2 = pulp.LpVariable("lamda2",0,1)
        lambda3 = pulp.LpVariable("lambda3",0,1)
        problem += lambda1 + lambda2 + lambda3 <= 1
        problem += lambda1 + lambda2 + lambda3 >= 1
        problem += c[0]*lambda1 + c[1]*lambda2 + c[2]*lambda3
        problem.solve()
        return ([lambda1.varValue,lambda2.varValue,lambda3.varValue])

    @staticmethod
    def updateWeights(example,P_prev,data_gradient,W):
        '''makes gradient update'''
        #psi_prev = log(P_prev/float(1-P_prev))
        adv_gradients = {}
        gradient = {}
        target = Utils.data.target.split('(')[0]
        for clause in Utils.data.expertAdviceClauses:
            adv_gradients[clause] = data_gradient
            if Prover.prove(Utils.data,example,clause.split('.')[1]):
                if target in Utils.data.expertAdviceClauses[clause]['preferred']:
                    adv_gradients[clause] = 1-P_prev
                elif target in Utils.data.expertAdviceClauses[clause]['nonPreferred']:
                    adv_gradients[clause] = 0-P_prev

        for clause in W:
            gradient[clause] = 0
        '''
        net_gradient = W["data"]*data_gradient
        for clause in W:
            if clause != "data":
                net_gradient += W[clause]*adv_gradients[clause]
        P = psi_prev + net_gradient
        '''
        for clause in W:
            if clause != "data":
                gradient[clause] += (adv_gradients[clause]*data_gradient)
            elif clause == "data":
                gradient[clause] += (data_gradient**2)
        '''
        for clause in W:
            W[clause] = W[clause]/float(sum(list(W.values())))
        '''
        return (gradient,adv_gradients)
            

    @staticmethod
    def computeLinearSum(exampleGradients,exampleProbabilities):
        '''computes combination weights by SGD'''

        target = Utils.data.target.split('(')[0]
        W = {}
        total = {}
        exampleAdvGradients = {}
        for clause in Utils.data.expertAdviceClauses:
            W[clause] = 0

        W["data"] = 0

        for clause in W:
            total[clause] = 0
            

        for example in exampleGradients:
            returned = Boosting.updateWeights(example,exampleProbabilities[example],exampleGradients[example],W)
            gradients = returned[0]
            exampleAdvGradients[example] = returned[1]
            for clause in W:
                total[clause] += gradients[clause]

        for clause in W:
            W[clause] = total[clause]


        for example in exampleGradients:
            exampleGradients[example] =  W["data"]*exampleGradients[example]
            for clause in W:
                if clause != "data":
                    exampleGradients[example] += W[clause]*exampleAdvGradients[example][clause]
            '''
            P_example = exp(exampleGradients[example])/(1+exp(exampleGradients[example]))
            exampleGradients[example] = I_example - P_example
            '''

        return (exampleGradients)

    @staticmethod
    def computeCombinedGradients(exampleGradients,exampleProbabilities):
        '''computes convex combination of gradients by
           solving linear program
        '''

        clauseIDs = {}
        #adviceGradients = {}
        i = 0
        for example in exampleGradients:
            adviceGradients = Boosting.computeAdviceGradients(example,exampleProbabilities[example],exampleGradients[example],clauseIDs)
            dataGradient = exampleGradients[example]
            #gradients = [dataGradient]+adviceGradients
            aggregateGradientIndex = [abs(g) for g in adviceGradients].index(max([abs(g) for g in adviceGradients])) #combined advice using max
            aggregateGradient = adviceGradients[aggregateGradientIndex]
            #print (lpSolution)
            #raw_input()
            combinedGradient = 0.3*dataGradient + 0.7*aggregateGradient
            '''
            n = len(adviceGradients)
            for i in range(n):
                combinedGradient += adviceGradients[i] #sum, lp solution would be max, also can do average
            '''
            exampleGradients[example] = combinedGradient

        '''
        sumOfDataGradients = sum(list(exampleGradients.values()))
        for example in exampleGradients:
            adviceGradients[example] = Boosting.computeAdviceGradients(example,exampleProbabilities[example],exampleGradients[example],clauseIDs)
        sumofAdviceGradients = Boosting.computeAdviceGradientSums(adviceGradients,clauseIDs)
        gradientSums = [sumOfDataGradients/float(len(exampleGradients))]+sumofAdviceGradients
        lpSolution = Boosting.computeLP([abs(g) for g in gradientSums])
        for example in exampleGradients:
            dataGradient = exampleGradients[example]
            combinedGradient = lpSolution[0]*dataGradient
            adviceGradient = adviceGradients[example]
            for gradient in adviceGradient:
                combinedGradient += lpSolution[gradient[0]+1]*gradient[1]
            exampleGradients[example] = combinedGradient
        '''
        
        return (exampleGradients)

    @staticmethod
    def getAntecedent(example,modularExpertAdviceClauses):
        '''gets relevant clauses'''

        for antecedent in modularExpertAdviceClauses:
            for clause in modularExpertAdviceClauses[antecedent]:
                if Prover.prove(Utils.data,example,clause.split('.')[1]):
                    return (antecedent)

    @staticmethod
    def computeMinDotProduct(exampleGradients,exampleProbabilities):
        '''computes advice gradient with the least dot product with data gradient'''

        target = Utils.data.target.split('(')[0]
        modularExpertAdviceClauses = {}
        minimumPerAntecedent = {}
        print (exampleGradients)

        for clause in Utils.data.expertAdviceClauses:
            antecedent = (clause.split('.')[1]).split(':-')[1]
            if antecedent not in modularExpertAdviceClauses:
                modularExpertAdviceClauses[antecedent] = [clause]
            else:
                modularExpertAdviceClauses[antecedent].append(clause)


        exampleGradientsPerAntecedent = {}

        for antecedent in modularExpertAdviceClauses:
            minimumPerAntecedent[antecedent] = 1

        for antecedent in modularExpertAdviceClauses:
            relClauses = modularExpertAdviceClauses[antecedent]
            A = len(relClauses)
            dotProducts = [0 for a in range(A)]
            for example in exampleGradients:
                exampleAntecedent = Boosting.getAntecedent(example,modularExpertAdviceClauses)
                if exampleAntecedent == antecedent:
                    exampleGradient = exampleGradients[example]
                    for clause in relClauses:
                        ind = relClauses.index(clause)
                        if Prover.prove(Utils.data,example,clause.split('.')[1]):
                            if target in Utils.data.expertAdviceClauses[clause]['preferred']:
                                dotProducts[ind] += ((1-exampleProbabilities[example])*exampleGradient)
                            elif target in Utils.data.expertAdviceClauses[clause]['nonPreferred']:
                                dotProducts[ind] += ((0-exampleProbabilities[example])*exampleGradient)
            minimumPerAntecedentValue = min(dotProducts)
            minInd = dotProducts.index(minimumPerAntecedentValue)
            minimumPerAntecedent[antecedent] = int(relClauses[minInd].split('.')[0])

        for example in exampleGradients:
            exampleAntecedent = Boosting.getAntecedent(example,modularExpertAdviceClauses)
            minimumDotID = minimumPerAntecedent[exampleAntecedent]
            for clause in Utils.data.expertAdviceClauses:
                if int(clause.split('.')[0]) ==  minimumDotID:
                    if Prover.prove(Utils.data,example,clause.split('.')[1]):
                        adviceGradient = 0
                        if target in Utils.data.expertAdviceClauses[clause]['preferred']:
                            adviceGradient = 1-exampleProbabilities[example]
                        else:
                            adviceGradient = 0-exampleProbabilities[example]
                        exampleGradients[example] = exampleGradients[example] + adviceGradient

        '''

        for example in exampleGradients:
            exampleAntecedent = Boosting.getAntecedent(example,modularExpertAdviceClauses)
            relevantClauses = modularExpertAdviceClauses[exampleAntecedent]
            A = len(relevantClauses)
            exampleGradientsPerAntecedent[
            for clause in relevantClauses:
        '''
            
        '''
        adviceDotProducts = [0 for i in range(len(Utils.data.expertAdviceClauses))]
        for example in exampleGradients:
            for clause in Utils.data.expertAdviceClauses:
                advice_index = int(clause.split('.')[0])-1
                if Prover.prove(Utils.data,example,clause.split('.')[1]):
                    if target in Utils.data.expertAdviceClauses[clause]['preferred']:
                        adviceDotProducts[advice_index] += ((1-exampleProbabilities[example])*exampleGradients[example])
                    elif target in Utils.data.expertAdviceClauses[clause]['nonPreferred']:
                        adviceDotProducts[advice_index] += ((0-exampleProbabilities[example])*exampleGradients[example])
        advice_id = adviceDotProducts.index(min(adviceDotProducts))+1
        min_advice = None
        '''
        return (exampleGradients)
            

    @staticmethod
    def computeProjection(exampleGradients,exampleProbabilities):
        '''computes projected gradients as constrained by expert advice'''

        exampleGradientIndices = {}
        no_of_examples = len(exampleGradients)
        count = 0
        for example in exampleGradients:
            exampleGradientIndices[example] = count
            count += 1
        exampleGradientVector = [0 for i in range(no_of_examples)]
        for example in exampleGradientIndices:
            exampleGradientVector[exampleGradientIndices[example]] = exampleGradients[example]
        exampleAdviceGradientVector = [0 for i in range(no_of_examples)]
        for example in exampleGradientIndices:
            expertAdviceGradient = Boosting.computeExpertAdviceGradient(example,exampleProbabilities[example],exampleGradients[example])
            exampleAdviceGradientVector[exampleGradientIndices[example]] = expertAdviceGradient
        scalar_projection = numpy.dot(exampleGradientVector,exampleAdviceGradientVector)/float(numpy.linalg.norm(exampleAdviceGradientVector))
        unit_vector = [x/float(numpy.linalg.norm(exampleAdviceGradientVector)) for x in exampleAdviceGradientVector]
        vector_projection = [scalar_projection*item for item in unit_vector]
        for example in exampleGradients:
            for i in range(no_of_examples):
                if exampleGradientIndices[example] == i:
                    exampleGradients[example] = vector_projection[i]
        return exampleGradients
        

    @staticmethod
    def updateGradients(data,trees,loss="LS",delta=1.3):
        '''updates the gradients of the data'''
        if not data.regression:
            logPrior = Boosting.logPrior
            #P = sigmoid of sum of gradients given by each tree learned so far
            exampleGradients = {}
            exampleProbabilities = {}
            for example in data.pos: #for each positive example compute 1 - P
                sumOfGradients = Boosting.computeSumOfGradients(example,trees,data)
                probabilityOfExample = Utils.sigmoid(logPrior+sumOfGradients)
                updatedGradient = 1 - probabilityOfExample
                if data.advice:
                    adviceGradient = Boosting.computeAdviceGradient(example)
                    updatedGradient += adviceGradient
                if data.expert_advice:
                    exampleGradients[example] = updatedGradient
                    exampleProbabilities[example] = probabilityOfExample
                if data.multiple_advice:
                    exampleGradients[example] = updatedGradient
                    exampleProbabilities[example] = probabilityOfExample
                data.pos[example] = updatedGradient
            for example in data.neg: #for each negative example compute 0 - P
                sumOfGradients = Boosting.computeSumOfGradients(example,trees,data)
                probabilityOfExample = Utils.sigmoid(logPrior+sumOfGradients)
                updatedGradient = 0 - probabilityOfExample
                if data.advice:
                    adviceGradient = Boosting.computeAdviceGradient(example)
                    updatedGradient += adviceGradient
                if data.expert_advice:
                    exampleGradients[example] = updatedGradient
                    exampleProbabilities[example] = probabilityOfExample
                if data.multiple_advice:
                    exampleGradients[example] = updatedGradient
                    exampleProbabilities[example] = probabilityOfExample
                data.neg[example] = updatedGradient
            if data.expert_advice:
                exampleGradients = Boosting.computeMinDotProduct(exampleGradients,exampleProbabilities)
                #exampleGradients = Boosting.computeLinearSum(exampleGradients,exampleProbabilities)
                for example in data.pos:
                    data.pos[example] = exampleGradients[example]
                for example in data.neg:
                    data.neg[example] = exampleGradients[example]
            if data.multiple_advice:
                exampleGradients = Boosting.computeCombinedGradients(exampleGradients,exampleProbabilities)
                for example in data.pos:
                    data.pos[example] = exampleGradients[example]
                for example in data.neg:
                    data.neg[example] = exampleGradients[example]
                
        if data.regression:
            for example in data.examples: #compute gradient as y-y_hat
                sumOfGradients = Boosting.computeSumOfGradients(example,trees,data)
                trueValue = data.getExampleTrueValue(example)
                exampleValue = sumOfGradients
                if loss == "LS":
                    updatedGradient = trueValue - exampleValue
                    data.examples[example] = updatedGradient
                elif loss == "LAD":
                    updatedGradient = 0
                    gradient = trueValue - exampleValue
                    if gradient:
                        updatedGradient = gradient/float(abs(gradient))
                    data.examples[example] = updatedGradient
                elif loss == "Huber":
                    gradient = trueValue - exampleValue
                    updatedGradient = 0
                    if gradient:
                        if abs(gradient) > float(delta):
                            updatedGradient = gradient/float(abs(gradient))
                        elif gradient <= float(delta):
                            updatedGradient = gradient
                    data.examples[example] = updatedGradient

    
    @staticmethod
    def computeResults(testData):
        #computes accuracy,AUC-ROC and AUC-PR
        yactual = [1 for i in range(len(list(testData.pos.values())))] + [0 for i in range(len(list(testData.neg.values())))]
        ypred = [int(x >= 0.5) for x in list(testData.pos.values())+list(testData.neg.values())]
        print ("accuracy: ",sk.accuracy_score(yactual,ypred))
        print ("AUC-ROC: ",sk.roc_auc_score(yactual,ypred))
        print ("AUC-PR: ",sk.average_precision_score(yactual,ypred))
        print ("Precision: ",sk.precision_score(yactual,ypred))
        print ("Recall: ",sk.recall_score(yactual,ypred))
        print ("F1: ",sk.f1_score(yactual,ypred))
    
        
    @staticmethod
    def performInference(testData,trees):
        '''computes probability for test examples'''
        logPrior = Boosting.logPrior
        if not testData.regression:
            logPrior = Boosting.logPrior #initialize log odds of assumed prior probability for example
            for example in testData.pos:
                print ("testing example: ",example)
                sumOfGradients = Boosting.computeSumOfGradients(example,trees,testData) #compute sum of gradients
                testData.pos[example] = Utils.sigmoid(logPrior+sumOfGradients) #calculate probability as sigmoid(log odds)
            for example in testData.neg:
                print ("testing example: ",example)
                sumOfGradients = Boosting.computeSumOfGradients(example,trees,testData) #compute sum of gradients
                testData.neg[example] = Utils.sigmoid(logPrior+sumOfGradients) #calculate probability as sigmoid(log odds)
        elif testData.regression:
            for example in testData.examples:
                sumOfGradients = Boosting.computeSumOfGradients(example,trees,testData)
                testData.examples[example] = sumOfGradients
        Boosting.computeResults(testData)
