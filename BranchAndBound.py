import json
import pandas as pd

from main import upperbound

screenlines = {}
counts = {}
tourDict = {}
tourOnODDict = {}
maxZoneNR = 0







def readInModelParams(interceptPath, screenlinesUsedBool, screenlinesPath, tourDictPath, tourOnODDictPath):
    global tourDict, tourOnODDict, screenlines, counts, maxZoneNR
    if screenlinesUsedBool:
        maxZoneNR = 1400
        with open(screenlinesPath, 'r') as file:
            screenlines = json.load(file)
        with open(interceptPath, 'r') as file:
            counts = json.load(file)
    else:
        interceptDF = pd.read_csv(interceptPath, sep=";", header=None)
        maxZoneNR = interceptDF.index.size
        screenlineIndex = 1
        for origin in range(1,maxZoneNR+1):
            for destination in range(1,maxZoneNR+1):
                screenlines[screenlineIndex] = [(origin, destination)]
                counts[screenlineIndex] = interceptDF.loc[origin, destination]
    with open(tourDictPath, 'r') as file:
        tourDict = json.load(file)
    with open(tourOnODDictPath, 'r') as file:
        tourOnODDict = json.load(file)
    return


class upperboundClass:
    def __init__(self, ubMeth="tabooSearch", solutionDict=None, ubVector=None,
                 lbVector=None, newConstraint=None, value=0, ubMethodParameters=None):
        if lbVector is None:
            lbVector = [0]*nrOfAgents
        if ubVector is None:
            ubVector = [upperbound]*nrOfAgents
        if ubMethodParameters is None:
            if ubMeth == "tabooSearch":
                ubMethodParameters = {"maxDepth":300, "tabooLength":50, "maxNoImprovement":30}
        if solutionDict is None:
            solutionDict = {"best":[max(lbVector[tourID], min(1, ubVector[tourID]))
                                    for tourID in toursDF.index]}
        self.ubMethod = ubMeth
        self.solution = solutionDict
        self.ubMethodParameters = ubMethodParameters
        self.value = value
        self.lbVector = lbVector
        self.ubVector = ubVector
        self.newConstraint = newConstraint
        self.updateSolutions()

    def updateSolutions(self):
        side, tourID, value = self.newConstraint
        for key, solutionList in self.solution.items():
            if side*solutionList[tourID] > side * value:
                self.solution[key][tourID] = value


    def bound(self):
        if self.ubMethod == "tabooSearch":
            self.tabooSearch(solutionKey="best", startingWeights=self.solution["best"])


    def tabooSearch(self, solutionKey="best", startingWeights=None):
        if startingWeights is None:
            startingWeights = self.solution[solutionKey]
        tempMax, diffDict = self.evaluateSolution(startingWeights)
        tempValue = tempMax
        changeDiffDict = {1: [0] * nrOfAgents, -1: [0] * nrOfAgents}
        for tourID in range(len(startingWeights)):
            changeDiffDict[1][tourID] = self.calculateImprovement(startingWeights, change=(tourID, 1),
                                                                  diffDict=diffDict)
            changeDiffDict[-1][tourID] = self.calculateImprovement(startingWeights, change=(tourID, -1),
                                                                  diffDict=diffDict)
        changeDF = pd.DataFrame(changeDiffDict)

        # incrementDiffList = [0] * nrOfAgents
        # decrementDiffList= [0] * nrOfAgents
        # for tourID in range(len(startingWeights)):
        #     incrementDiffList[tourID] = self.calculateImprovement(startingWeights, change=(tourID, 1),
        #                                                           diffDict=diffDict)
        #     decrementDiffList[tourID] = self.calculateImprovement(startingWeights, change=(tourID, -1),
        #                                                            diffDict=diffDict)
        depth = 0
        lastImprovement = 0
        tabooList = []
        while (depth < self.ubMethodParameters["maxDepth"] and
            lastImprovement < self.ubMethodParameters["maxNoImprovement"]):
            maxIDs = changeDF.idxmax(axis=1)
            if changeDF.at[maxIDs.at[1],1] > changeDF.at[maxIDs.at[-1],-1]:
                step = (maxIDs.at[-1],-1)
            else:
                step = (maxIDs.at[1],1)
            if changeDF.at[step[0],step[1]] <0:
                return
            # To do:
            # add penalty of 100 to entries in taboo list and remove them when they leave
            # get your negative and positive improvements sorted (my guess is it's not correct atm)
            # fix abs in improvement calcuations
            # write down your structure god damn it
            # run size test
            # create list of tours that share a trip
            # peepoo poopoo
            # bruh how you techdebting already




    def calculateImprovement(self, weights, change, diffDict):
        newWeights = weights.copy()
        newWeights[change[0]] += change[1]
        tempDifference = (abs(weights[change[0]]-1 + change[1])-(abs(weights[change[0]]-1)))/dimX
        for OD in tourODsDict[change[0]]:
            tempDifference += abs(self.calcODDiff(newWeights, OD[0], OD[1])-diffDict[OD[0]][OD[1]])
        return tempDifference

    @staticmethod
    def calcODDiff(weights, origin, destination):
        agents = ODtupledict.get(tuple([origin, destination]), [])
        value = sum((weights[tourID] * toursDF.at[tourID, "Prob_auto"])
                         for tourID in agents) - interceptDF[i][j]
        return value


    def evaluateSolution(self, solutionList=None):
        if solutionList is None:
            solutionList = self.solution["best"]
        value = sum(abs(weight-1) for weight in solutionList)/dimX
        diffDict = {}
        for i in range(1, maxZoneNR+1):
            diffDict[i] = {}
            for j in range(1, maxZoneNR+1):
                ODvalue = self.calcODDiff(solutionList,i,j)
                value += abs(ODvalue)
                diffDict[i][j] = ODvalue
        return value, diffDict



class lowerboundClass:
    # newConstraint = [+/- 1, tourID, value]. 1 for upperbound, -1 for lowerbound
    def __init__(self, lbMeth="tripBasedLP", solutionDict=None, ubVector=None, lbVector=None, newConstraint=None, value=0):
        if lbVector is None:
            lbVector = [0]*nrOfAgents
        if ubVector is None:
            ubVector = [upperbound]*nrOfAgents
        if solutionDict is None:
            solutionDict = {origin:{destination:
                                        {tourID: max(lbVector[tourID], 0) for tourID in ODtupledict[origin,destination]}
                                    for destination in zones} for origin in zones}
            self.firstRun = True
        else:
            self.firstRun = False
        self.lbMethod = lbMeth
        self.solution = solutionDict
        self.value = value
        self.lbVector = lbVector
        self.ubVector = ubVector
        self.newConstraint = newConstraint
        self.markedODs = []
        self.markODs()

    def markODs(self):
        if self.firstRun:
            self.markedODs = [(origin,destination) for origin in zones for destination in zones]
        else:
            side, tourID, value = self.newConstraint
            self.markedODs = [OD for OD in tourODsDict[tourID]]

    def bound(self):
        if self.lbMethod == "tripBasedLP":
            self.tripBasedLPBound()
        else:
            print(f"The lowerbound method '{self.lbMethod}' is not supported")


    def evaluateSolution(self):
        value = 0
        for origin, destDict in self.solution.items():
            for destination, odDict in destDict.items():
                ODval = 0
                for tourID, weight in odDict.items():
                    ODval += weight * toursDF.at[tourID, "prob_auto"]
                value += abs(ODval-interceptDF.at[origin,destination])
        for tourID, ODlist in tourODsDict.items():
            totalTourWeight = 0
            for OD in ODlist:
                totalTourWeight += self.solution[OD[0]][OD[1]][tourID]
            value+= abs(totalTourWeight/len(ODlist)-1) / dimX
        self.value = value



    def tripBasedLPBound(self):
        for OD in self.markedODs:

            interceptValue = interceptDF.at[OD[0],OD[1]]
            tourIDList = self.solution[OD[0]][OD[1]].keys()
            tempSolution = [self.lbVector[tourID] for tourID in tourIDList]
            upperBounds = [self.ubVector[tourID] for tourID in tourIDList]
            probList = [toursDF.at[tourID, "prob_auto"] for tourID in tourIDList]
            n = len(probList)
            # the two list provide the order in which the tours should be handled:
            # if the probabilities are [0.1, 0.5, 0.25], the order will be 1->3->2->2->3->1, the first pass is to raise
            # the coefficients to one, the second is to raise it past 1
            ascendingProbIndices = sorted(range(n), key=lambda k: probList[k])
            descendingProbIndices = [ascendingProbIndices[-1-i] for i in range(n)]
            tempValue = sum(probList[tourID]*tempSolution[tourID] for tourID in tourIDList)
            tempDifference = interceptValue - tempValue
            orderIndex = 0
            # reduce the local difference to 0, first raising to the local solution to 1, then past 1.
            while tempDifference > 0 and orderIndex < 2*n:
                if orderIndex < n:
                    currentIndex = ascendingProbIndices[orderIndex]
                    currentCoefficient = tempSolution[currentIndex]
                    if currentCoefficient < 1 <= upperBounds[currentIndex]:
                        currentProbability = probList[currentIndex]
                        coefficientIncrease = min(tempDifference/ currentProbability,1-currentCoefficient)
                        tempSolution[currentIndex] += coefficientIncrease
                        tempDifference -= coefficientIncrease*currentProbability
                        tempValue += coefficientIncrease*currentProbability
                else:
                    currentIndex = descendingProbIndices[orderIndex-n]
                    currentCoefficient = tempSolution[currentIndex]
                    currentProbability = probList[currentIndex]
                    coefficientIncrease = min(tempDifference/ currentProbability,
                                              upperBounds[currentIndex]-currentCoefficient)
                    tempSolution[currentIndex] += coefficientIncrease
                    tempDifference -= coefficientIncrease * currentProbability
                    tempValue += coefficientIncrease * currentProbability
                orderIndex += 1
        self.evaluateSolution()








def branchAndBound(ubParamDict,lbParamDict, splitParamDict,bnbParamDict):
    ub = upperboundClass(ubParamDict)
    lb = lowerboundClass(lbParamDict)
    splitIneq = splitInequalityClass(splitParamDict)

    return








if __name__ == '__main__':
    interceptFile = "NormObservedMatrix.txt"
    screenlinesUsed = False
    screenlinesFile = "NormObservedMatrix.txt"
    tourDictFile = "clusters.json"
    tourOnODDictFile = "clusterOnODDict.json"
    readInModelParams(interceptFile, screenlinesUsed, screenlinesFile, tourDictFile, tourOnODDictFile)
    upperboundParameterDict = {}
    lowerboundParameterDict = {}
    splitInequalityParameterDict = {}
    branchAndBoundParameterDict = {}
    branchAndBound(upperboundParameterDict,lowerboundParameterDict,
                   splitInequalityParameterDict,branchAndBoundParameterDict)



