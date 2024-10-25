import json
import pandas as pd
import igraph as ig
import scipy
import numpy as np
from main import upperbound, nrOfAgents


# screenlines = {screenlineName:{(O,D):value}
# counts = {screenlineName:count}
# tourDict = {id:[weight, [(O,D),(O,D)], set(neighbourID), noNeighbours, prob_auto]}
# tourOnODDict = {(O,D):[clusterID]}
# maxZonerNR = int
# nrOfClusters = size of tourDict
# baseWeightSum = sum of weights of tours in tourDict


# new dataset:
# screenlineNames = pd.Series of name as index and 0 indexed index as value             (size = alpha)
# ODNames = pd.Series of OD as index and 0 indexed index as value                       (size = m)
# tourNames = pd.Series() of tour id as index and 0 indexed index as value              (size = n)
# all following lists and matrices use the 0 indexed indices as described above
# counts = [count]                                                                      size is alpha
# tourBaseWeights = [tour_weight]                                                       size is n
# tourProbabilities = [tour_prob]                                                       size is n
# all sparce matrices below are in csr (Compressed Sparse Column array)
# adjTourOD = sparce adjacency matrix tours as rows, OD as columns                      size is nxm
# adjSlOD = sparce adjacency matrix screenlines as rows, OD as columns                  size is alphaxm
# adjSlOD = sparce adjacency matrix tours as rows, screenlines as columns               size is nxalpha
# neighboursOD = sparce adjacency matrix tours as rows and columns,                     size is nxn
#                   1 on off diagnal if they share OD, diagonal how many neighbours it has
# neighboursSl = sparce adjacency matrix tours as rows and columns,                     size is nxn
#                   1 on off diagnal if they share screenline, diagonal how many neighbours it has


# maxZonerNR = int
# nrOfClusters = size of tourDict
# baseWeightSum = sum of weights of tours in tourDict


screenlines = {}
screenlineNames = pd.Series()
counts = {}
tourDict = {}
tourNames = pd.Series()
tourOnODDict = {}
ODNames = pd.Series()
maxZoneNR = 0
nrOfClusters = 0
baseWeightSum = 0


def readInModelParams(interceptPath, screenlinesUsedBool, screenlinesPath, tourDictPath, tourOnODDictPath):
    global tourDict, tourOnODDict, screenlines, counts, maxZoneNR, nrOfClusters, baseWeightSum, \
        ODNames, tourNames, screenlineNames

    with open(tourDictPath, 'r') as file:
        tourDict = json.load(file)
    with open(tourOnODDictPath, 'r') as file:
        tourOnODDict = json.load(file)
    if screenlinesUsedBool:
        maxZoneNR = 1400
        with open(screenlinesPath, 'r') as file:
            screenlines = json.load(file)
        with open(interceptPath, 'r') as file:
            counts = json.load(file)
    else:
        interceptDF = pd.read_csv(interceptPath, sep=";", header=None)
        maxZoneNR = 1400
        interceptSize = interceptDF.index.size
        screenlineIndex = 0
        for origin in range(1,maxZoneNR+1):
            for destination in range(1,maxZoneNR+1):
                if f"({origin}, {destination})" in tourOnODDict:    # interceptDF.loc[origin-1, destination-1] > 0 and
                    screenlines[f"sl{screenlineIndex}"] = {f"({origin}, {destination})":1.0}
                    if origin > interceptSize or destination > interceptSize:
                        counts[f"sl{screenlineIndex}"] = 0
                    else:
                        counts[f"sl{screenlineIndex}"] = interceptDF.loc[origin-1, destination-1]
                    screenlineIndex += 1
    nrOfClusters = len(tourOnODDict)
    baseWeightSum = sum(tourOnODDict[tour][0] for tour in tourOnODDict)
    tourNames = pd.Series(data=range(len(tourDict.keys())), index=tourDict.keys())
    screenlineNames = pd.Series(data=range(len(screenlines.keys())), index=screenlines.keys())
    ODNames = pd.Series(data=range(len(tourOnODDict.keys())), index=tourOnODDict.keys())
    return





def makeSparceAdjacencyMatrices():
    tourNo = len(tourNames)
    ODNo = len(ODNames)
    slNo = len(screenlineNames)


    adjTourSl = scipy.sparse.csr_array((tourNo, slNo))
    rowList = []
    columnList = []
    valueList = []
    for tourID,tourIdx in tourNames.items():
        ODIDList = tourDict[tourID][1]
        ODIdxList = ODNames.loc[ODIDList]
        for ODIdx in ODIdxList:
            rowList.append(tourIdx)
            columnList.append(ODIdx)
            valueList.append(1.0)
    adjTourOD = scipy.sparse.csr_array((np.array(valueList), (np.array(rowList), np.array(columnList))))
    rowList2 = []
    columnList2 = []
    valueList2 = []
    for slID,slIdx in screenlineNames.items():
        ODIDList = list(screenlines[slID].keys())
        ODIdxList = ODNames.loc[ODIDList]
        for ODIdx in ODIdxList:
            rowList2.append(slIdx)
            columnList2.append(ODIdx)
            valueList2.append(1.0)
    adjSlOD = scipy.sparse.csr_array((np.array(valueList2), (np.array(rowList2), np.array(columnList2))),
                                     shape=(slNo, ODNo))
    adjTourSl = adjTourOD @ (adjSlOD.transpose())
    neighboursOD = adjTourOD @ (adjTourOD.transpose())
    neighboursSl = adjTourSl @ (adjTourSl.transpose())
    scipy.sparse.save_npz("adjTourOD",adjTourOD)
    scipy.sparse.save_npz("adjSlOD", adjSlOD)
    scipy.sparse.save_npz("adjTourSl", adjTourSl)
    scipy.sparse.save_npz("neighboursOD", neighboursOD)
    scipy.sparse.save_npz("neighboursSl", neighboursSl)
    return


class upperboundClass:
    def __init__(self, ubMeth="tabooSearch", solutionDict=None, ubVector=None,
                 lbVector=None, newConstraint=None, value=0, ubMethodParameters=None):
        if lbVector is None:
            lbVector = [0] * nrOfAgents
        if ubVector is None:
            ubVector = [upperbound] * nrOfAgents
        if ubMethodParameters is None:
            if ubMeth == "tabooSearch":
                ubMethodParameters = {"maxDepth": 300, "tabooLength": 50, "maxNoImprovement": 30}
        if solutionDict is None:
            solutionDict = {"best": [max(lbVector[tourID], min(1, ubVector[tourID]))
                                     for tourID in tourDict.keys()]}
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
            if side * solutionList[tourID] > side * value:
                self.solution[key][tourID] = value

    def bound(self):
        if self.ubMethod == "tabooSearch":
            self.tabooSearch(solutionKey="best", startingWeights=self.solution["best"])

    def tabooSearch(self, solutionKey="best", startingWeights=None):
        if startingWeights is None:
            startingWeights = self.solution[solutionKey]
        tempMax, diffDict = self.evaluateSolution(startingWeights)
        tempValue = tempMax
        # first make matrix of objective change when in/decrementing a tour by 1
        changeDiffDict = {1: [0] * nrOfClusters, -1: [0] * nrOfClusters}
        for tourID in range(len(nrOfClusters)):
            changeDiffDict[1][tourID] = self.calculateImprovement(startingWeights, change=(tourID, 1),
                                                                  diffDict=diffDict)
            changeDiffDict[-1][tourID] = self.calculateImprovement(startingWeights, change=(tourID, -1),
                                                                   diffDict=diffDict)
        changeDF = pd.DataFrame(changeDiffDict)

        depth = 0
        lastImprovement = 0
        tabooList = []
        while (depth < self.ubMethodParameters["maxDepth"] and
               lastImprovement < self.ubMethodParameters["maxNoImprovement"]):
            minIDs = changeDF.idxmin(axis=1)
            if changeDF.at[minIDs.at[1], 1] > changeDF.at[minIDs.at[-1], -1]:
                step = (minIDs.at[-1], -1)
            else:
                step = (minIDs.at[1], 1)

            if changeDF.at[step[0], step[1]] >= 0:
                lastImprovement += 1


    def calculateImprovement(self, weights, change, diffDict):
        newWeights = weights.copy()
        newWeights[change[0]] += change[1]
        tempDifference = (abs(weights[change[0]] - 1 + change[1]) - (abs(weights[change[0]] - 1))) / dimX
        for OD in tourDict[change[0]]:
            tempDifference += abs(self.calcScreenlineDiff(newWeights, ) - diffDict[OD[0]][OD[1]])
        return tempDifference



    def calcScreenlineDiff(self, weights, screenlineName):
        for OD in screenlines[screenlineName]:
            return
    # depreciated, use calcSceenlineDiff instead, supports
    @staticmethod
    def calcODweight(weights, origin, destination):
        agents = tourOnODDict.get(tuple([origin, destination]), [])
        value = sum((weights[tourID] * tourDict[-1])
                    for tourID in agents)
        return value

    def evaluateSolution(self, solutionList=None):
        if solutionList is None:
            solutionList = self.solution["best"]
        value = sum(abs(weight - 1) for weight in solutionList) / dimX
        diffDict = {}
        for i in range(1,maxZoneNR+1):
            diffDict[i] = {}
            for j in range(1,maxZoneNR+1):
                ODvalue = self.calcODDiff(solutionList, i, j)
                value += abs(ODvalue)
                diffDict[i][j] = ODvalue
        return value, diffDict


class lowerboundClass:
    # newConstraint = [+/- 1, tourID, value]. 1 for upperbound, -1 for lowerbound
    def __init__(self, lbMeth="tripBasedLP", solutionDict=None, ubVector=None, lbVector=None, newConstraint=None,
                 value=0):
        if lbVector is None:
            lbVector = [0] * nrOfAgents
        if ubVector is None:
            ubVector = [upperbound] * nrOfAgents
        if solutionDict is None:
            solutionDict = {origin: {destination:
                                         {tourID: max(lbVector[tourID], 0) for tourID in
                                          ODtupledict[origin, destination]}
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
            self.markedODs = [(origin, destination) for origin in zones for destination in zones]
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
                value += abs(ODval - interceptDF.at[origin, destination])
        for tourID, ODlist in tourODsDict.items():
            totalTourWeight = 0
            for OD in ODlist:
                totalTourWeight += self.solution[OD[0]][OD[1]][tourID]
            value += abs(totalTourWeight / len(ODlist) - 1) / dimX
        self.value = value

    def tripBasedLPBound(self):
        for OD in self.markedODs:

            interceptValue = interceptDF.at[OD[0], OD[1]]
            tourIDList = self.solution[OD[0]][OD[1]].keys()
            tempSolution = [self.lbVector[tourID] for tourID in tourIDList]
            upperBounds = [self.ubVector[tourID] for tourID in tourIDList]
            probList = [toursDF.at[tourID, "prob_auto"] for tourID in tourIDList]
            n = len(probList)
            # the two list provide the order in which the tours should be handled:
            # if the probabilities are [0.1, 0.5, 0.25], the order will be 1->3->2->2->3->1, the first pass is to raise
            # the coefficients to one, the second is to raise it past 1
            ascendingProbIndices = sorted(range(n), key=lambda k: probList[k])
            descendingProbIndices = [ascendingProbIndices[-1 - i] for i in range(n)]
            tempValue = sum(probList[tourID] * tempSolution[tourID] for tourID in tourIDList)
            tempDifference = interceptValue - tempValue
            orderIndex = 0
            # reduce the local difference to 0, first raising to the local solution to 1, then past 1.
            while tempDifference > 0 and orderIndex < 2 * n:
                if orderIndex < n:
                    currentIndex = ascendingProbIndices[orderIndex]
                    currentCoefficient = tempSolution[currentIndex]
                    if currentCoefficient < 1 <= upperBounds[currentIndex]:
                        currentProbability = probList[currentIndex]
                        coefficientIncrease = min(tempDifference / currentProbability, 1 - currentCoefficient)
                        tempSolution[currentIndex] += coefficientIncrease
                        tempDifference -= coefficientIncrease * currentProbability
                        tempValue += coefficientIncrease * currentProbability
                else:
                    currentIndex = descendingProbIndices[orderIndex - n]
                    currentCoefficient = tempSolution[currentIndex]
                    currentProbability = probList[currentIndex]
                    coefficientIncrease = min(tempDifference / currentProbability,
                                              upperBounds[currentIndex] - currentCoefficient)
                    tempSolution[currentIndex] += coefficientIncrease
                    tempDifference -= coefficientIncrease * currentProbability
                    tempValue += coefficientIncrease * currentProbability
                orderIndex += 1
        self.evaluateSolution()


def branchAndBound(ubParamDict, lbParamDict, splitParamDict, bnbParamDict):
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
    makeSparceAdjacencyMatrices()
    upperboundParameterDict = {}
    lowerboundParameterDict = {}
    splitInequalityParameterDict = {}
    branchAndBoundParameterDict = {}
    #branchAndBound(upperboundParameterDict, lowerboundParameterDict,
    #               splitInequalityParameterDict, branchAndBoundParameterDict)
