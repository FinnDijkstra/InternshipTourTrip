import json
import time

import pandas as pd
from multiprocessing import Pool
import scipy
import matplotlib.pyplot as plt
import numpy as np



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

aTO = scipy.sparse.csr_array((1,1))
aOT = scipy.sparse.csr_array((1,1))
aSO = scipy.sparse.csr_array((1,1))
aTS = scipy.sparse.csr_array((1,1))
aST = scipy.sparse.csr_array((1,1))
nOD = scipy.sparse.csr_array((1,1))
nSl = scipy.sparse.csr_array((1,1))
cl = []
tbw = []
tp = []
tComp = []
upperbound = 5
penalty = 1
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

                    if not(origin > interceptSize or destination > interceptSize):
                    #     cd[f"sl{screenlineIndex}"] = 0
                    # else:
                        counts[f"sl{screenlineIndex}"] = interceptDF.loc[origin-1, destination-1]
                        screenlines[f"sl{screenlineIndex}"] = {f"({origin}, {destination})": 1.0}
                        screenlineIndex += 1

    nrOfClusters = len(tourOnODDict)
    baseWeightSum = sum(tourOnODDict[tour][0] for tour in tourOnODDict)
    tourNames = pd.Series(data=range(len(tourDict.keys())), index=tourDict.keys())
    screenlineNames = pd.Series(data=range(len(screenlines.keys())), index=screenlines.keys())
    ODNames = pd.Series(data=range(len(tourOnODDict.keys())), index=tourOnODDict.keys())
    return


def readInModelParams2(interceptPath, screenlinesUsedBool, screenlinesPath, tourDictPath, tourOnODDictPath, sopath,
                       topath, tspath, oopath, sspath):
    global aTO, aOT, aSO, aTS, aST, nOD, nSl, cl, tbw, tp, maxZoneNR, nrOfClusters, baseWeightSum, ODNames, \
        tourNames, screenlineNames, penalty, tComp
    sld = {}
    cd = {}
    with open(tourDictPath, 'r') as file:
        tD = json.load(file)
    with open(tourOnODDictPath, 'r') as file:
        tOnODD = json.load(file)
    if screenlinesUsedBool:
        maxZoneNR = 1400
        with open(screenlinesPath, 'r') as file:
            sld = json.load(file)
        with open(interceptPath, 'r') as file:
            cd = json.load(file)
    else:
        interceptDF = pd.read_csv(interceptPath, sep=";", header=None)
        maxZoneNR = 1400
        interceptSize = interceptDF.index.size
        screenlineIndex = 0
        for origin in range(1,maxZoneNR+1):
            for destination in range(1,maxZoneNR+1):
                if f"({origin}, {destination})" in tOnODD:    # interceptDF.loc[origin-1, destination-1] > 0 and

                    if not(origin > interceptSize or destination > interceptSize):
                    #     cd[f"sl{screenlineIndex}"] = 0
                    # else:
                        cd[f"sl{screenlineIndex}"] = interceptDF.loc[origin-1, destination-1]
                        sld[f"sl{screenlineIndex}"] = {f"({origin}, {destination})": 1.0}
                        screenlineIndex += 1
    nrOfClusters = len(tD)
    baseWeightSum = sum(tD[tour][0] for tour in tD)
    tourNames = pd.Series(data=range(len(tD.keys())), index=tD.keys())
    screenlineNames = pd.Series(data=range(len(sld.keys())), index=sld.keys())
    cl = [0] * screenlineNames.size
    for name, idx in screenlineNames.items():
        cl[idx] = cd[name]
    tbw = [0] * nrOfClusters
    tp = [0] * nrOfClusters
    for name, idx in tourNames.items():
        tbw[idx] = tD[name][0]
        tp[idx] = tD[name][-1]
    ODNames = pd.Series(data=range(len(tOnODD.keys())), index=tOnODD.keys())
    aTO = scipy.sparse.load_npz(topath)
    aOT = aTO.tocsc(copy=True).transpose()
    aSO = scipy.sparse.load_npz(sopath)
    aTS = scipy.sparse.load_npz(tspath)
    tComp = [1/(max(1,aTS._getrow(tourIdx).indices.size)*baseWeightSum) for tourIdx in range(nrOfClusters)]
    aST = aTS.tocsc(copy=True).transpose()
    nOD = scipy.sparse.load_npz(oopath)
    nSl = scipy.sparse.load_npz(sspath)
    penalty = max(aTS._getrow(rowID).indices.size for rowID in range(nrOfClusters))
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


def process_trip(params):

    tourOrder, slSol, slVal = lowerboundClass.optimizeTrip(params)
    return params[-1], tourOrder, slSol, slVal



class upperboundClass:
    def __init__(self, ubParamDict):
        ubMeth = ubParamDict.get("method", "tabooSearch")
        solution = ubParamDict.get("solution", [])
        baseUb1 = [upperbound*baseWeight for baseWeight in tbw]
        ubVector = ubParamDict.get("ubVec", baseUb1)
        lbVector = ubParamDict.get("lbVec", [0] * nrOfClusters)
        newConstraint = ubParamDict.get("newConstraint", False)
        value = ubParamDict.get("value", 0)
        ubMethodParameters = ubParamDict.get("method", {})
        if not ubMethodParameters:
            if ubMeth == "tabooSearch":
                ubMethodParameters = {"maxDepth": 150000, "tabooLength": 100, "maxNoImprovement": 80}
        if not solution:
            solution = [max(lbVector[tourID], min(tbw[tourID], ubVector[tourID]))
                                     for tourID in range(nrOfClusters)]
        self.ubMethod = ubMeth
        self.solution = solution
        self.ubMethodParameters = ubMethodParameters
        self.value = value
        self.lbVector = lbVector
        self.ubVector = ubVector
        self.newConstraint = newConstraint

        if self.newConstraint:
            self.updateSolutions(ubParamDict.get("Constraint", (0,0,0)))

    def updateSolutions(self, Constraint):
        side, tourID, value = Constraint
        if side * self.solution[tourID] > side * value:
            self.solution[tourID] = value

    def bound(self):
        if self.ubMethod == "tabooSearch":
            self.solution, self.value = self.tabooSearch(startingWeights=self.solution)

    def tabooSearch(self, startingWeights=None):
        if startingWeights is None:
            minWeights = self.solution
        else:
            minWeights = startingWeights
        curWeights = minWeights
        minValue, solCounts = self.evaluateSolution(curWeights)
        tempValue = minValue
        # first make matrix of objective change when in/decrementing a tour by 1
        timeBeforeDiff = time.time()

        changeDiffList = [0] * 2*nrOfClusters
        for idx in range(2*nrOfClusters):
            changeDiffList[idx] = self.calculateImprovement(curWeights,changeIdx=idx % nrOfClusters,
                                                            changeSide=2*(idx // nrOfClusters)-1, solCounts=solCounts)


        depth = 0
        lastImprovement = 0
        sumOfSizes = 0
        tabooList = []
        improvementMoment = 0
        improvementCount = 0
        timeList = [0]
        valueList = [minValue]
        timeBeforeLoop = time.time()
        print(f"Created difference vector in {timeBeforeLoop - timeBeforeDiff:.3f} seconds")
        timeNow = time.time()
        print(f"{minValue:.4f} ({timeNow - timeBeforeLoop:.3f}s)")
        while (depth < self.ubMethodParameters["maxDepth"] and
               lastImprovement < self.ubMethodParameters["maxNoImprovement"]):
            # find best improvement
            nextStepValue = min(changeDiffList)
            nextStepIdx = changeDiffList.index(nextStepValue)
            nextStepTIdx = nextStepIdx % nrOfClusters
            nextStepSideBase = nextStepIdx // nrOfClusters
            nextStepSide = 2*nextStepSideBase-1


            # update taboolist and start list of steps that need to be checked
            tabooIdx = nextStepIdx + nrOfClusters % nrOfClusters*2
            updateDiffs = {nextStepTIdx}
            if tabooIdx in tabooList:
                tabooList.remove(tabooIdx)
            tabooList.append(tabooIdx)
            if len(tabooList) >= self.ubMethodParameters["tabooLength"]:
                updateDiffs.add(tabooList.pop(0) % nrOfClusters)

            # update current Solution
            tempValue += nextStepValue
            curWeights[nextStepTIdx] += nextStepSide
            for screenlineIdx in aTS._getrow(nextStepTIdx).indices:
                solCounts[screenlineIdx] += nextStepSide
            updateDiffs.update(nSl._getrow(nextStepTIdx).indices)

            # check if we found a new best solution
            if tempValue < minValue:
                minValue = tempValue
                minWeights = curWeights
                lastImprovement = 0
                improvementCount += 1

                if depth-improvementMoment >= 1000:
                    improvementMoment = depth
                    timeNow = time.time()
                    timeList.append(timeNow-timeBeforeLoop)
                    valueList.append(minValue)
                    print(f"{minValue:.4f} ({timeNow-timeBeforeLoop:.3f}s, average set size {sumOfSizes/depth:.1f}, {
                            improvementCount*100/(depth+1):.3f}% of steps are improvements, {-improvementCount+depth+1} non improvements steps)")
            else:
                lastImprovement += 1

            # update change vector
            for tourIdx in updateDiffs:
                for sideBase in [0,1]:
                    changeDiffList[tourIdx+sideBase*nrOfClusters] = (
                        self.calculatePenalizedImprovement(curWeights, tourIdx, sideBase, solCounts, tabooList))

            # incriment depth
            sumOfSizes += 2*len(updateDiffs)
            depth += 1
        plt.plot(timeList, valueList)
        print(depth)
        print(lastImprovement)
        return minWeights, minValue






    def calculateImprovement(self, weights, changeIdx, changeSide, solCounts):
        tempDifference = 1.0 / baseWeightSum
        if changeSide*weights[changeIdx] < changeSide*tbw[changeIdx]:
            tempDifference *= -1
        for screenlineIdx in aTS._getrow(changeIdx).indices:
            curAbs = abs(solCounts[screenlineIdx]-cl[screenlineIdx])
            newAbs = abs(solCounts[screenlineIdx]+changeSide-cl[screenlineIdx])
            tempDifference += newAbs-curAbs
        if weights[changeIdx]+changeSide < self.lbVector[changeIdx] or weights[changeIdx]+changeSide > self.ubVector[changeIdx]:
            tempDifference += 16*penalty
        return tempDifference


    def calculatePenalizedImprovement(self, weights, changeIdx, changeSideBase, solCounts, tabooList):
        tempDifference = self.calculateImprovement(weights, changeIdx, 2*changeSideBase-1, solCounts)
        if changeIdx + changeSideBase*nrOfClusters in tabooList:
            tempDifference += 4*penalty
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
        sT = time.time()
        if solutionList is None:
            solutionList = self.solution
        value = sum(abs(solutionList[idx] - tbw[idx]) for idx in range(nrOfClusters)) / baseWeightSum
        solCounts = [0]*screenlineNames.size
        for screenlineIdx in range(screenlineNames.size):
            toursInScreenline = aST._getrow(screenlineIdx).indices
            solCount = sum(solutionList[tour] for tour in toursInScreenline)
            solCounts[screenlineIdx] = solCount
            value += abs(solCount - cl[screenlineIdx])
        eT = time.time()
        print(f"Evaluated Solution in {eT-sT:.3f} seconds")
        return value, solCounts


class lowerboundClass:
    # newConstraint = [+/- 1, tourID, value]. 1 for upperbound, -1 for lowerbound
    def __init__(self, lbParamDict):
        self.lbVector = lbParamDict.get("lbVector", [0] * nrOfClusters)
        self.ubVector = lbParamDict.get("ubVector", [upperbound * tbw[idx] for idx in range(nrOfClusters)])
        self.solution = lbParamDict.get("solutionBase",
                                        [min(self.ubVector[idx], max(tbw[idx],self.lbVector[idx]))
                                         for idx in range(nrOfClusters)])
        self.solutionFinal = []
        if "solution" in lbParamDict:
            self.firstRun = False
            self.solutionFinal = lbParamDict["solution"]
        else:
            self.firstRun = True
        self.lbMethod = lbParamDict.get("method", "screenlineBasedLP")
        self.value = lbParamDict.get("value",0)
        self.newConstraint = lbParamDict.get("newConstraint",(0,0,0))
        self.markedSls = []
        self.markSls()

    def markSls(self):
        if self.firstRun:
            self.markedSls = list(range(screenlineNames.size))
        else:
            side, tourID, value = self.newConstraint
            self.markedSls = [slIdx for slIdx in aTS._getrow(tourID).indices]

    def bound(self):
        if self.lbMethod == "screenlineBasedLP":
            self.screenlineBasedLPBound()
        else:
            print(f"The lowerbound method '{self.lbMethod}' is not supported")

    def evaluateSolution(self):
        solCounts = []
        for slIdx in range(screenlineNames.size):
            slCount = 0
            for tourIdx, weight in aST._getrow(slIdx):
                slCount += weight * self.solution[tourIdx]
            solCounts.append(slCount)
        value = sum(abs(solCounts[slIdx]-cl[slIdx]) for slIdx in range(screenlineNames.size))
        value += sum(abs(self.solution[tourIdx]-tbw[tourIdx]) for tourIdx in range(nrOfClusters))/baseWeightSum
        self.value = value
        return solCounts





    def screenlineBasedLPBound(self):
        objVal = 0
        if not self.solutionFinal:
            self.solutionFinal = aTS.copy()

        # Prepare arguments for each slIdx
        tasks = [(self.listMakerSls(slIdx)) for slIdx in self.markedSls]

        # Use multiprocessing.Pool to parallelize the processing of trips
        with Pool() as pool:
            # Map each task to a process in the pool and gather results
            results = pool.map(process_trip, tasks)

        # Update solutionFinal and accumulate objVal with results
        dataList = []
        rowList = []
        colList = []
        for slIdx, tourOrder, slSol, slVal in results:
            objVal += slVal
            dataList += slSol
            rowList += [slIdx]*len(tourOrder)
            colList += tourOrder
        self.solutionFinal = scipy.sparse.csr_matrix((dataList, (rowList, colList)))

        self.value = objVal
        # objVal = 0
        # if not self.solutionFinal:
        #     self.solutionFinal = aTS.copy()
        # for slIdx in self.markedSls:
        #     tourOrder, slSol, slVal = self.optimizeTrip(slIdx)
        #     objVal += slVal
        #     for tourIdx in range(len(tourOrder)):
        #         self.solutionFinal[tourOrder[tourIdx],slIdx] = slSol[tourIdx]
        # self.value = objVal


    def listMakerSls(self, slIdx):
        count = cl[slIdx]
        tourRow = aST._getrow(slIdx)
        columnList = tourRow.indices
        n = len(columnList)
        valueList = [tComp[tourIdx] / (tp[tourIdx] * tourRow[0, tourIdx]) for tourIdx in columnList]
        ascendingDensities = sorted(range(n), key=lambda k: valueList[k])
        ascendingDensitiesKeys = [columnList[idx] for idx in ascendingDensities]
        curSol = [self.solution[tourIdx] for tourIdx in ascendingDensitiesKeys]
        influence = [tourRow[0, tourIdx] for tourIdx in ascendingDensitiesKeys]
        tbwLocal = [tbw[tourIdx] for tourIdx in ascendingDensitiesKeys]
        tCompLocal = [tComp[tourIdx] for tourIdx in ascendingDensitiesKeys]
        ubVecLocal = [self.ubVector[tourIdx]for tourIdx in ascendingDensitiesKeys]
        lbVecLocal = [self.lbVector[tourIdx] for tourIdx in ascendingDensitiesKeys]
        return (count, n, ascendingDensities, ascendingDensitiesKeys, curSol, influence, tbwLocal, tCompLocal,
                ubVecLocal, lbVecLocal, slIdx)

    @staticmethod
    def optimizeTrip(paramsInput):
        Linear = True
        (count, n, ascendingDensities, ascendingDensitiesKeys, curSol, influence, tbwLocal, tCompLocal,
         ubVecLocal, lbVecLocal, slIdx) = paramsInput
        curCount = sum(curSol[tourIdx] * influence[tourIdx] for tourIdx in range(n))

        if curCount < count:
            for tourIdx in range(n):
                ubTour = ubVecLocal[tourIdx]
                while curSol[tourIdx] < ubTour:
                    if curCount + influence[tourIdx] < count:
                        curCount += influence[tourIdx]
                        curSol[tourIdx] += 1
                    else:
                        if Linear:
                            curSol[tourIdx] += influence[tourIdx] / (count - curCount)
                            return ascendingDensitiesKeys, curSol, sum(abs(curSol[tourIdx] - tbwLocal[tourIdx]) * tCompLocal[tourIdx]
                                                                   for tourIdx in range(n))
                        else:
                            if curCount + influence[tourIdx]-count + tCompLocal[tourIdx] < count-curCount:
                                curCount += influence[tourIdx]
                                curSol[tourIdx] += 1
                                return ascendingDensitiesKeys, curSol, curCount -count + sum(
                                    abs(curSol[tourIdx] - tbwLocal[tourIdx]) * tCompLocal[tourIdx]
                                    for tourIdx in range(n))
                            else:
                                return ascendingDensitiesKeys, curSol,count-curCount + sum(
                                    abs(curSol[tourIdx] - tbwLocal[tourIdx]) * tCompLocal[tourIdx]
                                    for tourIdx in range(n))


        elif curCount > count:
            for tourIdx in range(n):
                lbTour = lbVecLocal[tourIdx]
                while curSol[tourIdx] > lbTour:
                    if curCount - influence[tourIdx] > count:
                        curCount -= influence[tourIdx]
                        curSol[tourIdx] -= 1
                    else:
                        if Linear:
                            curSol[tourIdx] -= influence[tourIdx] / (curCount - count)
                            return ascendingDensitiesKeys, curSol, sum(abs(curSol[tourIdx] - tbwLocal[tourIdx]) * tCompLocal[tourIdx]
                                                                   for tourIdx in range(n))
                        else:
                            if count + influence[tourIdx]-curCount + tCompLocal[tourIdx] < curCount-count:
                                curCount -= influence[tourIdx]
                                curSol[tourIdx] -= 1
                                return ascendingDensitiesKeys, curSol, count - curCount + sum(
                                    abs(curSol[tourIdx] - tbwLocal[tourIdx]) * tCompLocal[tourIdx]
                                    for tourIdx in range(n))
                            else:
                                return ascendingDensitiesKeys, curSol, curCount-count + sum(
                                    abs(curSol[tourIdx] - tbwLocal[tourIdx]) * tCompLocal[tourIdx]
                                    for tourIdx in range(n))
        return ascendingDensitiesKeys, curSol, abs(curCount - count) + sum(
            abs(curSol[tourIdx] - tbwLocal[tourIdx]) * tCompLocal[tourIdx] for tourIdx in range(n))


def branchAndBound(ubParamDict, lbParamDict, splitParamDict, bnbParamDict):
    ub = upperboundClass(ubParamDict)
    lb = lowerboundClass(lbParamDict)
    splitIneq = splitInequalityClass(splitParamDict)

    return


if __name__ == '__main__':
    parametersType = 2
    if parametersType == 1:
        interceptFile = "NormObservedMatrix.txt"
        screenlinesUsed = False
        screenlinesFile = "NormObservedMatrix.txt"
        tourDictFile = "clusters.json"
        tourOnODDictFile = "clusterOnODDict.json"

        readInModelParams(interceptFile, screenlinesUsed, screenlinesFile, tourDictFile, tourOnODDictFile)
        makeSparceAdjacencyMatrices()
    else:
        interceptFile = "NormObservedMatrix.txt"
        screenlinesUsed = False
        screenlinesFile = "NormObservedMatrix.txt"
        tourDictFile = "clusters.json"
        tourOnODDictFile = "clusterOnODDict.json"
        adjsofile = "adjSlOD.npz"
        adjtofile = "adjTourOD.npz"
        adjtsfile = "adjTourSl.npz"
        neighofile = "neighboursOD.npz"
        neighsfile = "neighboursSl.npz"
        startTime = time.time()
        readInModelParams2(interceptFile, screenlinesUsed, screenlinesFile, tourDictFile, tourOnODDictFile,adjsofile,
                           adjtofile, adjtsfile, neighofile, neighsfile)
        readTime = time.time()
        print(f"Read parameters in {readTime - startTime:.3f} seconds")
        upperboundParameterDict = {}
        lowerboundParameterDict = {}
        splitInequalityParameterDict = {}
        lowerbounder = lowerboundClass(lowerboundParameterDict)
        initTime = time.time()
        print(f"Created Class in {initTime - readTime:.3f} seconds")
        lowerbounder.bound()
        boundTime = time.time()
        print(f"Screenline base lowerbound finished in {boundTime - initTime:.3f} seconds")
        print(lowerbounder.value)
        readTime = time.time()
        upperbounder = upperboundClass(upperboundParameterDict)
        initTime = time.time()
        print(f"Created Class in {initTime - readTime:.3f} seconds")
        upperbounder.bound()
        boundTime = time.time()
        print(f"Taboosearch finished in {boundTime - initTime:.3f} seconds")
        print(upperbounder.value)
        plt.show()
        branchAndBoundParameterDict = {}

    #branchAndBound(upperboundParameterDict, lowerboundParameterDict,
    #               splitInequalityParameterDict, branchAndBoundParameterDict)
