import copy
import json
import time

import pandas as pd
from multiprocessing import Pool
import scipy
import matplotlib.pyplot as plt
import numpy as np
import gzip
from line_profiler import LineProfiler
from numpy.f2py.auxfuncs import throw_error

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
# nSl = scipy.sparse.csr_array((1,1))
cl = np.empty(1)
tbw = np.empty(1)
tp = np.empty(1)
tComp = np.empty(1)
slMaxDiff = np.empty(1)
baseUb1 = np.empty(1)
slSizes = np.empty(1)
slsOnTour = np.empty(1)

upperbound = 5
penalty = 1
measuringBool = False
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

def readGZJson(jsonfilename):
    with gzip.open(jsonfilename, 'r') as file:
        json_bytes = file.read()

    json_str = json_bytes.decode('utf-8')
    return json.loads(json_str)

def getRow(matrix, rowIdx):
    idxStart = matrix.indptr[rowIdx]
    idxEnd = matrix.indptr[rowIdx+1]
    return matrix.indices[idxStart:idxEnd].copy(), matrix.data[idxStart:idxEnd].copy()


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
            screenlinesPre = json.load(file)
        with open(interceptPath, 'r') as file:
            counts2 = json.load(file)

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

    ODNames = pd.Series(data=range(len(tourOnODDict.keys())), index=tourOnODDict.keys())
    if screenlinesUsedBool:
        screenlines2 = {slID: {OD:value for OD, value in slDict.items() if OD in ODNames.index}
                       for slID, slDict in screenlinesPre.items()}
        slNameSet = set(screenlines2.keys()) & set(counts2.keys())
        screenlines = {slID:screenlines2[slID] for slID in slNameSet}
        counts = {slID:counts2[slID] for slID in slNameSet}
    screenlineNames = pd.Series(data=range(len(screenlines.keys())), index=screenlines.keys())
    slJson = json.dumps(screenlines, indent=4)
    with open("ScreenlinesDiscreetV2.json", "w") as outfile:
        outfile.write(slJson)
    return


def readInModelParams2(interceptPath, screenlinesUsedBool, screenlinesPath, tourDictPath, tourOnODDictPath, sopath,
                       topath, tspath, oopath, sspath):
    global aTO, aOT, aSO, aTS, aST, nOD, cl, tbw, tp, maxZoneNR, nrOfClusters, baseWeightSum, ODNames, \
        tourNames, screenlineNames, penalty, tComp, slMaxDiff, baseUb1, slSizes, slsOnTour
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

    tourNames = pd.Series(data=range(len(tD.keys())), index=tD.keys())
    screenlineNames = pd.Series(data=range(len(sld.keys())), index=sld.keys())
    cl = np.empty(screenlineNames.size)
    for name, idx in screenlineNames.items():
        cl[idx] = cd[name]
    tbw = np.empty(nrOfClusters)
    tp = np.empty(nrOfClusters)
    for name, idx in tourNames.items():
        tbw[idx] = tD[name][0]
        tp[idx] = tD[name][-1]
    baseUb1 = tbw*upperbound
    baseWeightSum = np.sum(tbw)
    ODNames = pd.Series(data=range(len(tOnODD.keys())), index=tOnODD.keys())
    aTO = scipy.sparse.load_npz(topath)
    aOT = aTO.tocsc(copy=True).transpose()
    aSO = scipy.sparse.load_npz(sopath)
    aTS = scipy.sparse.load_npz(tspath)
    slsOnTour = np.array([max(1,aTS._getrow(tourIdx).indices.size) for tourIdx in range(nrOfClusters)])
    tComp = np.array([1/(max(1,aTS._getrow(tourIdx).indices.size)*baseWeightSum) for tourIdx in range(nrOfClusters)])
    aST = aTS.tocsc(copy=True).transpose()
    nOD = scipy.sparse.load_npz(oopath)
    # nSl = scipy.sparse.load_npz(sspath)
    penalties = aTS.sum(axis=1)
    penalty = aTS.sum(axis=1).max()
    slMaxDiff = aST.max(axis=1).data
    slSizes = np.array([aST._getrow(idx).size for idx in range(screenlineNames.size)])
    # slMaxDiff = slMaxDiffCoo.tolist()

    # penalty = max(sum(aTS[rowID, colID] for colID in aTS._getrow(rowID).indices) for rowID in range(nrOfClusters))
    # slMaxDiff = [max(aST[rowID,colID] for colID in aST._getrow(rowID).indices) for rowID in range(screenlineNames.size)]
    return





def makeSparceAdjacencyMatrices():
    tourNo = len(tourNames)
    ODNo = len(ODNames)
    slNo = len(screenlineNames)



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
            valueList2.append(screenlines[slID][ODNames.index[ODIdx]])
    adjSlOD = scipy.sparse.csr_array((np.array(valueList2), (np.array(rowList2), np.array(columnList2))),
                                     shape=(slNo, ODNo))
    adjTourSl = adjTourOD @ (adjSlOD.transpose())
    neighboursOD = adjTourOD @ (adjTourOD.transpose())
    # neighboursSl = adjTourSl @ (adjTourSl.transpose())
    # data = []
    # columnlist = []
    # rowlist = []
    # for screenlineIdx in range(tourNames.size):
    #     toursInScreenline = adjTourSl._getrow(screenlineIdx)
    #     neighboursOfTour = adjTourSl @ toursInScreenline.transpose()
    #     rowlist += [screenlineIdx]*neighboursOfTour.size
    #     columnlist += neighboursOfTour.indices.tolist()
    #     data += neighboursOfTour.data.tolist()
    #     x = 1
    #     # for screenlineIdx2 in range(tourNames.size):
    #     #     toursInScreenline2 = adjTourSl._getrow(screenlineIdx2)
    #     #     value = (toursInScreenline @ toursInScreenline2.transpose())[0,0]
    #     #     if value > 0:
    #     #         data.append(value)
    #     #         columnlist.append(screenlineIdx2)
    #     #         rowlist.append(screenlineIdx2)
    # neighboursSl = scipy.sparse.csr_array((data,(rowlist, columnlist)))
    scipy.sparse.save_npz("adjTourOD",adjTourOD)
    scipy.sparse.save_npz("adjSlOD", adjSlOD)
    scipy.sparse.save_npz("adjTourSl", adjTourSl)
    scipy.sparse.save_npz("neighboursOD", neighboursOD)
    # scipy.sparse.save_npz("neighboursSl", neighboursSl)
    return


def process_trip(params):

    tourOrder, slSol, slVal = lowerboundClass.optimizeTrip(params)
    return params[-1], tourOrder, slSol, slVal



class upperboundClass:
    def __init__(self, ubParamDict):
        ubMeth = ubParamDict.get("method", "tabooSearch")
        # solution = ubParamDict.get("solution", [])

        ubVector = ubParamDict.get("ubVec", baseUb1.copy())
        lbVector = ubParamDict.get("lbVec", np.zeros(nrOfClusters))
        newConstraint = ubParamDict.get("newConstraint", False)
        value = ubParamDict.get("value", 0)
        ubMethodParameters = ubParamDict.get("methodParameters", {})
        if not ubMethodParameters:
            if ubMeth == "tabooSearch":
                ubMethodParameters = {"maxDepth": 750000, "tabooLength": 1000, "maxNoImprovement": 800, "maxTime": 600,
                                      "printDepth": 10000, "recallDepth": 25000}
        if "solution" not in ubParamDict:
            self.solution = np.maximum(lbVector, np.minimum(tbw, ubVector))
        else:
            self.solution = ubParamDict.get("solution", np.zeros(nrOfClusters))
        self.ubMethod = ubMeth
        # self.solution = solution
        self.ubMethodParameters = ubMethodParameters
        self.value = value
        self.lbVector = lbVector
        self.ubVector = ubVector
        self.newConstraint = newConstraint
        self.updateBool = ubParamDict.get("updateBool", True)
        self.basicUpdateBool = ubParamDict.get("basicUpdateBool", True)
        self.boundNecessary = True

        if self.newConstraint:
            self.updateSolutions(ubParamDict.get("Constraint", (0,0,0)))

    def updateSolutions(self, Constraint):
        side, tourID, value = Constraint
        if side * self.solution[tourID] > side * value:
            self.solution[tourID] = value
            self.boundNecessary = True
        else:
            self.boundNecessary = False

    def bound(self):
        if self.ubMethod == "tabooSearch" and self.boundNecessary:
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


        changeDiffList = self.initDiffList(curWeights, solCounts)
        # changeDiffList = [0] * 2*nrOfClusters
        # for idx in range(2*nrOfClusters):
        #     changeDiffList[idx] = self.calculateImprovement(curWeights,changeIdx=idx % nrOfClusters,
        #                                                     changeSide=2*(idx // nrOfClusters)-1, solCounts=solCounts)

        maxDepth = self.ubMethodParameters["maxDepth"]
        maxTime = self.ubMethodParameters["maxTime"]
        maxNoImprovement = self.ubMethodParameters["maxNoImprovement"]
        printDepth = self.ubMethodParameters["printDepth"]
        recallDepth = self.ubMethodParameters["recallDepth"]

        depth = 0
        lastImprovement = 0
        sumOfSizes = 0
        sumOfPotentialSizes = 0
        sumOfMinTimes = 0
        sumOfUpdateTimes = 0
        tabooList = []
        improvementMoment = 0
        recalMoment = 0
        improvementCount = 0
        oldSum = 0
        newSum = 0

        if self.updateBool:
            timeList = np.zeros(maxDepth)
            # faults = 1
            valueList = np.zeros(maxDepth)
            valueList[0] = minValue
        else:
            timeList = np.empty(1)
            valueList = np.empty(1)
        timeBeforeLoop = time.time()
        stopTime = timeBeforeLoop + maxTime
        # print(f"Created difference vector in {timeBeforeLoop - timeBeforeDiff:.3f} seconds")
        timeNow = time.time()
        if self.updateBool:
            print(f"{minValue:.4f} ({timeNow - timeBeforeLoop:.3f}s)")
        while (depth < maxDepth and
               lastImprovement < maxNoImprovement
               and time.time() < stopTime):
            # find best improvement
            minStartTime = time.time()
            nextStepIdx = np.argmin(changeDiffList)
            nextStepTupIdx = np.unravel_index(nextStepIdx, changeDiffList.shape)
            nextStepValue = changeDiffList[nextStepTupIdx]
            nextStepTIdx = nextStepTupIdx[1]
            nextStepSideBase = nextStepTupIdx[0]
            nextStepSide = 2 * nextStepSideBase - 1
            # nextStepValue = min(changeDiffList)
            # nextStepIdx = changeDiffList.index(nextStepValue)
            # nextStepTIdx = nextStepIdx % nrOfClusters
            # nextStepSideBase = nextStepIdx // nrOfClusters
            # nextStepSide = 2*nextStepSideBase-1
            sumOfMinTimes += time.time() - minStartTime


            # update taboolist and start list of steps that need to be checked
            tabooIdx = (nextStepIdx + nrOfClusters) % (nrOfClusters*2)
            # updateDiffs = {nextStepTIdx}
            if tabooIdx in tabooList:
                tabooList.remove(tabooIdx)
                changeDiffList[np.unravel_index(tabooIdx, changeDiffList.shape)] -= 4*penalty
            tabooList.append(tabooIdx)
            changeDiffList[np.unravel_index(tabooIdx, changeDiffList.shape)] += 4*penalty
            if len(tabooList) >= self.ubMethodParameters["tabooLength"]:
                removedTaboo = tabooList.pop(0)
                changeDiffList[np.unravel_index(removedTaboo, changeDiffList.shape)] -= 4*penalty
                # updateDiffs.add(removedTaboo % nrOfClusters)

            # Update compensations for step taken
            bwDiff = curWeights[nextStepTIdx]-tbw[nextStepTIdx]
            curAbs = abs(bwDiff)
            incrAbs = abs(bwDiff+1)
            decrAbs = abs(bwDiff-1)
            newDiff = bwDiff+nextStepSide
            newAbs = abs(newDiff)
            newIncrAbs = abs(newDiff+1)
            newDecrAbs = abs(newDiff-1)
            incrComp = ((newIncrAbs-newAbs)-(incrAbs-curAbs))*tComp[nextStepTIdx]
            decrComp = ((newDecrAbs-newAbs)-(decrAbs-curAbs))*tComp[nextStepTIdx]
            changeDiffList[0,nextStepTIdx] += decrComp
            changeDiffList[1,nextStepTIdx] += incrComp

            # Update penalty for upper and lowerbounds
            curStepWeight = curWeights[nextStepTIdx]
            stepLB = self.lbVector[nextStepTIdx]
            stepUB = self.ubVector[nextStepTIdx]
            if nextStepSide == 1:
                if curStepWeight == stepLB:
                    changeDiffList[0,nextStepTIdx] -= 16*penalty
                if curStepWeight + 1 == stepUB:
                    changeDiffList[1,nextStepTIdx] += 16*penalty
            else:
                if curStepWeight - 1 == stepLB:
                    changeDiffList[0,nextStepTIdx] += 16*penalty
                if curStepWeight == stepUB:
                    changeDiffList[1,nextStepTIdx] -= 16*penalty

            updateStartTime = time.time()
            if measuringBool:
                lp = LineProfiler()
                lp_wrapper = lp(self.updateCurrentSolution)
                sizes, potentialSizes, newTime, oldTime, forBool = lp_wrapper(solCounts, nextStepSide, nextStepTIdx, changeDiffList)

                if forBool >= 5:
                    lp.print_stats()
                    print("hmm")
            else:
                sizes, potentialSizes, newTime, oldTime, forBool \
                    = self.updateCurrentSolution(solCounts, nextStepSide, nextStepTIdx, changeDiffList)
            sumOfUpdateTimes += time.time() - updateStartTime
            oldSum += oldTime
            newSum += newTime
            sumOfSizes += sizes
            sumOfPotentialSizes += potentialSizes
            # update current Solution
            tempValue += nextStepValue
            curWeights[nextStepTIdx] += nextStepSide

            # for screenlineIdx in aTS._getrow(nextStepTIdx).indices:
            #     solCounts[screenlineIdx] += nextStepSide * aTS[nextStepTIdx, screenlineIdx]
            #     updateDiffs.update(aST._getrow(screenlineIdx).indices)

            # check if we found a new best solution
            if tempValue < minValue:
                minValue = tempValue
                minWeights = curWeights
                lastImprovement = 0
                improvementCount += 1
                # val, sol = self.evaluateSolution(curWeights)
                # if abs(val - minValue) > 0.1:
                #     print("non matching values")
                #     faults += 1

                if depth-improvementMoment >= printDepth and self.updateBool:
                    improvementMoment = depth



                    timeNow = time.time()
                    timeList[depth] = timeNow - timeBeforeLoop
                    valueList[depth] = minValue

                    print(f"{minValue:.4f} (best step / updating time per thousand/ total times: {sumOfMinTimes/depth*1000:.3f}s/ {
                            sumOfUpdateTimes/depth*1000:.3f}s/ {timeNow-timeBeforeLoop:.3f}s, average set size {
                            sumOfSizes/depth:.1f} (without logic {sumOfPotentialSizes/depth:.1f}), \n {
                            improvementCount*100/(depth+1):.3f}% of steps are improvements, {-improvementCount+depth+1} non improvements steps)")
                if depth-recalMoment >= recallDepth:
                    recalMoment = depth
                    newDiffList = self.initDiffList(curWeights, solCounts)
                    newDiffList[np.unravel_index(tabooList, changeDiffList.shape)] += 4 * penalty
                    mask = (np.abs(changeDiffList - newDiffList) > 0.00002)
                    if np.any(mask):
                        print([changeDiffList[mask], newDiffList[mask]])
                        print("uh oh")
                    changeDiffList = newDiffList
                    minValue = self.evaluateSolution(minWeights)[0]
            else:
                lastImprovement += 1

            depth += 1
        if self.updateBool:
            depthsOfImprovement = valueList.nonzero()
            plt.plot(timeList[depthsOfImprovement], valueList[depthsOfImprovement])
            plt.show()
        if self.basicUpdateBool:
            print(f"Taboo finished in {time.time()-timeNow} with depth:{depth}, last improvement:{
                        lastImprovement}/{maxNoImprovement}")
        return minWeights, minValue




    def initDiffList(self, curWeights, solCounts):
        diffArray = solCounts-cl
        absDiffArray = np.abs(diffArray)
        curComp = curWeights-tbw
        absCurComp = np.abs(curComp)
        localATS = aTS.copy()
        nz = localATS.nonzero()
        localATS[nz] += diffArray[nz[1]]
        slImprovementCSRIncr = localATS.multiply(localATS.sign())
        slImprovementCSRIncr[nz] -= absDiffArray[nz[1]]
        localATS[nz] -= 2*diffArray[nz[1]]
        slImprovementCSRDecr = localATS.multiply(localATS.sign())
        slImprovementCSRDecr[nz] -= absDiffArray[nz[1]]
        absIncrComp = np.abs(curComp+1)
        IncrCompDiff = (absIncrComp - absCurComp)/baseWeightSum
        DecrCompDiff = -IncrCompDiff
        IncrDiffArray = slImprovementCSRIncr.sum(axis=1) + IncrCompDiff
        decrDiffArray = slImprovementCSRDecr.sum(axis=1) + DecrCompDiff
        diffArray = np.stack((decrDiffArray, IncrDiffArray), axis=0)
        mask = (curWeights == self.lbVector)
        diffArray[0, mask] += 16 * penalty
        mask = (curWeights == self.ubVector)
        diffArray[1, mask] += 16 * penalty
        return diffArray





    def calculateImprovement(self, weights, changeIdx, changeSide, solCounts):
        tempDifference = 1.0 / baseWeightSum
        if changeSide*weights[changeIdx] < changeSide*tbw[changeIdx]:
            tempDifference *= -1



        deltaSLArray = aTS._getrow(changeIdx).toarray()[0]
        diffArray = solCounts - cl
        curAbs = np.linalg.norm(diffArray, ord=1)
        newAbs = np.linalg.norm(diffArray+changeSide*deltaSLArray, ord=1)
        tempDifference += newAbs - curAbs
        # for screenlineIdx in aTS._getrow(changeIdx).indices:
        #     curAbs = abs(solCounts[screenlineIdx]-cl[screenlineIdx])
        #     newAbs = abs(solCounts[screenlineIdx]+changeSide*aTS[changeIdx,screenlineIdx]-cl[screenlineIdx])
        #     tempDifference += newAbs-curAbs
        if (weights[changeIdx]+changeSide < self.lbVector[changeIdx]
                or weights[changeIdx]+changeSide > self.ubVector[changeIdx]):
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
        value = np.sum(np.abs(solutionList - tbw)) / baseWeightSum


        # solCounts = [0]*screenlineNames.size
        # for screenlineIdx in range(screenlineNames.size):
        #     toursInScreenline = aST._getrow(screenlineIdx).indices
        #     solCount = sum(solutionList[tour] for tour in toursInScreenline)
        #     solCounts[screenlineIdx] = solCount
        #     value += abs(solCount - cl[screenlineIdx])
        solCounts = aST.dot(solutionList)

        absDiff = np.abs(solCounts-cl)
        compValue = absDiff.sum()

        eT = time.time()
        if self.updateBool:
            print(f"Upperbound evaluated solution with difference intercept total: {
                                value:.3f}, compensation factor: {compValue:.3f}"
                                f"Evaluated Solution in {eT-sT:.3f} seconds")
        value += compValue
        return value, solCounts

    def updateCurrentSolution(self, solCounts, nextStepSide, nextStepTIdx, changeDiffList):
        # screenlinesAffected = aTS._getrow(nextStepTIdx)
        slIdxArray, slVal = getRow(aTS, nextStepTIdx)
        slADict = {slIdxArray[idx]: slVal[idx] for idx in range(slVal.size)}
        toursUpdated = 0
        potentialTours = 0
        forBool = 0

        oldTimeSum = 0
        newTimeSum = 0
        for slIdx, stepInfluenceSl in slADict.items():
            countSL = cl[slIdx]
            solSL = solCounts[slIdx]
            # stepInfluenceSl = screenlinesAffected[0, slIdx]
            localSlMaxDiff = slMaxDiff[slIdx]
            prevDiff = countSL - solSL
            newDiff = -prevDiff + stepInfluenceSl*nextStepSide
            potentialTours += 2 * slSizes[slIdx]
            # if the below statement is false, (1) the new solution is too far below the count to need updates or
            # (2) the old solution is too far above.
            # This is under the assumption of increase in solution, decrease the same but reverse
            if (localSlMaxDiff + nextStepSide * newDiff > 0) and (localSlMaxDiff + nextStepSide * prevDiff > 0):
                compFactor = abs(prevDiff) - abs(newDiff)

                # (positive increase) sol is in (count - tourWeight, count + maxWeightOfTour)
                # (negative increase) sol is in (count - maxWeightOfTour, count + tourWeight)
                if nextStepSide * newDiff > 0:
                    # the tours in the reverse direction of the step need to be updated
                    checkSide = -1*nextStepSide
                    checkIdx = ((checkSide + 1) // 2)
                    # rowVector = aST._getrow(slIdx)
                    localTours, influences = getRow(aST, slIdx)
                    influences *= checkSide
                    toursUpdated += slSizes[slIdx]
                    forBool += 1
                    loopStartTime = time.time()
                    # rowVector *= checkSide
                    # rowVector2 = rowVector.copy()
                    # rowNZ = rowVector.nonzero()
                    # rowVector[rowNZ] -= prevDiff
                    # rowVector2[rowNZ] += newDiff
                    # tTotalDelta = (np.abs(rowVector2) - np.abs(rowVector))
                    # tTotalDelta[rowNZ] += compFactor

                    # localTours = rowVector.indices.copy()
                    # influences = checkSide*rowVector.data.copy()
                    influences2 = influences.copy()
                    tTotalDelta = (np.abs(influences2+newDiff) - np.abs(influences-prevDiff)) + compFactor

                    midTime = time.time()
                    changeDiffList[checkIdx, localTours] += tTotalDelta
                    # for checkTour in rowVector.indices:
                    #     checkTourWeight = rowVector[0, checkTour]
                    #     tDelta = checkTourWeight * checkSide
                    #     tTotalDelta = (compFactor + abs(tDelta + newDiff) - abs(tDelta - prevDiff))
                    #     if tTotalDelta != 0:
                    #         changeDiffList[checkSide, checkTour] += tTotalDelta
                    #         toursUpdated += 1
                    # oldTimeSum += time.time()-midTime
                    newTimeSum += midTime - loopStartTime

                # (positive increase) sol is in (count - tourWeight - maxWeightOfTour, count)
                # (negative increase) sol is in (count, count + tourWeight + maxWeightOfTour)
                if nextStepSide * prevDiff > 0:
                    # the tours in the same direction of the step need to be updated
                    checkSide = nextStepSide
                    checkIdx = ((checkSide+1)//2)
                    # rowVector = aST._getrow(slIdx)
                    localTours, influences = getRow(aST, slIdx)
                    influences *= checkSide
                    toursUpdated += slSizes[slIdx]
                    forBool += 1
                    loopStartTime = time.time()
                    # rowVector *= checkSide
                    # rowVector2 = rowVector.copy()
                    # rowNZ = rowVector.nonzero()
                    # rowVector[rowNZ] -= prevDiff
                    # rowVector2[rowNZ] += newDiff
                    # tTotalDelta = (np.abs(rowVector2) - np.abs(rowVector))
                    # tTotalDelta[rowNZ] += compFactor
                    # midTime = time.time()
                    # changeDiffList[checkIdx, rowNZ[1]] += tTotalDelta[rowNZ]

                    # localTours = rowVector.indices.copy()
                    # influences = checkSide*rowVector.data.copy()
                    influences2 = influences.copy()
                    tTotalDelta = (np.abs(influences2 + newDiff) - np.abs(influences - prevDiff)) + compFactor
                    midTime = time.time()
                    changeDiffList[checkIdx, localTours] += tTotalDelta
                    # for checkTour in rowVector.indices:
                    #     checkTourWeight = rowVector[0, checkTour]
                    #     tDelta = checkTourWeight * checkSide
                    #     tTotalDelta = (compFactor + abs(tDelta + newDiff) - abs(tDelta - prevDiff))
                    #     if tTotalDelta != 0:
                    #         changeDiffList[checkSide, checkTour] += tTotalDelta
                    #         toursUpdated += 1
                    # oldTimeSum += time.time()-midTime
                    newTimeSum += midTime-loopStartTime


            solCounts[slIdx] += stepInfluenceSl*nextStepSide

        return toursUpdated, potentialTours, newTimeSum, oldTimeSum, forBool


    def newNodePrep(self, constr):
        self.value = self.value.copy()
        if constr[0] == 1:
            self.ubVector = self.ubVector.copy()
            self.ubVector[constr[1]] = constr[2]
        else:
            self.lbVector = self.lbVector.copy()
            self.lbVector[constr[1]] = constr[2]
        self.solution = self.solution.copy()
        self.updateSolutions(constr)





class lowerboundClass:
    # newConstraint = [+/- 1, tourID, value]. 1 for upperbound, -1 for lowerbound
    def __init__(self, lbParamDict):
        self.lbVector = lbParamDict.get("lbVector", np.zeros(nrOfClusters))
        self.ubVector = lbParamDict.get("ubVector", upperbound*tbw)
        self.solution = lbParamDict.get("solutionBase",
                                        np.minimum(self.ubVector, np.maximum(tbw,self.lbVector)))
        if "solution" in lbParamDict:
            self.firstRun = False
            self.solutionFinal = lbParamDict["solution"]
            if "solutionTranspose" in lbParamDict:
                self.solutionTranspose = lbParamDict["solutionTranspose"]
            else:
                self.solutionTranspose = self.solutionFinal.copy().transpose().tocsr()
        else:
            self.firstRun = True
            self.solutionFinal = aTS.copy()
            nz = self.solutionFinal.nonzero()
            self.solutionFinal[nz] = self.solution[nz[0]]
            self.solutionTranspose = self.solutionFinal.copy().transpose().tocsr()
        self.lbMethod = lbParamDict.get("method", "screenlineBasedLP")
        self.extraParams = lbParamDict.get("extraParams", {})
        if self.lbMethod == "screenlineBasedLP" and not self.extraParams:
            self.extraParams["ValueMatrix"] = aST.power(-1).multiply(np.divide(tComp, tp)).tocsr()
        self.value = lbParamDict.get("value",0)
        self.newConstraint = lbParamDict.get("newConstraint",(0,0,0))
        self.markedSls = np.empty(1)
        self.markSls()
        self.updateBool = lbParamDict.get("updateBool", True)
        self.basicUpdateBool = lbParamDict.get("basicUpdateBool", True)
        if self.firstRun:
            self.prepForRun()


    def prepForRun(self):
        startPrepTime = time.time()
        if measuringBool:
            lp = LineProfiler()
            lp_wrapper = lp(self.listMakerSls)
            tasks = {slIdx:list(lp_wrapper(slIdx)) for slIdx in self.markedSls}
            lp.print_stats()
        else:
            tasks = {slIdx:list(self.listMakerSls(slIdx)) for slIdx in self.markedSls}
        self.extraParams["tasks"] = tasks
        if self.updateBool:
            print(f"created tasklist in {time.time() - startPrepTime:.2f} seconds")


    def markSls(self):
        # Add all Screenlines affected by the new constraint to self.markedSLs
        if self.firstRun:
            self.markedSls = np.arange(screenlineNames.size)
        else:
            side, tourID, value = self.newConstraint
            if self.lbMethod in ["screenlineBasedLP"]:
                indices, values = getRow(self.solutionFinal, tourID)
                # Depending on the added Constraint and the previous solution, only some screenlines need te be recalced
                if side == 1:
                    self.markedSls = indices[values > self.ubVector[tourID]]
                else:
                    self.markedSls = indices[values < self.lbVector[tourID]]
            else:
                self.markedSls = getRow(aTS, tourID)[0]
            for slIdx in self.markedSls:
                self.extraParams["tasks"][slIdx][4][
                    self.extraParams["tasks"][slIdx][3] == tourID] = self.solution[tourID]





    def bound(self):
        if self.lbMethod == "screenlineBasedLP":
            self.screenlineBasedLPBound()
            print(self.markedSls)
        else:
            print(f"The lowerbound method '{self.lbMethod}' is not supported")
        if self.basicUpdateBool:
            print(f"Lowerbound evaluated solution with value {self.value}")


    def evaluateSolution(self):

        solCountMatrix = aST.multiply(self.solutionTranspose)
        solCount2 = solCountMatrix.sum(axis=1)
        # for slIdx in range(screenlineNames.size):
        #     slCount = 0
        #     for tourIdx, weight in aST._getrow(slIdx):
        #         slCount += weight * self.solution[tourIdx]
        #     solCounts.append(slCount)
        clArray = cl
        value = np.sum(np.abs(solCount2-clArray))

        # value = sum(abs(solCounts[slIdx]-cl[slIdx]) for slIdx in range(screenlineNames.size))
        # revTbw = np.divide(np.ones(nrOfClusters), tbw)

        devMatrix = self.solutionTranspose.copy()
        nz = devMatrix.nonzero()
        devMatrix[nz] -= tbw[nz[1]]
        absdevMatrix = devMatrix.multiply(devMatrix.sign()).multiply(tComp)
        compValue = absdevMatrix.sum()

        value += compValue
        # value += sum(abs(self.solution[tourIdx]-tbw[tourIdx])*tComp[tourIdx] for tourIdx in range(nrOfClusters))
        self.value = value
        return solCount2





    def screenlineBasedLPBound(self):
        objVal = 0
        # if not self.solutionFinal:
        #     self.solutionFinal = aTS.copy()

        # Prepare arguments for each slIdx
        tasks = self.extraParams["tasks"]
        sumOfTimes = 0
        results = []

        if measuringBool:
            lp = LineProfiler()
            lp_wrapper = lp(self.optimizeTrip)
            for slIdx in self.markedSls:
                task = tasks[slIdx]
                startTimeSl = time.time()
                results.append([slIdx, task[2]] + list(lp_wrapper(task)))
                endTimeSl = time.time()
                sumOfTimes += (endTimeSl - startTimeSl)
                if (task[-1] + 1) % 100 == 0 and self.updateBool:
                    print(f"average time per screenline is {sumOfTimes / (task[-1] + 1)}")
            lp.print_stats()
        else:
            for slIdx in self.markedSls:
                task = tasks[slIdx]
                startTimeSl = time.time()
                results.append([slIdx, task[2]] + list(self.optimizeTrip(task)))
                endTimeSl = time.time()
                sumOfTimes += (endTimeSl - startTimeSl)
                if (task[-1]+1) % 100 == 0 and self.updateBool:
                    print(f"average time per screenline is {sumOfTimes/(task[-1]+1)}")
        # results = [(task[-1],) + self.optimizeTrip(task) for task in tasks]
        # # Use multiprocessing.Pool to parallelize the processing of trips
        # with Pool() as pool:
        #     # Map each task to a process in the pool and gather results
        #     results = pool.map(process_trip, tasks)

        # Update solutionFinal and accumulate objVal with results

        # Change to updating data based on indptr
        for slIdx, listOrder, tourOrder, slSol, slVal in results:
            startIdx = self.solutionTranspose.indptr[slIdx]
            endIdx = self.solutionTranspose.indptr[slIdx+1]

            self.solutionTranspose.data[startIdx:endIdx] = slSol[listOrder]
            self.solutionTranspose.indices[startIdx:endIdx] = tourOrder

        self.solutionFinal = self.solutionTranspose.copy().transpose().tocsr()


        # dataList = []
        # rowList = []
        # colList = []
        # for slIdx, tourOrder, slSol, slVal in results:
        #     objVal += slVal
        #     dataList += slSol.tolist()
        #     rowList += [slIdx]*len(tourOrder)
        #     colList += tourOrder.tolist()
        # self.solutionFinal = scipy.sparse.csc_matrix((dataList, (rowList, colList))).transpose()

        self.evaluateSolution()
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

        # tourRow = aST._getrow(slIdx)
        #
        # columnList = tourRow.indices

        # startList = time.time()
        # valueList = self.extraParams["ValueMatrix"][[slIdx],columnList].tolist()

        columnList, valueList = getRow(self.extraParams["ValueMatrix"], slIdx)
        n = columnList.size
        # valueList = [tComp[tourIdx] / (tp[tourIdx] * tourRow[0, tourIdx]) for tourIdx in columnList]
        # startSort = time.time()
        ascendingDensities = np.argsort(valueList)
        # ascendingDensities = sorted(range(n), key=lambda k: valueList[k])
        # endSort = time.time()
        ascendingDensitiesKeys = columnList[ascendingDensities]
        curSol = self.solution[ascendingDensitiesKeys]
        influence = getRow(aST,slIdx)[1][ascendingDensities]
        tbwLocal = tbw[ascendingDensitiesKeys]
        tCompLocal = tComp[ascendingDensitiesKeys]
        ubVecLocal = self.ubVector[ascendingDensitiesKeys]
        lbVecLocal = self.lbVector[ascendingDensitiesKeys]
        # ascendingDensitiesKeys = [columnList[idx] for idx in ascendingDensities]
        # curSol = [self.solution[tourIdx] for tourIdx in ascendingDensitiesKeys]
        # influence = [tourRow[0, tourIdx] for tourIdx in ascendingDensitiesKeys]
        # tbwLocal = [tbw[tourIdx] for tourIdx in ascendingDensitiesKeys]
        # tCompLocal = [tComp[tourIdx] for tourIdx in ascendingDensitiesKeys]
        # ubVecLocal = [self.ubVector[tourIdx]for tourIdx in ascendingDensitiesKeys]
        # lbVecLocal = [self.lbVector[tourIdx] for tourIdx in ascendingDensitiesKeys]
        # endList = time.time()
        # print(f"({(startSort-startList):.3f})/ ({(endSort-startSort):.3f})/ ({(endList-endSort):.3f}) total({(endList-startList):.3f}) list size {n}")
        return [count, n, ascendingDensities, ascendingDensitiesKeys, curSol, influence, tbwLocal, tCompLocal,
                ubVecLocal, lbVecLocal, slIdx]

    @staticmethod
    def optimizeTrip(paramsInput):
        Linear = True
        LinearSkip = True
        [count, n, ascendingDensities, ascendingDensitiesKeys, curSol, influence, tbwLocal, tCompLocal,
         ubVecLocal, lbVecLocal, slIdx] = paramsInput
        curCount = curSol.dot(influence)
        maskUb = (curSol > ubVecLocal)
        maskLb = (curSol < lbVecLocal)
        if np.any(maskUb) or np.any(maskLb):
            print(maskUb.nonzero(), maskLb.nonzero())
            x=1
        if curCount < count:
            for tourIdx in range(n):
                ubTour = ubVecLocal[tourIdx]
                curWeight = curSol[tourIdx]
                influenceTour = influence[tourIdx]
                if LinearSkip:
                    delta = (count - curCount) / influenceTour
                    if delta > ubTour-curWeight:
                        curSol[tourIdx] = ubTour
                        curCount += (ubTour - curWeight)*influenceTour
                    else:
                        curCount += delta*influenceTour
                        curSol[tourIdx] += delta
                        return ascendingDensitiesKeys, curSol, np.abs(curSol - tbwLocal).dot(tCompLocal)
                else:
                    while curSol[tourIdx] < ubTour:
                        if curCount + influence[tourIdx] < count:
                            delta = min((count - curCount) / influence[tourIdx], ubTour-curCount)
                            curCount += delta*influence[tourIdx]
                            curSol[tourIdx] += delta
                        else:
                            if Linear:
                                curSol[tourIdx] += (count - curCount) / influence[tourIdx]
                                return ascendingDensitiesKeys, curSol, np.abs(curSol - tbwLocal).dot(tCompLocal)
                            else:
                                if curCount + influence[tourIdx]-count + tCompLocal[tourIdx] < count-curCount:
                                    curCount += influence[tourIdx]
                                    curSol[tourIdx] += 1
                                    return (ascendingDensitiesKeys, curSol, curCount - count
                                            + np.abs(curSol - tbwLocal).dot(tCompLocal))
                                else:
                                    return (ascendingDensitiesKeys, curSol,count-curCount +
                                            np.abs(curSol - tbwLocal).dot(tCompLocal))


        elif curCount > count:
            for tourIdx in range(n):
                lbTour = lbVecLocal[tourIdx]
                curWeight = curSol[tourIdx]
                influenceTour = influence[tourIdx]
                if LinearSkip:
                    delta = (count - curCount) / influenceTour
                    if delta < lbTour - curWeight:
                        curSol[tourIdx] = lbTour
                        curCount += (lbTour - curWeight) * influenceTour
                    else:
                        curCount += delta * influenceTour
                        curSol[tourIdx] += delta
                        return ascendingDensitiesKeys, curSol, np.abs(curSol - tbwLocal).dot(tCompLocal)
                else:
                    while curSol[tourIdx] > lbTour:
                        if curCount - influence[tourIdx] > count:
                            delta = min((curCount-count) / influence[tourIdx], curCount-lbTour)
                            curCount -= delta * influence[tourIdx]
                            curSol[tourIdx] -= delta
                            # curCount -= influence[tourIdx]
                            # curSol[tourIdx] -= 1
                        else:
                            if Linear:
                                curSol[tourIdx] -= (curCount - count) / influence[tourIdx]
                                return ascendingDensitiesKeys, curSol, np.abs(curSol - tbwLocal).dot(tCompLocal)
                            else:
                                if count + influence[tourIdx]-curCount + tCompLocal[tourIdx] < curCount-count:
                                    curCount -= influence[tourIdx]
                                    curSol[tourIdx] -= 1
                                    return (ascendingDensitiesKeys, curSol, count - curCount +
                                            np.abs(curSol - tbwLocal).dot(tCompLocal))
                                else:
                                    return (ascendingDensitiesKeys, curSol, curCount-count +
                                            np.abs(curSol - tbwLocal).dot(tCompLocal))
        return ascendingDensitiesKeys, curSol, abs(curCount - count) + np.abs(curSol - tbwLocal).dot(tCompLocal)

    def newNodePrep(self, constr):
        self.value = self.value.copy()

        self.solutionFinal = self.solutionFinal.copy()
        self.solutionTranspose = self.solutionTranspose.copy()
        self.newConstraint = constr
        if constr[0] == 1:
            self.ubVector = self.ubVector.copy()
            self.ubVector[constr[1]] = constr[2]
        else:
            self.lbVector = self.lbVector.copy()
            self.lbVector[constr[1]] = constr[2]
        self.solution = np.minimum(self.ubVector,np.maximum(self.solution, self.lbVector))
        self.markSls()






class nodeClass:
    def __init__(self, ubParamDict, lbParamDict, splitParamDict, nodeTag):
        self.ineqType = splitParamDict['ineqType']
        self.lbOutputType = splitParamDict['lbOutputType']
        self.ubClass = upperboundClass(ubParamDict)
        self.lbClass = lowerboundClass(lbParamDict)
        self.tag = nodeTag


    def bound(self):
        self.ubClass.bound()
        self.lbClass.bound()
        return self.ubClass.value, self.ubClass.solution, self.lbClass.value


    def findInequality(self):
        if self.ineqType == "tourBased":
            if self.lbOutputType == "csr":
                avgTourWeight = self.lbClass.solutionFinal.mean(axis=1).flatten()
                copySol = self.lbClass.solutionFinal.copy()
                nz = copySol.nonzero()
                copySol[nz] -= avgTourWeight[nz[0]]
                stdDevs = np.sqrt(copySol.multiply(1 / upperbound).power(2).mean(axis=1).flatten())
                splitChoiceVec = np.abs(avgTourWeight - np.round(avgTourWeight)) + stdDevs
            elif self.lbOutputType == "array":
                avgTourWeight = self.lbClass.solution.copy()
                splitChoiceVec = np.abs(avgTourWeight - np.round(avgTourWeight))
            else:
                raise Exception("Not a valid lb output type")
            ineqIndex = np.argmax(splitChoiceVec)
            ubVal = np.floor(avgTourWeight[ineqIndex])
            lbVal = ubVal + 1
            lbConstr = (-1, ineqIndex, lbVal)
            ubConstr = (1, ineqIndex, ubVal)
            return lbConstr, ubConstr
        else:
            raise Exception("Not a valid inequality type")

    def split(self, lbConstr, ubConstr, nodeID):
        nextDepth = self.tag[1] + 1
        lbTag = (nodeID, nextDepth)
        ubTag = (nodeID+1, nextDepth)
        lbNode = copy.copy(self)
        lbNode.newNodePrep(lbConstr, lbTag)
        ubNode = copy.copy(self)
        ubNode.newNodePrep(ubConstr, ubTag)


        return lbNode, lbTag, ubNode, ubTag, nodeID+2


    def newNodePrep(self, constr, tag):
        self.tag = tag
        self.ubClass.newNodePrep(constr)
        self.lbClass.newNodePrep(constr)
        pass


def findBranch(branchMethod, lbValDict, ubValDict, gUb, gLb):
    branchTag = gLb[0]

    if branchMethod == "globalLb":
        return gLb[0]
    elif branchMethod == "bestUb":
        minUb = ubValDict[branchTag]
        for tag, value in ubValDict.items():
            if value < minUb:
                minUb = value
                branchTag = tag
    elif branchMethod == "smallestGap":
        minGap = ubValDict[branchTag]-gLb[-1]
        for tag, lbVal in lbValDict.items():
            ubVal = ubValDict[tag]
            gap = ubVal-lbVal
            if gap < minGap:
                minGap = gap
                branchTag = tag
    elif branchMethod == "worstLb":
        maxLb = gLb[-1]
        for tag, lbVal in lbValDict.items():
            if lbVal > maxLb:
                maxLb = lbVal
                branchTag = tag
    elif branchMethod == "depthFirst":
        for tag in lbValDict.keys():
            if tag[1] > branchTag[1]:
                branchTag = tag
    elif branchMethod == "breadthFirst":
        for tag in lbValDict.keys():
            if tag[1] < branchTag[1]:
                branchTag = tag
    elif branchMethod == "aroundGUb":
        dist = gUb[-1] - gLb[-1]
        for tag, value in ubValDict.items():
            if value-gUb < dist:
                dist = value-gUb
                branchTag = tag
        if dist > 0:
            for tag, value in lbValDict.items():
                if gUb-value < dist:
                    dist = gUb-value
                    branchTag = tag
    return branchTag


def setUpdateBools(bnbParamDict, lbParamDict, ubParamDict, ubDeepParamDict, lbDeepParamDict):
    lbBool = bnbParamDict["lbUpdates"]
    ubBool = bnbParamDict["ubUpdates"]
    lbBasicBool = bnbParamDict["lbBasicUpdates"]
    ubBasicBool = bnbParamDict["ubBasicUpdates"]
    lbParamDict["updateBool"] = lbBool
    lbDeepParamDict["updateBool"] = lbBool
    ubParamDict["updateBool"] = ubBool
    ubDeepParamDict["updateBool"] = ubBool
    lbParamDict["basicUpdateBool"] = lbBasicBool
    lbDeepParamDict["basicUpdateBool"] = lbBasicBool
    ubParamDict["basicUpdateBool"] = ubBasicBool
    ubDeepParamDict["basicUpdateBool"] = ubBasicBool

    return bnbParamDict["branchingUpdates"]



def branchAndBound(ubParamDict, lbParamDict, splitParamDict, bnbParamDict, ubDeepParamDict, lbDeepParamDict):
    nodeID = 0
    nodeTag = (nodeID, 0)
    updateBoolBranch = setUpdateBools(bnbParamDict, lbParamDict, ubParamDict, ubDeepParamDict, lbDeepParamDict)
    # ub = upperboundClass(ubParamDict)
    # lb = lowerboundClass(lbParamDict)
    baseNode = nodeClass(ubParamDict, lbParamDict, splitParamDict, nodeTag)
    branchMeth = bnbParamDict['branchMethod']

    if updateBoolBranch:
        print("Bounding root node, this will take a while")
    ubVal, ubSol, lbVal = baseNode.bound()

    # Initialize bounds and dicts
    globalUb = [nodeTag, ubSol, ubVal]
    globalLb = [nodeTag, lbVal]
    lbValDict = {nodeTag:lbVal}
    ubValDict = {nodeTag:ubVal}
    nodeDict = {nodeTag:baseNode}
    nodeID += 1

    # Set params to non base params
    if "methodParameters" in ubDeepParamDict:
        baseNode.ubClass.ubMethodParameters = ubDeepParamDict["methodParameters"]
    if "methodParameters" in lbDeepParamDict:
        baseNode.lbClass.lbMethodParameters = lbDeepParamDict["methodParameters"]
    baseNode.ubClass.newConstraint = True
    baseNode.lbClass.firstRun = False



    maxNodes = bnbParamDict['maxNodes']
    maxBranchDepth = bnbParamDict['maxBranchDepth']
    maxTime = bnbParamDict['maxTime']
    minObjGap = bnbParamDict['minObjGap']
    minPercObjGap = 1+bnbParamDict['minPercObjGap']
    endTime = maxTime + time.time()
    while (nodeID < maxNodes and time.time() < endTime and len(lbValDict) > 0 and
           (globalUb[-1] > globalLb[-1] + minObjGap or globalUb[-1] > minPercObjGap * globalLb[-1])):
        # find node to split on
        branchTag = findBranch(branchMeth, lbValDict, ubValDict, globalUb, globalLb)
        newNodeDepth = branchTag[1] + 1

        # get node info
        branchNode = nodeDict.pop(branchTag)
        branchLbVal = lbValDict.pop(branchTag)
        branchUbVal = ubValDict.pop(branchTag)

        # find split ineqs
        lbConstr, ubConstr = branchNode.findInequality()

        # create new nodes
        lbNode, lbTag, ubNode, ubTag, nodeID = branchNode.split(lbConstr, ubConstr, nodeID)

        # bound new nodes
        ubValLb, ubSolLb, lbValLb = lbNode.bound()
        ubValUb, ubSolUb, lbValUb = ubNode.bound()
        if updateBoolBranch:
            print(f"Branched on node {branchTag} (lb:{branchLbVal:.3f}, ub:{branchUbVal:.3f}),"
                  f" creating the following nodes:\n"
                  f"\t {lbTag} (lb:{lbValLb:.3f}, ub:{ubValLb:.3f}), tourweight[{lbConstr[1]}]>={lbConstr[2]}\n"
                  f"\t {ubTag} (lb:{lbValUb:.3f}, ub:{ubValUb:.3f}), tourweight[{ubConstr[1]}]<={ubConstr[2]}")
        # update global upperbound
        if ubValLb < globalUb[-1]:
            globalUb = [lbTag, ubSolLb, ubValLb]
        if ubValUb < globalUb[-1]:
            globalUb = [ubTag, ubSolUb, ubValUb]

        # check if new nodes are to be added
        if lbValLb <= globalUb[-1] and newNodeDepth <= maxBranchDepth:
            ubValDict[lbTag] = ubValLb
            lbValDict[lbTag] = lbValLb
            nodeDict[lbTag] = lbNode
        if lbValUb <= globalUb[-1] and newNodeDepth <= maxBranchDepth:
            ubValDict[ubTag] = ubValUb
            lbValDict[ubTag] = lbValUb
            nodeDict[ubTag] = ubNode

        # prune nodes
        nodesToPrune = []
        for tag, value in lbValDict.items():
            if value > globalUb[-1]:
                nodesToPrune.append(tag)
        for tag in nodesToPrune:
            ubValDict.pop(tag)
            lbValDict.pop(tag)
            nodeDict.pop(tag)

        # update global lowerbound
        globalLb = list(min(lbValDict.items(), key=lambda tup: tup[1]))


    return globalUb, globalLb


if __name__ == '__main__':
    parametersType = 3
    if parametersType == 1:
        interceptFile = "CountsV2.json"
        screenlinesUsed = True
        screenlinesFile = "ScreenlinesDiscreet.json"
        tourDictFile = "clusters.json"
        tourOnODDictFile = "clusterOnODDict.json"

        readInModelParams(interceptFile, screenlinesUsed, screenlinesFile, tourDictFile, tourOnODDictFile)
        makeSparceAdjacencyMatrices()
    elif parametersType == 2:
        interceptFile = "CountsV2.json"
        screenlinesUsed = True
        screenlinesFile = "ScreenlinesDiscreetV2.json"
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
        tbw2 = tbw.copy()

        # lowerboundParameterDict = {"lbVector":upperbounder.solution, "ubVector":upperbounder.solution}
        lowerboundParameterDict = {}
        splitInequalityParameterDict = {}
        lowerbounder = lowerboundClass(lowerboundParameterDict)
        initTime = time.time()
        lowerbounder.evaluateSolution()
        print(lowerbounder.value)
        print(f"Created Class in {initTime - readTime:.3f} seconds")
        lowerbounder.bound()
        boundTime = time.time()
        print(f"Screenline base lowerbound finished in {boundTime - initTime:.3f} seconds")
        print(lowerbounder.value)
        lowerbounder.evaluateSolution()
        print(lowerbounder.value)

        # print(copySol[copySol.nonzero()])


        upperboundParameterDict = {"solution": tbw.copy()}
        readTime = time.time()
        upperbounder = upperboundClass(upperboundParameterDict)
        initTime = time.time()
        print(f"Created Class in {initTime - readTime:.3f} seconds")
        upperbounder.bound()
        boundTime = time.time()
        print(f"Taboosearch finished in {boundTime - initTime:.3f} seconds")
        print(upperbounder.value)
        x,y = upperbounder.evaluateSolution()
        print(x)
        plt.show()
        branchAndBoundParameterDict = {}
    else:
        interceptFile = "CountsV2.json"
        screenlinesUsed = True
        screenlinesFile = "ScreenlinesDiscreetV2.json"
        tourDictFile = "clusters.json"
        tourOnODDictFile = "clusterOnODDict.json"
        adjsofile = "adjSlOD.npz"
        adjtofile = "adjTourOD.npz"
        adjtsfile = "adjTourSl.npz"
        neighofile = "neighboursOD.npz"
        neighsfile = "neighboursSl.npz"
        startTime = time.time()
        readInModelParams2(interceptFile, screenlinesUsed, screenlinesFile, tourDictFile, tourOnODDictFile, adjsofile,
                           adjtofile, adjtsfile, neighofile, neighsfile)

        upperboundParameterDict = {"method":"tabooSearch",
                                   "methodParameters":{"maxDepth": 7500, "tabooLength": 1000,
                                                       "maxNoImprovement": 800, "maxTime": 600,
                                                       "printDepth": 20000, "recallDepth": 100000}}
        lowerboundParameterDict = {"method":"screenlineBasedLP"}
        upperBoundDeeperParameterDict = {"method":"tabooSearch",
                                            "methodParameters":{"maxDepth": 750, "tabooLength": 1000,
                                                                "maxNoImprovement": 800, "maxTime": 60,
                                                                "printDepth": 20000, "recallDepth": 100000}}
        lowerBoundDeeperParameterDict = {"method":"screenlineBasedLP"}
        splitInequalityParameterDict = {"ineqType":"tourBased", "lbOutputType":"csr"}
        branchAndBoundParameterDict = {"branchMethod":"globalLb", "maxNodes":1000, "maxBranchDepth":100, "maxTime":3600,
                                       "minObjGap":0, "minPercObjGap":0.05,
                                       "ubUpdates":False, "ubBasicUpdates":False,
                                       "lbUpdates":False, "lbBasicUpdates":False,
                                       "branchingUpdates":True}
        branchAndBound(upperboundParameterDict, lowerboundParameterDict, splitInequalityParameterDict,
                       branchAndBoundParameterDict, upperBoundDeeperParameterDict, lowerBoundDeeperParameterDict)


    # branchAndBound(upperboundParameterDict, lowerboundParameterDict,
    #               splitInequalityParameterDict, branchAndBoundParameterDict)
