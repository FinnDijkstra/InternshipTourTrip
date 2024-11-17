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
import gurobipy as gp
from gurobipy import GRB
from numpy.f2py.auxfuncs import throw_error
from numpy.ma.core import indices

from main import method

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
slMaxVal = np.empty(1)

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
nrOfSls = 0
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
        tourNames, screenlineNames, penalty, tComp, slMaxDiff, baseUb1, slSizes, slsOnTour,nrOfSls, slMaxVal
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
    baseSLWeight = aST.dot(tbw)
    maxSLWeight = baseSLWeight*upperbound
    slMaxVal = np.maximum(maxSLWeight, cl)
    # slMaxDiff = slMaxDiffCoo.tolist()
    nrOfSls = screenlineNames.size
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

    @classmethod
    def from_copy(cls, lbVector, ubVector, solution, ubMethod, ubMethodParameters,
                  value, newConstraint, updateBool, boundNecessary, basicUpdateBool):
        instance = cls.__new__(cls)  # Bypass __init__
        instance.lbVector = lbVector
        instance.ubVector = ubVector
        instance.solution = solution
        instance.ubMethod = ubMethod
        instance.ubMethodParameters = ubMethodParameters
        instance.value = value
        instance.newConstraint = newConstraint
        instance.updateBool = updateBool
        instance.boundNecessary = boundNecessary
        instance.basicUpdateBool = basicUpdateBool
        return instance

    def __deepcopy__(self, memo):
        # Use from_copy to create a copy without __init__ processing
        new_copy = self.from_copy(
            lbVector=self.lbVector.copy(),
            ubVector=self.ubVector.copy(),
            solution=self.solution.copy(),
            ubMethod=self.ubMethod,
            ubMethodParameters=self.ubMethodParameters,
            value=self.value,
            newConstraint=self.newConstraint,
            updateBool=self.updateBool,
            boundNecessary=self.boundNecessary,
            basicUpdateBool=self.basicUpdateBool
        )
        return new_copy

    def updateSolutions(self, Constraint):
        side, tourID, value = Constraint
        if side * self.solution[tourID] > side * value:
            self.solution[tourID] = value
            self.boundNecessary = True
        else:
            self.boundNecessary = False

    def bound(self):
        if self.ubMethod == "tabooSearch" and self.boundNecessary:
            # lp = LineProfiler()
            # lp_wrapper = lp(self.tabooSearch)
            # self.solution, self.value = lp_wrapper(startingWeights=self.solution)
            # lp.print_stats()
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
        # lp = LineProfiler()
        # lp_wrapper = lp(self.initDiffList)
        # changeDiffList = lp_wrapper(curWeights, solCounts)
        # lp.print_stats()

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
            if depth - recalMoment >= recallDepth:
                recalMoment = depth
                newDiffList = self.initDiffList(curWeights, solCounts)
                newDiffList[np.unravel_index(tabooList, changeDiffList.shape)] += 4 * penalty

                mask = (np.abs(changeDiffList - newDiffList) > 0.00002)

                if np.any(mask):
                    newDiffList2 = self.initDiffListOld(curWeights, solCounts)
                    newDiffList2[np.unravel_index(tabooList, changeDiffList.shape)] += 4 * penalty
                    mask2 = (np.abs(newDiffList - newDiffList2) > 0.00002)
                    print([changeDiffList[mask], newDiffList[mask]])
                    print("uh oh")
                changeDiffList = newDiffList
                minValue = self.evaluateSolution(minWeights)[0]
            if tempValue < minValue:
                minValue = tempValue
                minWeights = curWeights
                lastImprovement = 0
                improvementCount += 1
                # val, sol = self.evaluateSolution(curWeights)
                # if abs(val - minValue) > 0.1:
                #     print("non matching values")
                #     faults += 1

                if depth-improvementMoment >= printDepth and self.basicUpdateBool:
                    improvementMoment = depth



                    timeNow = time.time()
                    # timeList[depth] = timeNow - timeBeforeLoop
                    # valueList[depth] = minValue

                    print(f"{minValue:.4f} (best step / updating time per thousand/ total times: {sumOfMinTimes/depth*1000:.3f}s/ {
                            sumOfUpdateTimes/depth*1000:.3f}s/ {timeNow-timeBeforeLoop:.3f}s, average set size {
                            sumOfSizes/depth:.1f} (without logic {sumOfPotentialSizes/depth:.1f}), \n {
                            improvementCount*100/(depth+1):.3f}% of steps are improvements, {-improvementCount+depth+1} non improvements steps)")

            else:
                lastImprovement += 1

            depth += 1
        if self.updateBool:
            depthsOfImprovement = valueList.nonzero()
            # plt.plot(timeList[depthsOfImprovement], valueList[depthsOfImprovement])
            # plt.show()
        if self.basicUpdateBool:
            print(f"Taboo finished in {time.time()-timeNow} with depth:{depth}, last improvement:{
                        lastImprovement}/{maxNoImprovement}")
        return minWeights, minValue



    def initDiffListOld(self, curWeights, solCounts):
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
        slImprovementCSRDecr[nz] -= absDiffArray[nz[1]]                         # this gives the warning probably
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



    def initDiffList(self, curWeights, solCounts):
        # diffArray = solCounts-cl
        # absDiffArray = np.abs(diffArray)
        # curComp = curWeights-tbw
        # absCurComp = np.abs(curComp)
        # localATS = aTS.copy()
        # nz = localATS.nonzero()
        # localATS[nz] += diffArray[nz[1]]
        # slImprovementCSRIncr = localATS.multiply(localATS.sign())
        # slImprovementCSRIncr[nz] -= absDiffArray[nz[1]]
        # localATS[nz] -= 2*diffArray[nz[1]]
        # slImprovementCSRDecr = localATS.multiply(localATS.sign())
        # slImprovementCSRDecr[nz] -= absDiffArray[nz[1]]                         # this gives the warning probably
        # absIncrComp = np.abs(curComp+1)
        # IncrCompDiff = (absIncrComp - absCurComp)/baseWeightSum
        # DecrCompDiff = -IncrCompDiff
        # IncrDiffArray = slImprovementCSRIncr.sum(axis=1) + IncrCompDiff
        # decrDiffArray = slImprovementCSRDecr.sum(axis=1) + DecrCompDiff
        # diffArray = np.stack((decrDiffArray, IncrDiffArray), axis=0)
        # mask = (curWeights == self.lbVector)
        # diffArray[0, mask] += 16 * penalty
        # mask = (curWeights == self.ubVector)
        # diffArray[1, mask] += 16 * penalty
        # return diffArray

        diffArray = solCounts-cl
        absDiffArray = np.abs(diffArray)
        curComp = curWeights-tbw
        absCurComp = np.abs(curComp)


        slImprovementCSRIncr = aTS.copy()
        indATS = slImprovementCSRIncr.indices

        completeDiffArray = diffArray[indATS]
        completeAbsDiffArray = absDiffArray[indATS]

        slImprovementCSRIncr.data += completeDiffArray
        np.abs(slImprovementCSRIncr.data, out=slImprovementCSRIncr.data)
        slImprovementCSRIncr.data -= completeAbsDiffArray

        slImprovementCSRDecr = aTS.copy()

        slImprovementCSRDecr.data -= completeDiffArray
        np.abs(slImprovementCSRDecr.data, out=slImprovementCSRDecr.data)
        slImprovementCSRDecr.data -= completeAbsDiffArray


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
                                value:.3f}, compensation factor: {compValue:.3f}. "
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
            newDiff = solSL + stepInfluenceSl*nextStepSide - countSL
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
        if constr[0] == 1:
            self.ubVector[constr[1]] = constr[2]
        else:
            self.lbVector[constr[1]] = constr[2]
        self.updateSolutions(constr)





class lowerboundClass:
    # newConstraint = [+/- 1, tourID, value]. 1 for upperbound, -1 for lowerbound
    def __init__(self, lbParamDict):
        self.lbVector = lbParamDict.get("lbVector", np.zeros(nrOfClusters))
        self.ubVector = lbParamDict.get("ubVector", upperbound*tbw)
        self.solution = lbParamDict.get("solutionBase",
                                        np.minimum(self.ubVector, np.maximum(tbw,self.lbVector)))
        self.lbMethod = lbParamDict.get("method", "screenlineBasedLP")
        if self.lbMethod == "screenlineBasedLP":
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
        elif self.lbMethod == "linearRelaxation":
            self.firstRun = False
        elif self.lbMethod == "linearRelaxationScipy":
            self.firstRun = False
        self.extraParams = lbParamDict.get("extraParams", {})
        if self.lbMethod == "screenlineBasedLP" and not self.extraParams:
            self.extraParams["ValueMatrix"] = aST.power(-1).multiply(np.divide(tComp, tp)).tocsr()
        elif self.lbMethod == "linearRelaxation" and not self.extraParams:
            self.extraParams["MIPGap"] = 0.5
        elif self.lbMethod == "linearRelaxationScipy" and not self.extraParams:
            self.extraParams["MIPGap"] = 0.05
        self.extraVars = lbParamDict.get("extraVars", {})

        if self.lbMethod == "linearRelaxation" and not self.extraVars:
            self.makeModel(False)
        if self.lbMethod == "linearRelaxationScipy" and not self.extraVars:
            self.makeModelScipy()
        self.value = lbParamDict.get("value",0)
        self.newConstraint = lbParamDict.get("newConstraint",(0,0,0))
        self.markedSls = np.empty(1)
        self.markSls()
        self.updateBool = lbParamDict.get("updateBool", True)
        self.basicUpdateBool = lbParamDict.get("basicUpdateBool", True)
        if self.firstRun:
            self.prepForRun()


    @classmethod
    def from_copy(cls, lbVector, ubVector, solution, solutionFinal, solutionTranspose, firstRun, lbMethod, extraParams,
                  extraVars, value, newConstraint, updateBool, markedSls, basicUpdateBool):
        instance = cls.__new__(cls)  # Bypass __init__
        instance.lbVector = lbVector
        instance.ubVector = ubVector
        instance.solution = solution
        instance.solutionFinal = solutionFinal
        instance.solutionTranspose = solutionTranspose
        instance.firstRun = firstRun
        instance.lbMethod = lbMethod
        instance.extraParams = extraParams
        instance.extraVars = extraVars
        instance.value = value
        instance.newConstraint = newConstraint
        instance.updateBool = updateBool
        instance.markedSls = markedSls
        instance.basicUpdateBool = basicUpdateBool
        return instance

    def __deepcopy__(self, memo):
        # Use from_copy to create a copy without __init__ processing
        if self.lbMethod == "screenlineBasedLP":
            new_copy = self.from_copy(
                lbVector=self.lbVector.copy(),
                ubVector=self.ubVector.copy(),
                solution=self.solution.copy(),
                solutionFinal=self.solutionFinal.copy(),
                solutionTranspose=self.solutionTranspose.copy(),
                firstRun=self.firstRun,
                lbMethod=self.lbMethod,
                extraParams=self.extraParams,
                extraVars=self.extraVars,
                value=self.value,
                newConstraint=self.newConstraint,
                updateBool=self.updateBool,
                markedSls=self.markedSls.copy(),
                basicUpdateBool=self.basicUpdateBool
            )
        else:
            new_copy = self.from_copy(
                lbVector=self.lbVector.copy(),
                ubVector=self.ubVector.copy(),
                solution=self.solution.copy(),
                solutionFinal=[],
                solutionTranspose=[],
                firstRun=self.firstRun,
                lbMethod=self.lbMethod,
                extraParams=self.extraParams,
                extraVars=self.extraVars,
                value=self.value,
                newConstraint=self.newConstraint,
                updateBool=self.updateBool,
                markedSls=self.markedSls.copy(),
                basicUpdateBool=self.basicUpdateBool
            )
        return new_copy


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
            self.markedSls = np.arange(nrOfSls)
        else:
            side, tourID, value = self.newConstraint
            if self.lbMethod in ["screenlineBasedLP"]:
                indices, values = getRow(self.solutionFinal, tourID)
                # Depending on the added Constraint and the previous solution, only some screenlines need te be recalced
                if side == 1:
                    self.markedSls = indices[values > value]
                else:
                    self.markedSls = indices[values < value]
            else:
                self.markedSls = getRow(aTS, tourID)[0]
            # for slIdx in self.markedSls:
            #     self.extraParams["tasks"][slIdx][4][
            #         self.extraParams["tasks"][slIdx][3] == tourID] = self.solution[tourID]





    def bound(self):
        timeStartBound = time.time()
        if self.lbMethod == "screenlineBasedLP":
            # lp = LineProfiler()
            # lp_wrapper = lp(self.screenlineBasedLPBound)
            # lp_wrapper()
            # lp.print_stats()
            self.screenlineBasedLPBound()
            # print(self.markedSls)
        elif self.lbMethod == "linearRelaxation":
            self.linearRelaxBound()
        elif self.lbMethod == "linearRelaxationScipy":
            self.linearRelaxBoundScipy()
        else:
            print(f"The lowerbound method '{self.lbMethod}' is not supported")
        if self.basicUpdateBool:
            print(f"Lowerbound evaluated solution with value {self.value} in {time.time() - timeStartBound:.2f} seconds")


    def linearRelaxBoundScipy(self):
        AMaster = self.extraVars["matrix"]
        cVec = self.extraVars["objective"]
        bVec = self.extraVars["constraint"]
        lrBaseLbVec = self.extraVars["lb"]
        lrBaseUbVec = self.extraVars["ub"]
        lrLbVec = np.concatenate([self.lbVector, lrBaseLbVec])
        lrUbVec = np.concatenate([self.ubVector, lrBaseUbVec])
        bounds = np.column_stack([lrLbVec, lrUbVec])
        options = {"presolve":True, "mip_rel_gap":self.extraParams["MIPGap"], "disp":True}
        result = scipy.optimize.linprog(cVec,A_ub=AMaster, b_ub=bVec, bounds=bounds, method="highs", options=options)
        self.value = result.mip_dual_bound
        self.solution = result.x[:nrOfClusters]
        print("solved")


    def linearRelaxBound(self):
        clusterWeights = self.extraVars["clusterWeights"]
        lbnz = self.lbVector.nonzero()[0]
        ubnz = (self.ubVector-baseUb1).nonzero()[0]
        for idx in lbnz:
            clusterWeights[idx].lb = self.lbVector[idx]
        for idx in ubnz:
            clusterWeights[idx].ub = self.ubVector[idx]
        for idx in range(nrOfClusters):
            clusterWeights[idx].Start = self.solution[idx]

        m = self.extraVars["model"]
        # m.reset()
        m.setParam(GRB.Param.MIPGap, self.extraParams["MIPGap"])

        m.optimize()
        self.value = m.ObjBound
        self.solution = np.array([clusterWeights[clusterIdx].X
                                  for clusterIdx in range(nrOfClusters)])
        for idx in lbnz:
            clusterWeights[idx].lb = 0
        for idx in ubnz:
            clusterWeights[idx].ub = upperbound*tbw[idx]


    def evaluateSolution(self):
        solCountMatrix = aST.copy()
        solCountMatrix.data *= self.solutionTranspose.data

        # solCountMatrix = aST.multiply(self.solutionTranspose)
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
        dmInd = devMatrix.indices
        devMatrix.data -= tbw[dmInd]
        np.abs(devMatrix.data, out=devMatrix.data)
        tCompData = tComp[dmInd]
        np.multiply(devMatrix.data, tCompData, out=devMatrix.data)
        # absdevMatrixComp = absdevMatrix.multiply(tComp)
        compValue = devMatrix.sum()

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
                aDK = task[3]
                ubVecLocal = self.ubVector[aDK]
                lbVecLocal = self.lbVector[aDK]
                curSol = self.solution[aDK]
                startTimeSl = time.time()
                results.append([slIdx, task[2]] + list(lp_wrapper(task, curSol, ubVecLocal, lbVecLocal)))
                endTimeSl = time.time()
                sumOfTimes += (endTimeSl - startTimeSl)
                if (task[-1] + 1) % 100 == 0 and self.updateBool:
                    print(f"average  time per screenline is {sumOfTimes / (task[-1] + 1)}")
            lp.print_stats()
        else:
            for slIdx in self.markedSls:
                task = tasks[slIdx]
                startTimeSl = time.time()
                aDK = task[3]
                ubVecLocal = self.ubVector[aDK]
                lbVecLocal = self.lbVector[aDK]
                curSol = self.solution[aDK]
                results.append([slIdx, task[2]] + list(self.optimizeTrip(task, curSol, ubVecLocal, lbVecLocal)))
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

            self.solutionTranspose.data[startIdx:endIdx] = slSol
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
        # lp = LineProfiler()
        # lp_wrapper = lp(self.evaluateSolution)
        # lp_wrapper()
        # lp.print_stats()
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

        influence = getRow(aST,slIdx)[1][ascendingDensities]
        tbwLocal = tbw[ascendingDensitiesKeys]
        tCompLocal = tComp[ascendingDensitiesKeys]

        # ascendingDensitiesKeys = [columnList[idx] for idx in ascendingDensities]
        # curSol = [self.solution[tourIdx] for tourIdx in ascendingDensitiesKeys]
        # influence = [tourRow[0, tourIdx] for tourIdx in ascendingDensitiesKeys]
        # tbwLocal = [tbw[tourIdx] for tourIdx in ascendingDensitiesKeys]
        # tCompLocal = [tComp[tourIdx] for tourIdx in ascendingDensitiesKeys]
        # ubVecLocal = [self.ubVector[tourIdx]for tourIdx in ascendingDensitiesKeys]
        # lbVecLocal = [self.lbVector[tourIdx] for tourIdx in ascendingDensitiesKeys]
        # endList = time.time()
        # print(f"({(startSort-startList):.3f})/ ({(endSort-startSort):.3f})/ ({(endList-endSort):.3f}) total({(endList-startList):.3f}) list size {n}")
        return [count, n, ascendingDensities, ascendingDensitiesKeys, influence, tbwLocal, tCompLocal,
                 slIdx]

    @staticmethod
    def optimizeTrip(paramsInput, curSol, ubVecLocal, lbVecLocal):
        Linear = True
        LinearSkip = True
        [count, n, ascendingDensities, ascendingDensitiesKeys, influence, tbwLocal, tCompLocal,
          slIdx] = paramsInput
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

    def makeModel(self, integerBool):
        m = gp.Model(f"Upperbound: {upperbound}")
        m.setParam("OutputFlag", 0)
        # m.setParam(GRB.Param.Presolve, 0)
        m.setParam(GRB.Param.DisplayInterval, 10)
        m.setParam(GRB.Param.Method, 3)
        m.setParam("ConcurrentMethod", 3)
        # m.setParam(GRB.Param.BarIterLimit, 0)
        if integerBool:
            tourWeight = m.addVars(nrOfClusters, vtype=GRB.INTEGER, ub=self.ubVector.tolist(), lb=self.lbVector.tolist(), name="clusterWeights")
        else:
            tourWeight = m.addVars(nrOfClusters, vtype=GRB.SEMICONT, ub=self.ubVector.tolist(),
                                   lb=self.lbVector.tolist(), name="clusterWeights")
        totalError = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=upperbound * baseWeightSum, name="Deviation")
        # largeErrors = m.addVars(toursDF.index.tolist(), vtype=GRB.SEMICONT, ub=upperbound-1, lb=0.0, name="LargeErrors")
        tourDeviation = m.addVars(nrOfClusters,
                                  vtype=GRB.CONTINUOUS, lb=(-tbw).tolist(), ub=((upperbound - 1.0)*tbw).tolist(), name="clusterDeviation")
        absoluteErrors = m.addVars(nrOfSls, vtype=GRB.CONTINUOUS, lb=0.0, name="AbsoluteErrors")
        # absoluteErrors = m.addVars(zones,zones,  vtype=GRB.CONTINUOUS, lb=0.0, name="AbsoluteErrors" )



        for slIdx in range(nrOfSls):
            slindices, slvalues = getRow(aST, slIdx)
            rowSize = len(slindices)
            m.addConstr((gp.quicksum((tourWeight[slindices[clusterIdx]] * slvalues[clusterIdx])
                                        for clusterIdx in range(rowSize)) + absoluteErrors[slIdx]
                        >= cl[slIdx]), name=f"positive errors Screenline {slIdx}")
            m.addConstr((gp.quicksum((tourWeight[slindices[clusterIdx]] * slvalues[clusterIdx])
                                        for clusterIdx in range(rowSize)) - absoluteErrors[slIdx]
                        <= cl[slIdx]), name=f"negative errors Screenline {slIdx}")

        m.addConstrs(((tourWeight[clusterIdx] - 1 == tourDeviation[clusterIdx]) for clusterIdx in range(nrOfClusters)),
                     name="Deviation of cluster")
        # m.addConstrs(((tourWeight[ODpair] - 1 <= largeErrors[ODpair]) for ODpair in toursDF.index.tolist()),
        #              name="Large errors in tour")
        m.addConstr(totalError == gp.quicksum(tourDeviation[idx]*tComp[idx] for idx in range(nrOfClusters)), name=f"total error")



        objective = (gp.quicksum(absoluteErrors[slIdx] for slIdx in range(nrOfSls))
                     +  (
                         totalError))  # + gp.quicksum(largeErrors[ODpair] for ODpair in toursDF.index.tolist())
        m.setObjective(objective, GRB.MINIMIZE)
        m.update()
        self.extraVars["model"] = m
        self.extraVars["clusterWeights"] = tourWeight
        self.extraVars["totalError"] = totalError
        self.extraVars["clusterDeviation"] = tourDeviation
        self.extraVars["absoluteErrors"] = absoluteErrors

    def makeModelScipy(self):
        # x = (tw, tdev, slAbs)
        # c = (0 , tComp, 1)
        # l = (self.lb, 0 , 0)
        # u = (self.ub, baseUB1, none)
        # Aub_1 = (-aTS, 0, -I) bub_1 = -cl
        # Aub_2 = (aTS, 0, -I) bub_2 = cl
        # Aub_3 = (I, -I, 0) bub_3 = tbw
        # Aub_4 = (-I, I, 0) bub_4 = tbw
        zeroClusterVec = np.zeros(nrOfClusters)
        zeroSLVec = np.zeros(nrOfSls)
        slNegIdent = scipy.sparse.eye_array(nrOfSls, nrOfSls, format='csr')
        slNegIdent.data *= -1
        clusterIdent = scipy.sparse.eye_array(nrOfClusters, nrOfClusters, format='csr')
        clusterNegIdent = scipy.sparse.eye_array(nrOfClusters, nrOfClusters, format='csr')
        clusterNegIdent.data *= -1
        zeroClusterMatrix = scipy.sparse.csr_array((nrOfSls, nrOfClusters), dtype=np.float64)
        zeroSLMatrix = scipy.sparse.csr_array((nrOfClusters, nrOfSls), dtype=np.float64)
        negATS = aST.copy()
        negATS.data *= -1
        A1 = scipy.sparse.hstack([negATS, zeroClusterMatrix, slNegIdent])
        A2 = scipy.sparse.hstack([aST, zeroClusterMatrix, slNegIdent])
        A3 = scipy.sparse.hstack([clusterIdent, clusterNegIdent, zeroSLMatrix])
        A4 = scipy.sparse.hstack([clusterNegIdent, clusterIdent, zeroSLMatrix])
        AMaster = scipy.sparse.vstack([A1, A2, A3, A4])
        cVec = np.concatenate([zeroClusterVec, tComp, np.ones(nrOfSls)])
        bVec = np.concatenate([-cl, cl, tbw, tbw])
        lbVec = np.concatenate([zeroClusterVec, zeroSLVec])
        ubVec = np.concatenate([baseUb1, slMaxVal])
        self.extraVars["matrix"] = AMaster
        self.extraVars["objective"] = cVec
        self.extraVars["constraint"] = bVec
        self.extraVars["lb"] = lbVec
        self.extraVars["ub"] = ubVec

    def newNodePrep(self, constr, MIPGap):
        self.newConstraint = constr
        if constr[0] == 1:
            self.ubVector[constr[1]] = constr[2]
        else:
            self.lbVector[constr[1]] = constr[2]
        self.solution = np.minimum(self.ubVector,np.maximum(self.solution, self.lbVector))
        self.markSls()
        self.extraVars["MIPGap"] = MIPGap






class nodeClass:
    def __init__(self, ubParamDict, lbParamDict, splitParamDict, nodeTag):
        self.ineqType = splitParamDict['ineqType']
        self.lbOutputType = splitParamDict['lbOutputType']
        self.ubClass = upperboundClass(ubParamDict)
        self.lbClass = lowerboundClass(lbParamDict)
        self.tag = nodeTag

    @classmethod
    def from_copy(cls, ineqType, lbOutputType, ubClass, lbClass, tag):
        instance = cls.__new__(cls)  # Bypass __init__
        instance.ineqType = ineqType
        instance.lbOutputType = lbOutputType
        instance.ubClass = ubClass
        instance.lbClass = lbClass
        instance.tag = tag
        return instance

    def __deepcopy__(self, memo):
        new_copy = self.from_copy(
            self.ineqType,
            self.lbOutputType,
            copy.deepcopy(self.ubClass, memo),
            copy.deepcopy(self.lbClass, memo),
            self.tag

        )
        return new_copy

    def bound(self):
        self.ubClass.bound()
        self.lbClass.bound()
        return self.ubClass.value, self.ubClass.solution, self.lbClass.value


    def findInequality(self):
        if self.ineqType == "tourBased":
            if self.lbOutputType == "csr":
                avgTourWeight2 = self.lbClass.solutionFinal.sum(axis=1).flatten()
                avgTourWeight = np.divide(avgTourWeight2, slsOnTour)
                copySol = self.lbClass.solutionFinal.copy()
                nz = copySol.nonzero()
                copySolCheck1 = copySol.copy()
                copySol[nz] -= avgTourWeight[nz[0]]
                copySolCheck2 = copySol.copy()
                copySolCheck3 = copySol.multiply(1 / upperbound)
                copySolCheck4 = copySolCheck3.copy().power(2)
                copySolCheck5 = copySolCheck4.copy().sum(axis=1)
                copySolCheck6 = np.divide(copySolCheck5.copy().flatten(), slsOnTour)
                stdDevs = np.sqrt(copySolCheck6)
                splitChoiceVec = np.abs(avgTourWeight - np.round(avgTourWeight)) + stdDevs
            elif self.lbOutputType == "array":
                avgTourWeight = self.lbClass.solution.copy()
                splitChoiceVec = np.abs(avgTourWeight - np.round(avgTourWeight))
            else:
                raise Exception("Not a valid lb output type")
            ineqIndex = np.argmax(splitChoiceVec)
            if avgTourWeight[ineqIndex]%1<0.002:
                print("found int sol")
            ubVal = np.floor(avgTourWeight[ineqIndex])
            lbVal = ubVal + 1
            lbConstr = (-1, ineqIndex, lbVal)
            ubConstr = (1, ineqIndex, ubVal)
            return lbConstr, ubConstr
        else:
            raise Exception("Not a valid inequality type")

    def split(self, lbConstr, ubConstr, nodeID, MIPGap):
        nextDepth = self.tag[1] + 1
        lbTag = (nodeID, nextDepth)
        ubTag = (nodeID+1, nextDepth)
        lbNode = copy.deepcopy(self, {})
        lbNode.newNodePrep(lbConstr, lbTag, MIPGap)
        ubNode = copy.deepcopy(self, {})
        ubNode.newNodePrep(ubConstr, ubTag, MIPGap)


        return lbNode, lbTag, ubNode, ubTag, nodeID+2



    def newNodePrep(self, constr, tag, MIPGap):
        self.tag = tag
        self.ubClass.newNodePrep(constr)
        self.lbClass.newNodePrep(constr, MIPGap)
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
    maxMIPGap = bnbParamDict["maxMIPGap"]
    debugDict = {nodeTag:[baseNode,[]]}
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
        branchGap = (branchUbVal - branchLbVal)/branchUbVal
        # find split ineqs
        lbConstr, ubConstr = branchNode.findInequality()

        # create new nodes
        newMIPGap = min(maxMIPGap, branchGap)
        lbNode, lbTag, ubNode, ubTag, nodeID = branchNode.split(lbConstr, ubConstr, nodeID, newMIPGap)
        lbConstrList = debugDict[branchNode.tag][1].copy()
        lbConstrList.append(lbConstr)
        debugDict[lbTag] = [lbNode, lbConstrList]
        ubConstrList = debugDict[branchNode.tag][1].copy()
        ubConstrList.append(ubConstr)
        debugDict[ubTag] = [ubNode, ubConstrList]

        # bound new nodes
        ubValLb, ubSolLb, lbValLb = lbNode.bound()
        ubValUb, ubSolUb, lbValUb = ubNode.bound()
        if updateBoolBranch:
            print(f"Branched on node {branchTag} (lb:{branchLbVal:.3f}, ub:{branchUbVal:.3f}),"
                  f" creating the following nodes:\n"
                  f"\t {lbTag} \t(lb:{lbValLb:.3f}, ub:{ubValLb:.3f}), \ttourweight[{lbConstr[1]}]>={lbConstr[2]}\n"
                  f"\t {ubTag} \t(lb:{lbValUb:.3f}, ub:{ubValUb:.3f}), \ttourweight[{ubConstr[1]}]<={ubConstr[2]}")
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


def clusterTester():
    relatedDictionary = []
    notInAGroup = [True]*nrOfClusters
    np.arange(nrOfClusters)
    sumOfGroupSizes = 0
    sumOfNotInAGroup = 0
    timeStart = time.time()
    for i in range(nrOfClusters):
        if notInAGroup[i]:
            iVec = np.zeros(screenlineNames.size)
            iSLSize = slsOnTour[i]
            iInd, iVal = getRow(aTS,i)
            iVec[iInd] = iVal
            iIsInGroup = False
            iGroup = []
            for j in range(i+1,nrOfClusters):
                if notInAGroup[j]:
                    jSLSize = slsOnTour[j]
                    if jSLSize == iSLSize:
                        jVec = np.zeros(screenlineNames.size)
                        jInd, jVal = getRow(aTS, j)
                        jVec[jInd] = jVal
                        if np.all(iVec == jVec):
                            iIsInGroup = True
                            iGroup.append(j)
                            notInAGroup[j] = False
            if iIsInGroup:
                iGroup.insert(0,i)
                relatedDictionary.append(iGroup)
                sumOfGroupSizes += len(iGroup)
                print(f"found group {iGroup},\n\t{sumOfGroupSizes} are in a group so far "
                      f"({len(relatedDictionary)} groups), "
                      f"{nrOfClusters-sumOfGroupSizes-sumOfNotInAGroup} left, "
                      f"{sumOfNotInAGroup} found that are not in a group,\n"
                      f"\t{(sumOfGroupSizes+sumOfNotInAGroup)/nrOfClusters*100:.3f}% "
                      f"Expected end time: {time.strftime("%a, %d %b %Y %H:%M:%S", 
                                                          time.localtime((time.time()-timeStart) *
                                                                         (nrOfClusters /
                                                                          (sumOfGroupSizes+sumOfNotInAGroup)) 
                                                                         + timeStart))}")
                notInAGroup[i] = False
            else:
                sumOfNotInAGroup += 1
    sumOfSavings = 0
    largestGroup = 0
    for group in relatedDictionary:
        sumOfSavings += len(group)-1
        if len(group) > largestGroup:
            largestGroup = len(group)
    print(relatedDictionary)
    print(sumOfSavings)
    print(largestGroup)
    write_list(relatedDictionary)
    x=1

def write_list(a_list):
    print("Started writing list data into a json file")
    with open("names.json", "w") as fp:
        json.dump(a_list, fp)
        print("Done writing JSON data into .json file")


def superClustering(superclustersfile, superclustersOutfile, adjcsfile):
    f = open(superclustersfile, "r")
    superclusters = json.load(f)
    f.close()
    processed = [True] * nrOfClusters
    timeStart = time.time()
    superClusterIdx = 0
    superClusterDict = {}
    namesIdx = 0
    rowVec = np.empty(1)
    indicesVec = np.empty(1)
    valuesVec = np.empty(1)
    lastPrint = 0
    for i in range(nrOfClusters):
        if processed[i]:
            indices, values = getRow(aTS, i)
            if superclusters[namesIdx][0] == i:
                tourList = superclusters[namesIdx]
                namesIdx += 1
                namesIdx = min(namesIdx, len(superclusters)-1)
                sclbw = 0
                for cluster in tourList:
                    processed[cluster] = False
                    sclbw += tbw[cluster]
            else:
                processed[i] = False
                sclbw = tbw[i]
                tourList = [i]
            superClusterDict[superClusterIdx] = [sclbw, tourList,1.0]
            if superClusterIdx == 0:
                indicesVec = indices
                valuesVec = values
                rowVec = np.full(indices.size, superClusterIdx)
                firstAdd = False
            else:
                indicesVec = np.append(indicesVec, indices)
                valuesVec = np.append(valuesVec, values)
                rowVec = np.append(rowVec, np.full(indices.size, superClusterIdx))
            superClusterIdx += 1
            if i - lastPrint >= 10000:
                lastPrint = i
                print(f"processed {namesIdx}/{len(superclusters)} aka {i}/{nrOfClusters}")
    superclusterData2 = {f"{key}": value for key, value in superClusterDict.items()}
    superclusterJson = json.dumps(superclusterData2, indent=4)
    with open(superclustersOutfile, "w") as outfile:
        outfile.write(superclusterJson)
    superclusterSparceMatrix = scipy.sparse.csr_array((valuesVec, (rowVec, indicesVec)))
    scipy.sparse.save_npz(adjcsfile, superclusterSparceMatrix)
    return


if __name__ == '__main__':
    parametersType = -3
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
    elif parametersType == -1:
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
        clusterTester()
    elif parametersType == -2:
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
        superclustersfile = "names.json"
        superclustersOutfile = "superclusters.json"
        adjcsfile = "adjClSl.npz"

        startTime = time.time()
        readInModelParams2(interceptFile, screenlinesUsed, screenlinesFile, tourDictFile, tourOnODDictFile, adjsofile,
                           adjtofile, adjtsfile, neighofile, neighsfile)
        superClustering(superclustersfile,superclustersOutfile,adjcsfile)
    elif parametersType == -3:
        interceptFile = "CountsV2.json"
        screenlinesUsed = True
        screenlinesFile = "ScreenlinesDiscreetV2.json"
        tourDictFile = "superclusters.json"
        tourOnODDictFile = "clusterOnODDict.json"
        adjsofile = "adjSlOD.npz"
        adjtofile = "adjTourOD.npz"
        adjtsfile = "adjClSl.npz"
        neighofile = "neighboursOD.npz"
        neighsfile = "neighboursSl.npz"
        startTime = time.time()
        readInModelParams2(interceptFile, screenlinesUsed, screenlinesFile, tourDictFile, tourOnODDictFile, adjsofile,
                           adjtofile, adjtsfile, neighofile, neighsfile)
        # clusterTester()
        upperboundParameterDict = {"method":"tabooSearch",
                                   "methodParameters":{"maxDepth": 750000, "tabooLength": 10000,
                                                       "maxNoImprovement": 8000, "maxTime": 600,
                                                       "printDepth": 200000, "recallDepth": 100000}}
        lowerboundParameterDict = {"method":"linearRelaxation"}
        upperBoundDeeperParameterDict = {"method":"tabooSearch",
                                            "methodParameters":{"maxDepth": 75000, "tabooLength": 10000,
                                                                "maxNoImprovement": 8000, "maxTime": 60,
                                                                "printDepth": 20000, "recallDepth": 100000}}
        lowerBoundDeeperParameterDict = {"method":"linearRelaxation"}
        splitInequalityParameterDict = {"ineqType":"tourBased", "lbOutputType":"array"}
        branchAndBoundParameterDict = {"branchMethod":"breadthFirst",
                                       "maxNodes":100000, "maxBranchDepth":1000, "maxTime":3600,
                                       "minObjGap":0, "minPercObjGap":0.05, "maxMIPGap":0.0005,
                                       "ubUpdates":False, "ubBasicUpdates":True,
                                       "lbUpdates":False, "lbBasicUpdates":False,
                                       "branchingUpdates":True}
        branchAndBound(upperboundParameterDict, lowerboundParameterDict, splitInequalityParameterDict,
                       branchAndBoundParameterDict, upperBoundDeeperParameterDict, lowerBoundDeeperParameterDict)
    elif parametersType == -4:
        interceptFile = "CountsV2.json"
        screenlinesUsed = True
        screenlinesFile = "ScreenlinesDiscreetV2.json"
        tourDictFile = "superclusters.json"
        tourOnODDictFile = "clusterOnODDict.json"
        adjsofile = "adjSlOD.npz"
        adjtofile = "adjTourOD.npz"
        adjtsfile = "adjClSl.npz"
        neighofile = "neighboursOD.npz"
        neighsfile = "neighboursSl.npz"
        startTime = time.time()
        readInModelParams2(interceptFile, screenlinesUsed, screenlinesFile, tourDictFile, tourOnODDictFile, adjsofile,
                           adjtofile, adjtsfile, neighofile, neighsfile)


        lb = lowerboundClass({"method":"linearRelaxation"})
        lb.makeModel(True)
        lb.bound()
        print(np.where(lb.solution - np.round(lb.solution)>0.0002))
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
        # clusterTester()
        upperboundParameterDict = {"method":"tabooSearch",
                                   "methodParameters":{"maxDepth": 7500, "tabooLength": 1000,
                                                       "maxNoImprovement": 800, "maxTime": 600,
                                                       "printDepth": 20000, "recallDepth": 100000}}
        lowerboundParameterDict = {"method":"screenlineBasedLP"}
        upperBoundDeeperParameterDict = {"method":"tabooSearch",
                                            "methodParameters":{"maxDepth": 75, "tabooLength": 1000,
                                                                "maxNoImprovement": 800, "maxTime": 60,
                                                                "printDepth": 20000, "recallDepth": 100000}}
        lowerBoundDeeperParameterDict = {"method":"screenlineBasedLP"}
        splitInequalityParameterDict = {"ineqType":"tourBased", "lbOutputType":"csr"}
        branchAndBoundParameterDict = {"branchMethod":"globalLb", "maxNodes":1000, "maxBranchDepth":100, "maxTime":3600,
                                       "minObjGap":0, "minPercObjGap":0.05, "maxMIPGap":0.005,
                                       "ubUpdates":False, "ubBasicUpdates":False,
                                       "lbUpdates":False, "lbBasicUpdates":False,
                                       "branchingUpdates":True}
        branchAndBound(upperboundParameterDict, lowerboundParameterDict, splitInequalityParameterDict,
                       branchAndBoundParameterDict, upperBoundDeeperParameterDict, lowerBoundDeeperParameterDict)


    # branchAndBound(upperboundParameterDict, lowerboundParameterDict,
    #               splitInequalityParameterDict, branchAndBoundParameterDict)
