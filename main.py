import math

import numpy as np
import os.path
import pandas as pd
import ast
import math
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import strconv
import matplotlib.pyplot as plt


upperbound = 5
toursDF = pd.DataFrame()
interceptDF = pd.DataFrame()
maxZoneNR = 0
nrOfAgents = 0
dimX = 0
correctionValue = 0
zones = []
m = gp.Model()
tourWeight = gp.tupledict([])
tourDeviation = gp.tupledict([])
# largeErrors = gp.tupledict([])
totalError = []
ODtupledict = gp.tupledict([])
tourODsDict = {}
absoluteErrors = gp.tupledict([])
processNewData = False
writeODDictFile = False
writeBaseModel = False
method = "bnb"



class LowerBounder:
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




class UpperBounders:
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




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def initializeModel(threshold=0.01):
    global m
    global tourWeight
    global absoluteErrors
    global tourDeviation
    global largeErrors
    global totalError
    m = gp.Model(f"Threshold: {threshold}, Upperbound: {upperbound}")
    tourWeight = m.addVars(toursDF.index.tolist(), vtype=GRB.SEMICONT, ub=upperbound, lb=0.0, name="TourWeight")
    totalError = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub = upperbound*dimX, name="Deviation")
    # largeErrors = m.addVars(toursDF.index.tolist(), vtype=GRB.SEMICONT, ub=upperbound-1, lb=0.0, name="LargeErrors")
    tourDeviation = m.addVars(toursDF.index.tolist(),
                              vtype=GRB.SEMICONT, lb=-1.0, ub=upperbound-1.0, name="TourWeight")
    absoluteErrors = m.addVars(list(ODtupledict.keys()),  vtype=GRB.CONTINUOUS, lb=0.0, name="AbsoluteErrors")
    # absoluteErrors = m.addVars(zones,zones,  vtype=GRB.CONTINUOUS, lb=0.0, name="AbsoluteErrors" )
    m.update()

    return


def addConstraints(threshold=0.01):
    global correctionValue
    setOfPairs = set()
    # for ODpair, agents in ODtupledict.items():
    #     if ODpair[0]<=maxZoneNR and ODpair[1]<=maxZoneNR:
    #         interceptValue = interceptDF.at[ODpair[1],ODpair[0]]
    #         thresholdValueCheck = interceptDF.at[ODpair]
    #         if thresholdValueCheck > threshold:
    #             m.addConstr((gp.quicksum((tourWeight[tourID]*toursDF.at[tourID, "Prob_auto"]) for tourID in agents)
    #                             <= absoluteErrors[ODpair] + thresholdValueCheck) , name=f"positive errors ODpair {ODpair}")
    #             m.addConstr((gp.quicksum((tourWeight[tourID] * toursDF.at[tourID, "Prob_auto"]) for tourID in agents)
    #                           >= -absoluteErrors[ODpair] + thresholdValueCheck), name=f"negative errors ODpair {ODpair}")
    #         else:
    #             setOfPairs.add(ODpair)
    #             correctionValue += thresholdValueCheck
    for i in range(1,maxZoneNR+1):
        for j in range(1,maxZoneNR+1):
            ODpair = tuple([i,j])
            interceptValue = interceptDF.at[ODpair[1], ODpair[0]]
            thresholdValueCheck = interceptDF.at[ODpair]
            agents = ODtupledict.get(ODpair,[])

            if thresholdValueCheck > threshold and agents:
                m.addConstr((gp.quicksum((tourWeight[tourID]*toursDF.at[tourID, "Prob_auto"]) for tourID in agents)
                                <= absoluteErrors[ODpair] + thresholdValueCheck) , name=f"positive errors ODpair {ODpair}")
                m.addConstr((gp.quicksum((tourWeight[tourID] * toursDF.at[tourID, "Prob_auto"]) for tourID in agents)
                              >= -absoluteErrors[ODpair] + thresholdValueCheck), name=f"negative errors ODpair {ODpair}")
            else:
                setOfPairs.add(ODpair)
                correctionValue += thresholdValueCheck
                for tourID in agents:
                    m.addConstr(tourWeight[tourID]==0)
    print(correctionValue)
    m.addConstrs(((tourWeight[ODpair]-1 == tourDeviation[ODpair]) for ODpair in toursDF.index.tolist()),
                 name="Deviation of tour")
    # m.addConstrs(((tourWeight[ODpair] - 1 <= largeErrors[ODpair]) for ODpair in toursDF.index.tolist()),
    #              name="Large errors in tour")
    m.addConstr(totalError == gp.norm(tourDeviation,1), name=f"total error")

    m.update()
    return setOfPairs


def addObjective():
    objective = (gp.quicksum(absoluteErrors[ODpair] for ODpair in ODtupledict.keys())
                 + (1.0/dimX) * (totalError)) # + gp.quicksum(largeErrors[ODpair] for ODpair in toursDF.index.tolist())
    m.setObjective(objective, GRB.MINIMIZE)
    m.update()


def readData(interceptName="", processedTours=""):
    global toursDF
    global interceptDF
    global maxZoneNR
    global zones
    tourColumnTypes = {"agentID":"Int64", "Woonzone":"Int64", "Vervoerswijze":"S", "Volgorde":"S",
                       "Hoofdbestemming_auto":"Int64", "Nevenbestemming_auto":"Int64", "AantalTrips":"Int64",
                       "Prob_auto":"f"}
    toursDF = pd.read_csv(processedTours, dtype=tourColumnTypes)
    dimX = toursDF.index.size
    maxZoneNR = toursDF.Woonzone.max()
    zones = list(range(1, maxZoneNR + 1))
    interceptDF = pd.read_csv(interceptName, sep=";", header=None, names=zones)
    interceptDF.index = zones





def processData(agentDF=pd.DataFrame()):
    headerList = ["agentID", "Woonzone", "Vervoerswijze", "Volgorde", "Hoofdbestemming_auto", "Nevenbestemming_auto",
                  "AantalTrips"]
    secondaryHeaderList = ["agentID", "Woonzone", "VervoerswijzeTour2", "TweedeHoofdbestemming_auto"]
    headerDFList = ["agentID", "Woonzone", "Vervoerswijze", "Volgorde", "Hoofdbestemming_auto", "Nevenbestemming_auto",
                    "AantalTrips", "VervoerswijzeTour2", "TweedeHoofdbestemming_auto"]
    overlookedHeaderList = ["agentID", "Woonzone", "Vervoerswijze", "Hoofdbestemming_auto"]

    usedAgentsDF = agentDF.copy().loc[:, headerDFList]
    firstDestinationDF = usedAgentsDF[["Hoofdbestemming_auto","Nevenbestemming_auto"]]
    secondDestinationDF = usedAgentsDF["TweedeHoofdbestemming_auto"]


    maxzonenr = max(agentDF["Woonzone"])

    # ValidBoolseries is a 2xn df with true if the destination is internal,
    # a tour has a relevant trip if any destination in a tour is internal
    validBoolSeries = firstDestinationDF.le(maxzonenr)
    toursDF = usedAgentsDF.loc[validBoolSeries.any(axis="columns"),headerList].copy()
    # mainOutgoing = firstDestinationDF["Hoofdbestemming_auto"].gt(maxzonenr, fill_value=0)
    # noSecond = firstDestinationDF["Nevenbestemming_auto"].isna()
    # overlookedDF = usedAgentsDF.loc[mainOutgoing & noSecond,overlookedHeaderList].copy().assign(Nevenbestemming_auto=np.nan, AantalTrips=2, Volgorde=np.nan)

    # Creates a new dataframe of with the second trips all having 1 destination
    secondValidBoolSeries = secondDestinationDF.le(maxzonenr)
    secondaryToursDF = (usedAgentsDF.loc[secondValidBoolSeries, secondaryHeaderList]
                        .assign(Nevenbestemming_auto=np.nan, AantalTrips=2, Volgorde=np.nan))
    secondaryToursDF.rename(columns={"TweedeHoofdbestemming_auto":"Hoofdbestemming_auto",
                                     "VervoerswijzeTour2":"Vervoerswijze"}, inplace=True)

    # Add the two dataframes of relevent tours together
    finalToursDF = pd.concat([toursDF, secondaryToursDF],ignore_index=True)

    # The least computationally expensive way to extract the probabilities I could think of
    test4 = finalToursDF.Vervoerswijze.str.partition('"auto"=>')
    test5 = test4[2].str.partition(',')
    finalToursDF2 = finalToursDF.assign(Prob_auto = test5[0])
    finalToursDF2.to_csv("processedTours.csv",index=False)




def toursToODDict():
    global ODtupledict, tourODsDict
    boolHNTrip = toursDF.Volgorde.eq("hoofdmotief_eerst")
    boolNHTrip = toursDF.Volgorde.eq("nevenmotief_eerst")
    bool2Trip = toursDF.AantalTrips.eq(2)
    ODdefaultDict = defaultdict(list)
    nhDF = toursDF.loc[boolNHTrip]
    # Adds the trips of the tours with 1 destination
    for index, info in toursDF.loc[bool2Trip].iterrows():
        ODdefaultDict[(info["Woonzone"], info["Hoofdbestemming_auto"])].append(index)
        ODdefaultDict[(info["Hoofdbestemming_auto"], info["Woonzone"])].append(index)
        tourODsDict[index] = [(info["Woonzone"], info["Hoofdbestemming_auto"]),
                              (info["Hoofdbestemming_auto"], info["Woonzone"])]

    # Adds the internal trips of the tours with 2 destinations, main location first
    for index, info in toursDF.loc[boolHNTrip].iterrows():
        if info["Hoofdbestemming_auto"] <= maxZoneNR:
            ODdefaultDict[(info["Woonzone"], info["Hoofdbestemming_auto"])].append(index)
            tourODsDict[index] = [(info["Woonzone"], info["Hoofdbestemming_auto"])]
            if info["Nevenbestemming_auto"] <= maxZoneNR:
                ODdefaultDict[(info["Hoofdbestemming_auto"], info["Nevenbestemming_auto"])].append(index)
                tourODsDict[index].append((info["Hoofdbestemming_auto"], info["Nevenbestemming_auto"]))
                ODdefaultDict[(info["Nevenbestemming_auto"], info["Woonzone"])].append(index)
                tourODsDict[index].append((info["Nevenbestemming_auto"], info["Woonzone"]))
        elif info["Nevenbestemming_auto"] <= maxZoneNR:
            ODdefaultDict[(info["Nevenbestemming_auto"], info["Woonzone"])].append(index)
            tourODsDict[index] = [(info["Nevenbestemming_auto"], info["Woonzone"])]

    # Adds the internal trips of the tours with 2 destinations, secondary location first
    x=0
    for index, info in toursDF.loc[boolNHTrip].iterrows():
        x = info["Nevenbestemming_auto"]
        if info["Nevenbestemming_auto"] <= maxZoneNR:
            ODdefaultDict[(info["Woonzone"], info["Nevenbestemming_auto"])].append(index)
            tourODsDict[index] = [(info["Woonzone"], info["Nevenbestemming_auto"])]
            if info["Hoofdbestemming_auto"] <= maxZoneNR:
                ODdefaultDict[(info["Nevenbestemming_auto"], info["Hoofdbestemming_auto"])].append(index)
                tourODsDict[index].append((info["Nevenbestemming_auto"], info["Hoofdbestemming_auto"]))
                ODdefaultDict[(info["Hoofdbestemming_auto"], info["Woonzone"])].append(index)
                tourODsDict[index].append((info["Hoofdbestemming_auto"], info["Woonzone"]))
        elif info["Hoofdbestemming_auto"] <= maxZoneNR:
            ODdefaultDict[(info["Hoofdbestemming_auto"], info["Woonzone"])].append(index)
            tourODsDict[index] = [(info["Hoofdbestemming_auto"], info["Woonzone"])]

    ODtupledict = gp.tupledict(ODdefaultDict)
    # for key, value in ODdefaultDict.items():
    #     ODtupledict[key] = value
    return


def writeTupleDict(tupleDict, location):
    with open(location, "w") as f:
        for key, value in tupleDict.items():
            keyString = str(key[0])
            for i in range(1,len(key)):
                keyString += "," + str(key[i])
            valueString = str(value[0])
            for i in range(1,len(value)):
                valueString += "," + str(value[i])
            entryString = keyString + ":" + valueString +"\n"
            f.write(entryString)


def readTupleDict(location):
    newTD = gp.tupledict([])
    with open(location, "r") as f:
        for entry in f.readlines():
            key, value = entry.split(":")
            tupleKey = tuple([int(k) for k in key.split(",")])
            listValue = [int(v) for v in value.split(",")]
            newTD[tupleKey] = listValue
    return newTD


def tourwiseODLister():
    global tourODsDict
    for OD, tourList in ODtupledict.items():
        for tour in tourList:
            tourODsDict[tour] = tourODsDict.get(tour, []) + [OD]



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if processNewData or not os.path.isfile("processedTours.csv"):
        tourColumnTypes = {"agentID": "Int64", "Woonzone": "Int64", "Vervoerswijze": "str", "Volgorde": "str",
                           "Hoofdbestemming_auto": "Int64", "Nevenbestemming_auto": "Int64", "AantalTrips": "Int64"
                           , "VervoerswijzeTour2":"str", "TweedeHoofdbestemming_auto":"Int64"}
        processData(pd.read_csv("populationAfterOctavius.csv", dtype=tourColumnTypes))
        print("Created ProcessedTours.csv!")
        writeODDictFile = True
    else:
        print("ProcessedTours.csv found!")
    readData("NormObservedMatrix.txt", "processedTours.csv")
    print("Read input.")
    if writeODDictFile or not os.path.isfile("ODtupledict.txt"):
        toursToODDict()
        writeTupleDict(ODtupledict, "ODtupledict.txt")
        print("Created ODtupledict.txt!")
    else:
        ODtupledict = readTupleDict("ODtupledict.txt")
        print("Read ODtupledict.txt!")
        tourwiseODLister()
    print("Model intialization")
    if method == "gurobi":
        initializeModel(0.01)
        print("Adding Constraints")
        setOfPairs = addConstraints(0.0)
        addObjective()
        m.setParam("MIPGap", 0.0025)
        m.setParam("MIPFocus", 2)
        m.optimize()
        objective = 0
        for i in range(1, maxZoneNR+1):
            for j in range(1, maxZoneNR+1):
                if tuple([i,j]) in setOfPairs:
                    agents = ODtupledict.get(tuple([i,j]), [])
                    objective += abs(sum((tourWeight[tourID].x*toursDF.at[tourID, "Prob_auto"])
                                         for tourID in agents) - interceptDF[i][j])
                else: # if tuple([i,j]) in absoluteErrors.keys():
                    objective += absoluteErrors[i,j].x
        print(objective)
        # print(len(setOfPairs))
        print((1.0/dimX) * totalError.x)
        tourWeightDict = {tourID: tour.x for tourID, tour in tourWeight.items()}
        tourWeightSeries = pd.Series(tourWeightDict)
        tourWeightSeries.to_csv("weightSeries.csv")
        tourWeightResults = [tour.x for tour in tourWeight.values()]
        tourWeightResults.sort()
        print(tourWeightResults[-1])
        bucketList = [0,1,2,3,4,5,6]
        plt.scatter(range(len(tourWeightResults)), tourWeightResults, s=1)
        plt.savefig('plot' + str(0.1) + 'ub' + str(5) + '.png', dpi=500)
        plt.show()

        plt.hist(tourWeightResults, bins=bucketList)
        plt.savefig('histplot' + str(0.1) + 'ub' + str(5) + '.png', dpi=500)
        plt.show()
    elif method == "bnb":
        x=1




# Currently finding an objective of 193211.88903641407,
# while Sanne found 174398.99584534226. Possible explainations that come to mind:
# She throws out all tours that have a secondary location of 696. (Update, no large improvement 193355.46796705993)
# She reads the observation matrix with higher precision, potentially allowing closer results
# no difference 193211.88903641407
# Other factors I fail to notice
#
# if I just sum over the entries where the agents walk over it becomes 79337.01191589674
# 156042.3574979772 is best with 0.01
# 158097.85623693644 is best with 0.1
# 155353.43769193138 is best with 0
# Investigate if it is necessary to set tours that use roads below the threshold need to be set to 0.
