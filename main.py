
import numpy as np
import os.path
import pandas as pd
import ast
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
toursDF = pd.DataFrame()
interceptDF = pd.DataFrame()
maxZoneNR = 0
zones = []
m = gp.Model()
tourWeight = gp.tupledict([])
ODtupledict = gp.tupledict([])
absoluteErrors =gp.tupledict([])
processNewData = False




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def initializeModel(upperbound=5, threshold=0.01):
    global m
    global tourWeight
    global absoluteErrors
    m = gp.Model(f"Threshold: {threshold}, Upperbound: {upperbound}")
    tourWeight = m.addVars(toursDF.index.tolist(), vtype=GRB.SEMICONT, lb=0.0, ub=upperbound)
    absoluteErrors = m.addVars(zones, zones,  vtype=GRB.CONTINUOUS, lb=0.0)
    return


def addConstraints(threshold = 0.01):
    return


def readData(interceptName="", processedTours=""):
    global toursDF
    global interceptDF
    global maxZoneNR
    global zones
    tourColumnTypes = {"agentID":"Int64", "Woonzone":"Int64", "Vervoerswijze":"S", "Volgorde":"S",
                       "Hoofdbestemming_auto":"Int64", "Nevenbestemming_auto":"Int64", "AantalTrips":"Int64",
                       "Prob_auto":"f"}
    toursDF = pd.read_csv(processedTours, dtype=tourColumnTypes)
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


    usedAgentsDF = agentDF.copy().loc[:, headerDFList]
    firstDestinationDF = usedAgentsDF[["Hoofdbestemming_auto","Nevenbestemming_auto"]]
    secondDestinationDF = usedAgentsDF["TweedeHoofdbestemming_auto"]


    maxzonenr = max(agentDF["Woonzone"])

    # ValidBoolseries is a 2xn df with true if the destination is internal,
    # a tour has a relevant trip if any destination in a tour is internal
    validBoolSeries = firstDestinationDF.le(maxzonenr)
    toursDF = usedAgentsDF.loc[validBoolSeries.any(axis="columns"),headerList].copy()

    # Creates a new dataframe of with the second trips all having 1 destination
    secondValidBoolSeries = secondDestinationDF.le(maxzonenr)
    secondaryToursDF = (usedAgentsDF.loc[secondValidBoolSeries, secondaryHeaderList]
                        .assign(Nevenbestemming_auto=np.nan, AantalTrips=2))
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
    global ODtupledict
    boolHNTrip = toursDF.Volgorde.eq("hoofdmotief_eerst")
    boolNHTrip = toursDF.Volgorde.eq("nevenmotief_eerst")
    bool2Trip = toursDF.AantalTrips.eq(2)
    ODdefaultDict = defaultdict(list)

    # Adds the trips of the tours with 1 destination
    for index, info in toursDF.loc[bool2Trip].iterrows():
        ODdefaultDict[(info["Woonzone"], info["Hoofdbestemming_auto"])].append(info["agentID"])
        ODdefaultDict[(info["Hoofdbestemming_auto"], info["Woonzone"])].append(info["agentID"])

    # Adds the internal trips of the tours with 2 destinations, main location first
    for index, info in toursDF.loc[boolHNTrip].iterrows():
        if info["Hoofdbestemming_auto"] <= maxZoneNR:
            ODdefaultDict[(info["Woonzone"], info["Hoofdbestemming_auto"])].append(info["agentID"])
            if info["Nevenbestemming_auto"] <= maxZoneNR:
                ODdefaultDict[(info["Hoofdbestemming_auto"], info["Nevenbestemming_auto"])].append(info["agentID"])
                ODdefaultDict[(info["Nevenbestemming_auto"], info["Woonzone"])].append(info["agentID"])
        elif info["Nevenbestemming_auto"] <= maxZoneNR:
            ODdefaultDict[(info["Nevenbestemming_auto"], info["Woonzone"])].append(info["agentID"])

    # Adds the internal trips of the tours with 2 destinations, secondary location first
    for index, info in toursDF.loc[boolNHTrip].iterrows():
        if info["Nevenbestemming_auto"] <= maxZoneNR:
            ODdefaultDict[(info["Woonzone"], info["Nevenbestemming_auto"])].append(info["agentID"])
            if info["Hoofdbestemming_auto"] <= maxZoneNR:
                ODdefaultDict[(info["Nevenbestemming_auto"], info["Hoofdbestemming_auto"])].append(info["agentID"])
                ODdefaultDict[(info["Hoofdbestemming_auto"], info["Woonzone"])].append(info["agentID"])
        elif info["Hoofdbestemming_auto"] <= maxZoneNR:
            ODdefaultDict[(info["Hoofdbestemming_auto"], info["Woonzone"])].append(info["agentID"])

    ODtupledict = gp.tupledict(ODdefaultDict)
    return



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if processNewData or not os.path.isfile("processedTours.csv"):
        tourColumnTypes = {"agentID": "Int64", "Woonzone": "Int64", "Vervoerswijze": "str", "Volgorde": "str",
                           "Hoofdbestemming_auto": "Int64", "Nevenbestemming_auto": "Int64", "AantalTrips": "Int64"
                           , "VervoerswijzeTour2":"str", "TweedeHoofdbestemming_auto":"Int64"}
        processData(pd.read_csv("populationAfterOctavius.csv", dtype=tourColumnTypes))
        print("Created ProcessedTours.csv!")
    else:
        print("ProcessedTours.csv found!")
    readData("NormObservedMatrix.txt", "processedTours.csv")
    toursToODDict()
    initializeModel(5, 0.01)
    print("Read input.")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
