
import numpy as np
import os.path
import pandas as pd
import ast
import gurobipy as gp
from gurobipy import GRB
toursDF = pd.DataFrame()
interceptDF = pd.DataFrame()
m =  gp.Model()
processNewData = False




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def initializeModel(upperbound=5, threshold=0.01, agents=pd.DataFrame()):
    global m
    m = gp.Model(f"Threshold: {threshold}, Upperbound: {upperbound}")


def readData(interceptName="", processedTours=""):
    global toursDF
    global interceptDF
    interceptDF = pd.read_csv(interceptName, sep=";", header=None)
    tourColumnTypes = {"agentID":"i", "Woonzone":"i", "Vervoerswijze":"S", "Volgorde":"S",
                       "Hoofdbestemming_auto":"f", "Nevenbestemming_auto":"f", "AantalTrips":"i", "Prob_auto":"f"}
    toursDF = pd.read_csv(processedTours, dtype=tourColumnTypes)



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
    maxagentnr = max(agentDF["agentID"])
    # Marks outgoing and empty trips
    outgoingBoolDF = firstDestinationDF.gt(maxzonenr)
    nanBoolDF = firstDestinationDF.isna()
    validBoolSeries = firstDestinationDF.le(maxzonenr)
    toursDF = usedAgentsDF.loc[validBoolSeries.any(axis="columns"),headerList].copy()
    # Marks all agents with internal second tour
    secondValidBoolSeries = secondDestinationDF.le(maxzonenr)
    secondaryToursDF = (usedAgentsDF.loc[secondValidBoolSeries, secondaryHeaderList]
                        .assign(Nevenbestemming_auto=np.nan, AantalTrips=2.0))
    secondaryToursDF.rename(columns={"TweedeHoofdbestemming_auto":"Hoofdbestemming_auto",
                                     "VervoerswijzeTour2":"Vervoerswijze"}, inplace=True)
    finalToursDF = pd.concat([toursDF, secondaryToursDF],ignore_index=True)
    # test2 = finalToursDF.Vervoerswijze.str.replace("=>",":")
    # test3 = test2.apply(lambda x: ast.literal_eval(x))
    # test = finalToursDF.Vervoerswijze.to_dict()
    test4 = finalToursDF.Vervoerswijze.str.partition('"auto"=>')
    test5 = test4[2].str.partition(',')
    # autoProb = {}
    # for index, vervoersString in test.items():
    #     vervoeren = ast.literal_eval(vervoersString.replace("=>",":"))
    #     autoProb[index] = vervoeren.get("auto")
    finalToursDF2 = finalToursDF.assign(Prob_auto = test5[0])
    finalToursDF2.to_csv("processedTours.csv",index=False)
    agentsUsed = finalToursDF["agentID"].unique()
    x = 1

def toursToODDict():
    return



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if processNewData or not os.path.isfile("processedTours.csv"):
        processData(pd.read_csv("populationAfterOctavius.csv"))
        print("Created ProcessedTours.csv!")
    else:
        print("ProcessedTours.csv found!")
    readData("NormObservedMatrix.txt", "processedTours.csv")

    print("Read input.")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
