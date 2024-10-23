import gzip
import ast
import json
import pandas


jsonfilename = "DiscreetClusters.json.gz"


def readToData(jsonfilename):
    with gzip.open(jsonfilename, 'r') as file:
        json_bytes = file.read()

    json_str = json_bytes.decode('utf-8')
    return json.loads(json_str)


def clustersToList(data):
    tours = {}
    tourId = 1
    for listString, value in data.items():
        lsv2 = listString[1:-1].split('][')
        tourList = []
        for ODno in range(len(lsv2)):
            ODstr = lsv2[ODno]
            if ODno != 0:
                ODstr = "[" + ODstr
            if ODno != len(lsv2)-1:
                ODstr += "]"
            tourList.append(tuple(ast.literal_eval(ODstr)))
        tours[tourId] = [value,tourList]
        tourId += 1

    print(len(tours))
    print(sum(tour[0] for tour in tours.values()))
    return tours


def findNeighbours(tours):
    tourOnODDict = {}
    for tourID, tourInfo in tours.items():
        tripsList = tourInfo[1]
        for OD in tripsList:
            tourOnODDict.setdefault(OD,[]).append(tourID)


    maxNeighbours = 0
    totalNeighbours = 0
    maxID = 0
    for tourID,tourInfo in tours.items():
        tripsList = tourInfo[1]
        neighbours = set()
        for OD in tripsList:
            neighbours.update(set(tourOnODDict[OD]))
        tourInfo.append(list(neighbours))
        noNeighbours = len(neighbours)
        totalNeighbours += noNeighbours
        if noNeighbours > maxNeighbours:
            maxNeighbours = noNeighbours
            maxID = tourID
        tourInfo.append(noNeighbours)
        tours[tourID] = tourInfo

    print(tours[maxID])
    print(maxNeighbours)
    print(totalNeighbours/len(tours))
    listOfKeys = []
    for key,value in tourOnODDict.items():
        if len(value) > 1:
            listOfKeys.append(key)
    print(listOfKeys)
    return tours, tourOnODDict, listOfKeys


if __name__ == '__main__':
    data = readToData(jsonfilename)
    clusters = clustersToList(data)
    clusters, clusterOnODDict, listOfKeys = findNeighbours(clusters)
    clustersJson = json.dumps(clusters, indent=4)
    clusterOnODDictStrKeys = {f"{OD}": value for OD, value in clusterOnODDict.items()}
    clusterOnODDictJSson = json.dumps(clusterOnODDictStrKeys, indent=4)

    # Writing to sample.json
    with open("clusters.json", "w") as outfile:
        outfile.write(clustersJson)
    with open("clusterOnODDict.json", "w") as outfile:
        outfile.write(clusterOnODDictJSson)

