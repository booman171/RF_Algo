#open and read csv files
#chunk data into 2 minute segments
#find avg power, avg time difference, and amnt of hits per chunk
### WILL BREAK if column titles exist in CSV file. Erase first line!

import csv # csv.reader
import os
import pandas as pd 

def chunk(fileName, chunkNum, timeLen = 120000):
	returnList = []
	with open(fileName, 'r') as csvfile:
		dataReader = list(csv.reader(csvfile, delimiter = ","))
		start = int(dataReader[0][3])
		for row in dataReader:
			timestamp = int(row[3])
			if (timestamp >= (start + (timeLen * chunkNum))) and (timestamp <= (start + (timeLen * (chunkNum + 1)))):
				returnList.append([int(row[0].replace(",", "")), float(row[1]), row[2], timestamp])
		csvfile.close()
	return returnList
def avgPwr(listName):
    return sum([subList[1] for subList in listName]) / len(listName)
def avgDiff(listName):
	try:
		return (listName[-1][3] - listName[0][3]) / (len(listName)- 1)
	except:
		return 0
def getStats(fileName):
	chunkStats = [None]
	statList = []
	i = 0
	while len(chunkStats) != 0:
		chunkStats = chunk(fileName, i)
		if len(chunkStats) > 0:
			statList.append([chunkStats[0][0], len(chunkStats), avgPwr(chunkStats), avgDiff(chunkStats), chunkStats[0][2], chunkStats[0][3] ])
		i += 1
	return statList
def writeStats(infileName, outfileName):
	statList = getStats(infileName)
	df = pd.DataFrame(statList)
	print (df)
	with open(outfileName, 'a') as f:
		df.to_csv(f, header=None,index=False)
Path = "C:/Users/Jenario/Documents/RF_Algo"
filelist = os.listdir(Path)
for x in filelist:
	if x.endswith(".csv"):
		writeStats(x, 'all.txt')