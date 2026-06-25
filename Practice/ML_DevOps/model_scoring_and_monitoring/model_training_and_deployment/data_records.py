import pandas as pd
from datetime import datetime

sourcelocation = "./record/"
filename = 'recordkeeping.csv'
output_location = 'records.txt'

data = pd.read_csv(sourcelocation + filename)

dateTimeObj = datetime.now()
thetimenow = str(dateTimeObj.year) + '/' + str(dateTimeObj.month) + '/' + str(dateTimeObj.day)

allrecords = [sourcelocation, filename, len(data.index), thetimenow]

myFile = open(output_location, 'w')
for element in allrecords:
    myFile.write(str(element))