import os
import pandas as pd
from constants import quarter_ranges
import numpy as np
from array import Array
# This program averages the stock value into quarters

directory = './stocks'
new_directory = './stocks2'
filelist = os.listdir(directory)
def change_to_numeric(df):
    for a in ['High', 'Close',  'Low', 'Open', 'Volume']:
        df[a] = pd.to_numeric(df[a])

def add_quarter(year):
    array = [] 
    for quarter in quarter_ranges:
        array.append(str(year) + '-' + quarter[0])
        array.append(str(year) + '-' + quarter[1])
    return array
def special_add():
    length = 0
    array 
def process_file(file: str):    
    print(file)
    df = pd.read_csv(directory + '/' + file)
    df = df.iloc[2:]
    df['Date'] = pd.to_datetime(df['Price'])
    change_to_numeric(df)
    df = df.drop('Price', axis=1)

    currentYear = df['Date'].min().to_datetime64().astype('datetime64[Y]').astype(int) + 1970
    lastYear = df['Date'].max().to_datetime64().astype('datetime64[Y]').astype(int) + 1970

    quarters = np.array(list(map(add_quarter, range(currentYear, lastYear + 1)))).flatten()
    
    index = df.columns.values
    otherData = {}
    for i in index:
        if i == 'Date':
            otherData[i] = Array('S10')
        else:
            otherData[i] = Array(np.float64) 
    for i in range(0, len(quarters), 2) :
        start_date = str(quarters[i])
        end_date = str(quarters[i + 1])
        means = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].mean(numeric_only=True)
        if means.isnull().values.all(): continue
        for i in means.index:
            otherData[i].append(means[i])
        otherData['Date'].append(start_date)

    for i in index:
        otherData[i] = otherData[i].build()
    otherData['Volume'] = np.round(otherData['Volume']).astype(np.int64)
    otherData = pd.DataFrame(data=otherData)
    otherData.to_csv(new_directory + '/' + file)
count = 0
for file in filelist:
    if file.endswith('.csv'):
        print('Processed: ', count,'/', len(filelist))
        process_file(file)
        count+=1

