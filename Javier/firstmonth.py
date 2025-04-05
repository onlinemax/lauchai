import os
import pandas as pd
from constants import quarter_ranges
import numpy as np
directory = '/home/max/Documents/Programs/Python/data/stocks'
new_directory = '/home/max/Documents/Programs/Python/data/stocks2'
filelist = os.listdir(directory)

def process_file(file: str):    
    print(file)
    df = pd.read_csv(directory + '/' + file)
    df = df.iloc[2:]
    df['Date'] = pd.to_datetime(df['Price'])
    for a in ['High', 'Close',  'Low', 'Open', 'Volume']:
        df[a] = pd.to_numeric(df[a])
    df = df.drop('Price', axis=1)
    currentYear = df['Date'].min().to_datetime64().astype('datetime64[Y]').astype(int) + 1970
    lastYear = df['Date'].max().to_datetime64().astype('datetime64[Y]').astype(int) + 1970

    # print(df['Date'].min(), df['Date'].max())
    index = df.columns.values
    otherData = {}
    for i in index:
        otherData[i] = np.array([])

    for year in range(currentYear, lastYear + 1):
        for quarter in quarter_ranges:
            start_date = str(year) + '-' + quarter[0]
            end_date = str(year) + '-' + quarter[1]
            means = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].mean(numeric_only=True)
            if means.isnull().values.all(): continue
            for i in means.index:
                otherData[i] = np.append(otherData[i], means[i])
            otherData['Date'] = np.append(otherData['Date'], start_date)
    otherData['Volume'] = np.round(otherData['Volume']).astype(np.int64)
    otherData = pd.DataFrame(data=otherData)
    otherData.to_csv(new_directory + '/' + file)
for file in filelist:
    if file.endswith('.csv'):
        process_file(file)

