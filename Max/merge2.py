import os
import pandas as pd
import dateutil.parser
input_dir = ["stocks2", "stocks3"]
master_df = pd.DataFrame([])
for file in os.listdir(input_dir[1]):
    if not file.endswith('.csv'):
        continue
    company = file[:-4]
    other_df = pd.read_csv(input_dir[0] + '/' + file)
    df = pd.read_csv(input_dir[1] + '/' + file)
    other_df['Date'] = other_df['Date'].apply(lambda x: x[2:-1])
    other_df['Date'] = pd.to_datetime(other_df['Date'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'Date'});
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    other_df.drop(other_df.columns[other_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    merge_df = pd.merge(df, other_df, 'inner', "Date")
    merge_df['Company'] = company
    master_df = pd.concat([master_df, merge_df])

master_df.to_csv("data.csv")


