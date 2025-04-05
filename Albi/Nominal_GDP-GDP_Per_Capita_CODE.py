#https://datacommons.org/explore?hl=fr#q=gdp

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Nominal_GDP_Capita.csv')

df = df[df['observation_date'] >= "2013-01-01"]
df = df[df['observation_date']<="2024-12-01"]

print(df.head())
print(df.columns)
print(df.tail())



df['GDP_PER_CAPITA_NOMINAL'] = df['A939RC0Q052SBEA'].pct_change()*100
df =df.drop(columns=['A939RC0Q052SBEA'])
print(df.head())
print(df.columns)
print(df.tail())


df['GDP_PER_CAPITA_NOMINAL'] = ((1+df['GDP_PER_CAPITA_NOMINAL']/100)**4 -1)*100


print(df.head())
print(df.columns)
print(df.tail())





plt.plot(df['observation_date'],df['GDP_PER_CAPITA_NOMINAL'])
plt.xlabel("Date")
plt.ylabel("Quartely GDP per capita growth")
plt.title("GDP per capita growth")

plt.show()


# Rotation et espacement des dates
plt.xticks(rotation=45)

print(len(df))

df.reset_index(drop=True, inplace=True)
df_GDP_PER_CAPITA_NOMINAL = df.copy()

df = pd.read_csv('GDP.csv')

df = df[df['observation_date'] >= "2013-01-01"]
df = df[df['observation_date']<="2024-12-01"]

df['GDP_NOMINAL'] = df['GDP'].pct_change()*100
df =df.drop(columns=['GDP'])

df['GDP growth'] = ((1+df['GDP_NOMINAL']/100)**4 -1)*100



print(len(df))
print(df.head())
print(df.columns)
print(df.tail())

df['observation_date'] = pd.to_datetime(df['observation_date'])

# Set 'Date' as the index
df.set_index('observation_date', inplace=True)

# Resample data into quarterly averages
df_quarterly = df.resample('Q').mean()

# Reset index for better visibility
df = df_quarterly.reset_index()

plt.plot(df['observation_date'],df['GDP_NOMINAL'])
plt.xlabel("Date")
plt.ylabel("Quartely GDP growth")
plt.title("GDP  growth")

# Rotation et espacement des dates
plt.xticks(rotation=45)

plt.show()
df.reset_index(drop=True, inplace=True)
df_GDP_NOMINAL = df.copy()
