#INTEREST

#GDP https://fred.stlouisfed.org/series/FEDFUNDS

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('FEDFUNDS.csv')


df = df[df['observation_date'] >= "2013-01-01"]
df = df[df['observation_date']<="2024-12-01"]




print(df.head())
print(df.columns)
print(df.tail())


plt.plot(df['observation_date'],df['FEDFUNDS'])
plt.xlabel("Date")
plt.ylabel("Interest")
plt.title("US Interest Rate Over Time")

# Rotation et espacement des dates
plt.xticks(rotation=45)



plt.tight_layout()  # Ajuster les marges pour éviter que les labels ne soient coupés
plt.show()

df['observation_date'] = pd.to_datetime(df['observation_date'])

# Set 'Date' as the index
df.set_index('observation_date', inplace=True)

# Resample data into quarterly averages
df_quarterly = df.resample('Q').mean()

# Reset index for better visibility
df = df_quarterly.reset_index()

plt.plot(df['observation_date'],df['FEDFUNDS'])
plt.xlabel("Date")
plt.ylabel("Quartely Interest")
plt.title("US Interest Rate Over Time")

# Rotation et espacement des dates
plt.xticks(rotation=45)

print(len(df))


df['INTEREST'] = df['FEDFUNDS']
df = df.drop(columns=['FEDFUNDS'])

df.reset_index(drop=True, inplace=True)
df_INTEREST = df.copy()
