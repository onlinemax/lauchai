import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv('UNRATE.csv')





df = df[df['observation_date'] >= "2013-01-01"]
df = df[df['observation_date']<="2024-12-01"]


df['UNEMPLOYMENT'] = df['UNRATE']
df = df.drop(columns=['UNRATE'])
print(df.head())

print(df.tail())
print(df.columns)

plt.plot(df['observation_date'],df['UNEMPLOYMENT'])
plt.xlabel("Date")
plt.ylabel("Unemployment Rate")
plt.title("US Unemployment Rate Over Time")

# Rotation et espacement des dates
plt.xticks(rotation=45)



plt.tight_layout()  # Ajuster les marges pour éviter que les labels ne soient coupés
plt.show()

print(len(df))

df['observation_date'] = pd.to_datetime(df['observation_date'])

df.set_index('observation_date', inplace=True)

# Resample data into quarterly averages
df_quarterly = df.resample('Q').mean()

# Reset index for better visibility
df = df_quarterly.reset_index()

plt.plot(df['observation_date'],df['UNEMPLOYMENT'])
plt.title("US Quartely unemployment")
plt.show()

print(len(df))

df.reset_index(drop=True, inplace=True)
df_UNEMPLOYMENT = df.copy()
