import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('/workspaces/lauchai/Albi/Price_index.csv')
df['observation_date'] = pd.to_datetime(df['observation_date'])




df = df[df['observation_date'] >= "2013-01-01"]
df = df[df['observation_date']<="2024-12-01"]

df['INFLATION'] = df['MEDCPIM158SFRBCLE']
df = df.drop(columns=['MEDCPIM158SFRBCLE'])
print(df.head())

print(df.tail())
print(df.columns)

plt.plot(df['observation_date'],df['INFLATION'])
plt.xlabel("Date")
plt.ylabel("Inflation Rate")
plt.title("US Inflation Rate Over Time")

# Rotation et espacement des dates
plt.xticks(rotation=45)



plt.tight_layout()  # Ajuster les marges pour éviter que les labels ne soient coupés
plt.show()

print(len(df))


df['observation_date'] = pd.to_datetime(df['observation_date'])

df.set_index('observation_date', inplace=True)

# Resample data into quarterly averages
df_quarterly = df.resample('QE-DEC').mean()

# Reset index for better visibility
df = df_quarterly.reset_index()

plt.plot(df['observation_date'],df['INFLATION'])
plt.title("US Quartely inflation rate")
plt.show()

df.reset_index(drop=True, inplace=True)
df_INFLATION = df.copy()
