#Real GDP
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('GDP_percapita.csv')

df = df[df['observation_date'] >= "2013-01-01"]
df = df[df['observation_date']<="2024-12-01"]

print(df.head())
print(df.columns)
print(df.tail())



df['GDP_PER_CAPITA_REAL'] = df['A939RX0Q048SBEA'].pct_change()*100
df =df.drop(columns=['A939RX0Q048SBEA'])
print(df.head())
print(df.columns)
print(df.tail())


df['GDP_PER_CAPITA_REAL'] = ((1+df['GDP_PER_CAPITA_REAL']/100)**4 -1)*100


print(df.head())
print(df.columns)
print(df.tail())





plt.plot(df['observation_date'],df['GDP_PER_CAPITA_REAL'])
plt.xlabel("Date")
plt.ylabel("Quartely GDP per capita growth")
plt.title("GDP per capita growth")

plt.show()


# Rotation et espacement des dates
plt.xticks(rotation=45)

print(len(df))

df.reset_index(drop=True, inplace=True)
df_GDP_PER_CAPITA_REAL = df.copy()



df = pd.read_csv('GDPC1.csv')


df = df[df['observation_date'] >= "2013-01-01"]
df = df[df['observation_date']<="2024-12-01"]

df['GDP_REAL'] = df['GDPC1'].pct_change()*100
df =df.drop(columns=['GDPC1'])

df['GDP_REAL'] = ((1+df['GDP_REAL']/100)**4 -1)*100


plt.plot(df['observation_date'],df['GDP_REAL'])
plt.xlabel("Date")
plt.ylabel("Quartely GDP growth")
plt.title("GDP  growth")

# Rotation et espacement des dates
plt.xticks(rotation=45)

plt.show()
df.reset_index(drop=True, inplace=True)
df_GDP_REAL = df.copy()
