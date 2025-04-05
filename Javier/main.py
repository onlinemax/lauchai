import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

stock_data = pd.read_csv('', index_col=0)

# Load macroeconomic data
gdp_data = pd.read_csv('GDP.csv')
inflation_data = pd.read_csv('Price_index.csv')
interest_data = pd.read_csv('FEDFUNDS.csv')
unemployment_data = pd.read_csv('UNRATE.csv')


# Merge stock data with macroeconomic data based on date
merged_data = pd.merge(stock_data, gdp_data, on='observation_date', how='left')
merged_data = pd.merge(merged_data, inflation_data, on='observation_date', how='left')
merged_data = pd.merge(merged_data, interest_data, on='observation_date', how='left')
merged_data = pd.merge(merged_data, unemployment_data, on='observation_date', how='left')

# Select features and target
features = ['Open', 'High', 'Low', 'Volume', 'GDP growth', 'Price index', 'FEDFUNDS', 'UNRATE']
target = 'Close'

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    merged_data[features], merged_data[target], test_size=0.2, random_state=42
)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}") 