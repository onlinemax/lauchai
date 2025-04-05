import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

# Let's pick a company to predict - I'll use Apple (AAPL) as an example
COMPANY = 'AAPL'
DATA_DIR = '/home/max/Documents/Programs/Python/data/stocks2/'

def load_and_preprocess_data(company):
    """Load and preprocess the quarterly stock data"""
    file_path = os.path.join(DATA_DIR, f"{company}.csv")
    df = pd.read_csv(file_path)
    
    # Convert date to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Create target - next quarter's closing price
    df['target'] = df['Close'].shift(-1)
    
    # Drop the last row which will have NaN target
    df = df[:-1]
    
    return df

def create_features(df):
    """Create features for the model"""
    # Basic price features
    df['price_change'] = df['Close'].pct_change()
    df['rolling_avg_3'] = df['Close'].rolling(window=3).mean()
    df['rolling_std_3'] = df['Close'].rolling(window=3).std()
    
    # Volume features
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_rolling_avg'] = df['Volume'].rolling(window=3).mean()
    
    # Technical indicators (simplified)
    df['high_low_spread'] = (df['High'] - df['Low']) / df['Close']
    df['close_open_spread'] = (df['Close'] - df['Open']) / df['Open']
    
    # Drop rows with NaN values from rolling calculations
    df = df.dropna()
    
    return df

def train_model(X, y):
    """Train and evaluate a Random Forest model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Initialize and train model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=5,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Train MAE: {train_mae:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    
    # Plot feature importance
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    feature_importance.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Important Features')
    plt.show()
    
    return model, X_test, y_test, test_preds

def plot_predictions(y_test, predictions):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time Period')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def main():
    # Load and preprocess data
    df = load_and_preprocess_data(COMPANY)
    df = create_features(df)
    
    # Prepare features and target
    feature_cols = ['Close', 'Volume', 'price_change', 'rolling_avg_3', 
                   'rolling_std_3', 'volume_change', 'volume_rolling_avg',
                   'high_low_spread', 'close_open_spread']
    X = df[feature_cols]
    y = df['target']
    
    # Train model
    model, X_test, y_test, test_preds = train_model(X, y)
    
    # Plot predictions
    plot_predictions(y_test, test_preds)
    
    return model

if __name__ == "__main__":
    model = main()