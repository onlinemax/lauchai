import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

DATA_DIR = '/home/max/Documents/Programs/Python/data/stocks2/'

class StockPredictor:
    def __init__(self, company='AAPL'):
        self.company = company
        self.model = None
        self.df = None
        self.feature_cols = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the quarterly stock data"""
        file_path = os.path.join(DATA_DIR, f"{self.company}.csv")
        self.df = pd.read_csv(file_path)
        
        # Convert date to datetime and sort
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        return self.df
    
    def create_features(self, forecast_horizon=8):
        """
        Create features for the model
        forecast_horizon: Number of quarters to predict ahead (8 = 2 years)
        """
        # Create target - price forecast_horizon quarters in the future
        self.df['target'] = self.df['Close'].shift(-forecast_horizon)
        
        # Basic price features
        for window in [1, 2, 4, 8]:  # Different lookback periods
            self.df[f'pct_change_{window}q'] = self.df['Close'].pct_change(window)
            self.df[f'rolling_avg_{window}q'] = self.df['Close'].rolling(window=window).mean()
            self.df[f'rolling_std_{window}q'] = self.df['Close'].rolling(window=window).std()
        
        # Volume features
        self.df['volume_pct_change'] = self.df['Volume'].pct_change()
        for window in [2, 4]:
            self.df[f'volume_rolling_avg_{window}q'] = self.df['Volume'].rolling(window=window).mean()
        
        # Technical indicators
        self.df['high_low_spread'] = (self.df['High'] - self.df['Low']) / self.df['Close']
        self.df['close_open_spread'] = (self.df['Close'] - self.df['Open']) / self.df['Open']
        self.df['volatility'] = self.df['High'] / self.df['Low'] - 1
        
        # Drop rows with NaN values (from target shift and rolling calculations)
        self.df = self.df.dropna()
        
        return self.df
    
    def train_model(self, test_size=0.2):
        """Train and evaluate the Random Forest model"""
        # Define feature columns
        self.feature_cols = [col for col in self.df.columns 
                            if col not in ['Date', 'target', 'Unnamed: 0']]
        X = self.df[self.feature_cols]
        y = self.df['target']
        
        # Split data chronologically
        split_idx = int(len(self.df) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Initialize and train model
        self.model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=8,
            min_samples_split=5,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Make predictions
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        train_mape = np.mean(np.abs((y_train - train_preds) / y_train)) * 100
        test_mape = np.mean(np.abs((y_test - test_preds) / y_test)) * 100
        
        print(f"\nModel Evaluation for {self.company}:")
        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Train MAPE: {train_mape:.2f}%")
        print(f"Test MAPE: {test_mape:.2f}%")
        
        # Plot feature importance
        feature_importance = pd.Series(self.model.feature_importances_, index=X.columns)
        feature_importance.nlargest(10).plot(kind='barh')
        plt.title('Top 10 Important Features')
        plt.show()
        
        return test_preds, y_test
    
    def predict_from_date(self, input_date, plot=True):
        """
        Make a 2-year prediction from a specific historical date
        and compare with actual values if available
        """
        # Convert input date to datetime
        if isinstance(input_date, str):
            input_date = pd.to_datetime(input_date)
        
        # Find the closest date in the data
        idx = self.df['Date'].sub(input_date).abs().idxmin()
        prediction_date = self.df.loc[idx, 'Date']
        print(f"\nMaking prediction from: {prediction_date.date()}")
        
        # Get features for prediction
        X = self.df.loc[idx, self.feature_cols].values.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        actual = self.df.loc[idx, 'target'] if idx + 8 < len(self.df) else np.nan
        
        print(f"Predicted price 2 years later: ${prediction:.2f}")
        if not np.isnan(actual):
            print(f"Actual price 2 years later: ${actual:.2f}")
            pct_error = (prediction - actual) / actual * 100
            print(f"Prediction error: {pct_error:.2f}%")
        else:
            print("Actual future price not available in dataset")
        
        # Plot historical and future prices if available
        if plot:
            plt.figure(figsize=(12, 6))
            
            # Historical data
            hist_df = self.df[self.df['Date'] <= prediction_date]
            plt.plot(hist_df['Date'], hist_df['Close'], 'b-', label='Historical Prices')
            
            # Prediction point
            plt.plot(prediction_date, hist_df['Close'].iloc[-1], 'bo')
            
            # Actual future if available
            if not np.isnan(actual):
                future_date = prediction_date + pd.DateOffset(months=24)
                future_df = self.df[(self.df['Date'] > prediction_date) & 
                                  (self.df['Date'] <= future_date)]
                plt.plot(future_df['Date'], future_df['Close'], 'g-', label='Actual Future')
                plt.plot(future_date, actual, 'go')
            
            # Predicted future
            future_date = prediction_date + pd.DateOffset(months=24)
            plt.plot(future_date, prediction, 'ro', label='Predicted Future')
            
            plt.title(f'{self.company} Stock Price Prediction from {prediction_date.date()}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()
            plt.show()
        
        return prediction, actual if not np.isnan(actual) else None
    
    def run_pipeline(self):
        """Complete pipeline from data loading to model training"""
        self.load_and_preprocess_data()
        self.create_features(forecast_horizon=8)
        test_preds, y_test = self.train_model()
        
        # Plot test predictions
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Actual Prices')
        plt.plot(test_preds, label='Predicted Prices')
        plt.title(f'{self.company} - Actual vs Predicted Prices (2-year forecast)')
        plt.xlabel('Time Period')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        
        return self

if __name__ == "__main__":
    # Initialize predictor - change company as needed
    predictor = StockPredictor(company='AAPL').run_pipeline()
    
    # Example predictions from historical dates
    test_dates = ['2010-01-01', '2015-06-15', '2020-03-01']  
    
    for date in test_dates:
        predictor.predict_from_date(date)