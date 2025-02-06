#import sys
#print("Python path:", sys.executable)
import numpy as np
import pandas as pd
import yfinance as yf 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
from datetime import datetime
from sklearn.model_selection import train_test_split

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
end_date = datetime.now().strftime('%Y-%m-%d')
data = yf.download(tickers, start='2015-01-01', end=end_date)['Close']
print(yf.download(tickers, start='2015-01-01', end=end_date).columns)
returns = data.pct_change().dropna()

plt.figure(figsize=(12,6))
for stock in returns.columns:
    plt.plot(returns[stock], label=stock)
plt.legend()
plt.title("Stock Returns Over Time")
plt.show()

def moving_average(data, window):
    return data.rolling(window=window).mean()
def rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)  # Add small number to prevent division by zero
    return 100 - (100 / (1 + rs))

features = pd.DataFrame()
for stock in tickers:
    features[f'{stock}_SMA'] = moving_average(data[stock], 20)
    features[f'{stock}_RSI'] = rsi(data[stock])
features.dropna(inplace=True)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(returns.T)
cluster_map = dict(zip(tickers, cluster_labels))
print("Stock Clusters:", cluster_map)

X = features[:-1]
y = returns.shift(-1).dropna()
rf_models = {}
predictions = {}

train_size = int(len(X) * 0.8)
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

for stock in tickers:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train[stock])
    rf_models[stock] = model
    predictions[stock] = model.predict(X[-1:].values)

def portfolio_volatility(weights, returns):
    portfolio_return = np.dot(weights, np.mean(returns, axis=0))
    portfolio_vol = np.sqrt(weights.T @ returns.cov() @ weights)
    return portfolio_vol

initial_weights = np.ones(len(tickers)) / len(tickers)
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = tuple((0, 1) for _ in range(len(tickers)))
opt_result = minimize(portfolio_volatility, initial_weights, args=(returns,), bounds=bounds, constraints=constraints)
optimized_weights = opt_result.x
portfolio_allocations = dict(zip(tickers, optimized_weights))
print("Optimized Portfolio Weights:", portfolio_allocations)

# Step 7: Backtesting Portfolio Performance
optimized_returns = (returns * optimized_weights).sum(axis=1)
cumulative_returns = (1 + optimized_returns).cumprod()
plt.figure(figsize=(12,6))
plt.plot(cumulative_returns, label="Optimized Portfolio")
plt.legend()
plt.title("Portfolio Performance")
plt.show()

# Save the script as a Python file
if __name__ == "__main__":
    try:
        data.to_csv("stock_data.csv")
        features.to_csv("features.csv")
        pd.DataFrame(predictions, index=[0]).to_csv("predictions.csv")
        print("Files saved successfully")
    except Exception as e:
        print(f"Error saving files: {e}")
