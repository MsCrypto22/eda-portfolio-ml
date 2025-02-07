# EDA Portfolio ML

## Overview
This project explores Exploratory Data Analysis (EDA) and Portfolio Optimization using Machine Learning techniques. It leverages historical stock price data, feature engineering, clustering, return forecasting, and portfolio optimization to build an optimal investment strategy.

## Features
- **Data Acquisition**: Fetches historical stock price data using `yfinance`.
- **EDA**: Visualizes stock returns and trends.
- **Feature Engineering**: Computes moving averages and RSI indicators.
- **Clustering**: Groups assets using K-Means clustering.
- **Return Forecasting**: Uses a Random Forest model to predict future stock returns.
- **Portfolio Optimization**: Implements Modern Portfolio Theory (MPT) to optimize asset allocation.
- **Backtesting**: Evaluates portfolio performance over time.

## Installation
Clone this repository and install dependencies:
```bash
git clone https://github.com/your-username/eda-portfolio-ml.git
cd eda-portfolio-ml
pip install -r requirements.txt
```

## Usage
Run the main script to execute the analysis:
```bash
python eda_portfolio_ml.py
```

## Dependencies
- `numpy`
- `pandas`
- `yfinance`
- `matplotlib`
- `seaborn`
- `sklearn`
- `scipy`

Install missing packages using:
```bash
pip install -r requirements.txt
```

## Results
- **Stock Clustering**: Groups stocks based on return patterns.
- **Optimized Portfolio Weights**: Allocates assets for risk minimization.
- **Performance Plots**: Visualizes portfolio growth.

## Contributing
Feel free to fork this repository and submit pull requests for improvements or additional features.
- Will be adding more code to this eda over time

## License
This project is licensed under the MIT License.

