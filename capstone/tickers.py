import pandas as pd


nasdaq_tickers = pd.read_csv('nasdaq_tickers.csv')
nyse_tickers = pd.read_csv('nyse_tickers.csv')

both_tickers = pd.concat([nasdaq_tickers, nyse_tickers])
both_tickers.to_csv('tickers.csv', index=False)