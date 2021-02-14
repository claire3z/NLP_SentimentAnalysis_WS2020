"Owner: Claire SUN"

''"""
Created by: Claire Z. Sun
Date: 2021.02.12
''"""

# Installation: $pip install yfinance --upgrade --no-cache-dir OR $ conda install -c ranaroussi yfinance
# Ref "yfinance": https://github.com/ranaroussi/yfinance
# Note: pip install yahoo-finance is out-dated (last version 2016)
#       https://pypi.org/project/yahoo-finance/#history


import yfinance as yf
import pandas as pd

def Download_SharePrice(save_path, ticker,start,end):
    data_YF = yf.download(ticker, start, end)
    data_YF.to_csv(save_path)


# # Download daily share price and trading volume
# start = "2020-11-01"
# end = "2021-01-05"
# tickers_YF = "TSLA NIO NKLA MSFT GOOGL GM"
# filename = tickers_YF+'.'+end+'.csv'
# data_YF = yf.download(tickers_YF, start,end)
# data_YF.to_csv(filename)
# print("{} days of trading data downloaded and saved at {}".format(len(data_YF),filename))
#
#
# # Combine files
# files = ['TSLA NIO NKLA MSFT GOOGL GM.2020-12-08.csv','','']
# master = 'YFinance_SharePrice_all.csv'
# combined = pd.concat([pd.read_csv(filename,header=[0, 1],index_col=0,infer_datetime_format=True) for f in files])
# combined.to_csv(master)
#
#
# # Load share price data
# df = pd.read_csv(filepath_or_buffer = master,header=[0, 1],index_col=0,infer_datetime_format=True)
#
# # df.head()
# # len(df)
# # df.index, df.columns
# # df['Adj Close','GM'] # just the first column values
# # df.columns.get_level_values(0)
# # df.columns.values
#
