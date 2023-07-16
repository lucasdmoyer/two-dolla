# from fastapi import FastAPI, File, UploadFile
# from fastapi.encoders import jsonable_encoder
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from starlette.middleware.cors import CORSMiddleware
# import json
# from yahoofinancials import YahooFinancials as YF
# import time
# import datetime
import pandas as pd
from pandas_datareader import data
# from typing import List, Optional
# import os
import pickle as pickle
from fastapi.encoders import jsonable_encoder
from backtest import Stock, Portfolio

from utils import *
import matplotlib.pyplot as plt
path=r"C:\Users\moyer\OneDrive\development\fin-dashboard\app\stocks.pkl"
# %matplotlib qt
# from tqdm import tqdm
# import scipy.stats as sp
# from utils import *
# from pydantic import BaseModel
# import matplotlib.pyplot as plt
# import quandl
#
tickers = ['MMM','AAPL',"MSFT","T"]
stocks = getStocksData(tickers, 100, save_new=False)
print(stocks.keys())
# save_new = True

# def convert_time(epoch):
#     return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch))
# days_back = 100
# epoch_time = int(time.time())
# day_epoch = 60*60*24
# # tickers = df['Symbol'][:10]
# stocks = {}
# if (save_new):
#     # print("@@@@#$$$$$$$$$$$$$$$$$$$$$$")
#     for tick in tqdm(tickers):
#         print(tick)
#         try:
#             stock_data = data.DataReader(tick,
#                                             start=convert_time(
#                                                 epoch_time - (int(days_back) * day_epoch)),
#                                             end=convert_time(epoch_time),
#                                             data_source='yahoo')
#             stocks[tick] = stock_data
#             with open(path, 'wb') as handle:
#                 pickle.dump(stocks, handle,
#                             protocol=pickle.HIGHEST_PROTOCOL)
#                 print(stocks)
#         except:
#             print("Skipping stock for {}, bad data :<".format(tick))
# else:
#     with open(path, 'rb') as handle:
#         stocks = pickle.load(handle)
#         print(stocks)



# stock = Stock(stocks['T'], 'T')
# strats = [stock.buyWeights(4)]
# stock.strategies = strats
# stock.backtest()
# print(stock.positions)



# port = Portfolio(stocks,10000);
# port.stocks = [Stock(stocks[x], x,port.initial_capital/len(tickers)) for x in tickers]
# markowtiz_params = {
#     'tickers': ['AAPL', 'T','MSFT'],
#     'weight':1
# }
# strats = [
#     port.markowitz(params = markowtiz_params,weight=200)
# ]
# port.strategies = strats
# port.backtest_portfolio()
# # port.positions.head()
# print(port.positions.total.plot())
# plt.show()