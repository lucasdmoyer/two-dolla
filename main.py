from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware
import json
from yahoofinancials import YahooFinancials as YF
import time
import pandas as pd
from pandas_datareader import data
import os
import pickle
from pydantic import BaseModel
from tqdm import tqdm
import scipy.stats as sp
from utils import *
from backtester import *
from typing import List, Optional, Dict

tags_metadata = [
    {
        "name": "markow",
        "description": 'Put in a list of tickers, and it will give you a markowitz portfolio optimization. Example, "AAPL,MSFT,T" ',
    },
    {
        "name": "black-sholes",
        "description": 'Put in c for call or p for put, and fill in the rest of the options detail to get the option price',
    },
]


class Weights(BaseModel):
    # weights:  List[float] = []
    weights: Dict[str, float] = None


app = FastAPI(openapi_tags=tags_metadata)

# origins = [
#     "http://localhost:4200/",
#     "https://localhost:4200/",
#     "http://localhost",
#     "http://localhost:8080",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
app.add_middleware(CORSMiddleware, allow_origins=[
                   "*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])
# uvicorn main:app --reload


@app.get("/")
async def root():
    return {"message": "Hello! Please go to /docs to continue!"}


@app.get("/getMarketCap")
async def marketCap(ticker):
    return getMarketCap(ticker)


@app.get("/getStocks")
async def getStocks(ticker_data: str = "MSFT,TSLA,AAPL,T", days_back: int = 365):
    tickers = ticker_data.split(",")
    stocks = getStockData(tickers, 365)
    result = {}
    for stock in list(stocks):
        stocks[stock]['Date'] = stocks[stock].index.strftime('%Y-%m-%d')
        result[stock] = stocks[stock].to_dict(orient='records')
    json_compatible_item_data = jsonable_encoder(result)
    return json_compatible_item_data


@app.get("/getBars")
async def getBars(ticker_data: str = "AMD,GOOG,MMM", initial_captial: int = 100000):
    tickers = ticker_data.split(",")
    print(os.getcwd())
    with open("./stocks.pkl", "rb") as pkl_handle:
        bars = pickle.load(pkl_handle)
    # tickers = ['GOOG', 'MMM', 'AMD']
    bars = {key: bars[key] for key in tickers}
    bars = addReturns(bars, 'Close')
    strategy = [
        {
            'name': 'MACDStrat',
            'params': {
                'long': 26,
                'short': 12
            }
        }
    ]
    rfs = RandomStrategy(tickers, bars, strategy)
    signals = rfs.genSignals()
    portfolio = MyPortfolio(tickers, bars, signals, initial_captial)
    returns = portfolio.backtest_portfolio()
    result = {}
    for stock in list(bars):
        bars[stock]['Date'] = bars[stock].index.strftime('%Y-%m-%d')
        result[stock] = bars[stock].dropna().to_dict(orient='records')
    json_compatible_item_data = jsonable_encoder(result)
    return json_compatible_item_data


@app.get("/backtestStrategy")
async def getBars(ticker_data: str = "AMD,GOOG,MMM", initial_captial: int = 100000):
    tickers = ticker_data.split(",")
    print(os.getcwd())
    with open("./stocks.pkl", "rb") as pkl_handle:
        bars = pickle.load(pkl_handle)
    # tickers = ['GOOG', 'MMM', 'AMD']
    bars = {key: bars[key] for key in tickers}
    bars = addReturns(bars, 'Close')
    strategy = [
        {
            'name': 'MACDStrat',
            'params': {
                'long': 26,
                'short': 12
            }
        }
    ]
    rfs = RandomStrategy(tickers, bars, strategy)
    signals = rfs.genSignals()
    portfolio = MyPortfolio(tickers, bars, signals, initial_captial)
    returns = portfolio.backtest_portfolio()
    returns = returns.dropna().to_dict(orient='records')
    # result={}
    # for stock in list(bars):
    #     bars[stock]['Date'] = bars[stock].index.strftime('%Y-%m-%d')
    #     result[stock] = bars[stock].dropna().to_dict(orient='records')

    json_compatible_item_data = jsonable_encoder(returns)
    return json_compatible_item_data


@app.get("/getForecast")
async def getBars(ticker_data: str = "AMD,GOOG,MMM", initial_captial: int = 100000):
    tickers = ticker_data.split(",")
    print(os.getcwd())
    with open("./stocks.pkl", "rb") as pkl_handle:
        bars = pickle.load(pkl_handle)
    # tickers = ['GOOG', 'MMM', 'AMD']
    bars = {key: bars[key] for key in tickers}
    bars = addReturns(bars, 'Close')
    strategy = [
        {
            'name': 'MACDStrat',
            'params': {
                'long': 26,
                'short': 12
            }
        }
    ]
    rfs = RandomStrategy(tickers, bars, strategy)
    signals = rfs.genSignals()
    portfolio = MyPortfolio(tickers, bars, signals, initial_captial)
    returns = portfolio.backtest_portfolio()
    # print(returns)
    forecast = portfolio.forecast_portfolio()
    # returns['total'].plot().get_figure().savefig('output.png')
    bars = forecast

    result = {}
    for stock in list(bars):
        print(bars[stock].columns)
        bars[stock]['Date'] = bars[stock].index.strftime('%Y-%m-%d')
        bars[stock] = bars[stock].fillna('')
        result[stock] = bars[stock].to_dict(orient='records')
        # result[stock] = bars[stock].to_json(orient='records')
    json_compatible_item_data = jsonable_encoder(result)
    return json_compatible_item_data


@app.get("/backtestSingleStock")
async def getBars(ticker: str = "MMM", initial_captial: int = 100000):
    with open("./stocks.pkl", "rb") as pkl_handle:
        bars = pickle.load(pkl_handle)
    # tickers = ['GOOG', 'MMM', 'AMD']
    tickers = [ticker]
    bars = {key: bars[key] for key in tickers}
    bars = addReturns(bars, 'Close')
    strategy = [
        # {
        #     'name':'MACDStrat',
        #     'params':{
        #         'long': 26,
        #         'short': 12
        #     }
        # },
        {
            'name': 'single',
            'params': {
                'ticker': ticker
            }
        }
    ]
    rfs = RandomStrategy(tickers, bars, strategy)
    signals = rfs.genSignals()
    portfolio = MyPortfolio(tickers, bars, signals, initial_captial)
    returns = portfolio.backtest_portfolio()
    returns = returns.dropna().to_dict(orient='records')
    # returns['total'].plot().get_figure().savefig('output.png')
    # result={}
    # for stock in list(bars):
    #     bars[stock]['Date'] = bars[stock].index.strftime('%Y-%m-%d')
    #     result[stock] = bars[stock].dropna().to_dict(orient='records')

    json_compatible_item_data = jsonable_encoder(returns)
    return json_compatible_item_data


@app.get("/getTickers")
async def getTickers():
    with open("./stocks.pkl", "rb") as pkl_handle:
        bars = pickle.load(pkl_handle)
    # tickers = ['GOOG', 'MMM', 'AMD']
    return list(bars.keys())


@app.post("/backtestPortfolio")
async def getBars(
        weights_data: Weights = {
            "weights":
                {
                    "AMD": 0.33,
                    "GOOG": 0.33,
                    "MMM": 0.33
                }
        },
        initial_captial: int = 1000):
    with open("./stocks.pkl", "rb") as pkl_handle:
        bars = pickle.load(pkl_handle)
    # tickers = ['GOOG', 'MMM', 'AMD']
    # tickers = [ticker]
    #tickers = ticker_data.split(",")
    tickers = weights_data.weights.keys()
    bars = {key: bars[key] for key in tickers}
    bars = addReturns(bars, 'Close')
    strategy = [
        {
            'name': 'weighted',
            'params': weights_data
        }
    ]
    rfs = RandomStrategy(tickers, bars, strategy)
    signals = rfs.genSignals()
    portfolio = MyPortfolio(tickers, bars, signals, initial_captial)

    returns = portfolio.backtest_portfolio()
    # limit to past 50 days
    returns = returns.tail(20)

    forecasted_initial_cap = returns['total'].iat[-1]

    forecast = portfolio.forecast_portfolio()
    forecast = addReturns(forecast, 'yhat')

    forecast_rfs = RandomStrategy(tickers, forecast, strategy)
    forecasted_signals = forecast_rfs.genSignals()
    forecasted_portfolio = MyPortfolio(
        tickers, forecast, forecasted_signals, forecasted_initial_cap)
    forecast_returns = forecasted_portfolio.backtest_forecast().dropna()

    result = pd.concat([returns, forecast_returns], axis=1).fillna('null')
    result['date'] = result.index
    result = result.to_dict(orient='records')
    #result = pd.concat([returns, forecast_returns], axis=1).fillna('null').replace(to_replace=0, value="null").to_dict(orient='records')

    json_compatible_item_data = jsonable_encoder(result)
    return json_compatible_item_data


@app.post("/portfolio")
async def portfolio(file_path, file: UploadFile = File(...)):
    contents = await file.read()
    df = convertBytesToString(contents)
    tickers = list(df.iloc[:, 0])
    numstocks = len(tickers)
    stocks = getStocksData(tickers, 365)
    stocks = addBB(stocks, 20)
    stocks = addMACD(stocks)
    stocks = addRSI(stocks, 14)
    stocks = addReturns(stocks)
    stocks = addVol(stocks, 50)
    # if (not os.path.exists("file_path")):
    #     with open(file_path + "stocks.pkl", "wb") as pkl_handle:
    #         pickle.dump(stocks, pkl_handle)
    return("Success! {} stock data gotten and saved to {}".format(len(stocks), file_path + "stocks.pkl"))


# file: UploadFile=File(...)
@app.post("/markowitz-optimize-portfolio", tags=["markow"])
async def optimizePortfolio(markov_runs, MSR_or_GMV, ticker_data="AMD,GOOG,MMM"):
    # contents = await file.read()
    # df = convertBytesToString(contents)
    # tickers = list(df.iloc[:,0])
    # tickers = ticker_data.split(",")
    # stocks = getStocksData(tickers, 365)
    with open("./stocks.pkl", "rb") as pkl_handle:
        bars = pickle.load(pkl_handle)
    tickers = ticker_data.split(",")
    bars = {key: bars[key] for key in tickers}
    bars = addReturns(bars, 'Close')
    # stocks = addBB(stocks, 20)
    # stocks = addMACD(stocks)
    # stocks = addRSI(stocks, 14)
    # stocks = addReturns(stocks)
    # stocks = addVol(stocks, 50)
    port_returns = getPortReturns(bars)
    risk_free = 0
    df = pd.DataFrame(columns=["id", "return", "volatility", "weights"])
    for x in range(0, int(markov_runs)):
        weights = getRandomWeights(len(tickers))
        volatility = getPortWeightedVol(port_returns, weights)
        ann_ret = getPortWeightedAnnualReturn(port_returns, weights)
        row = {
            "id": x,
            "return": ann_ret,
            "volatility": volatility,
            "weights": weights
        }
        df = df.append(row, ignore_index=True)
    df["sharpe"] = (df["return"] - risk_free) / df["volatility"]

    MSR = df.sort_values(by=["sharpe"], ascending=False).head(1)
    GMV = df.sort_values(by=["volatility"], ascending=True).head(1)

    # MSR_weights = [{tickers[index]:x.astype(
    #     'str')} for index, x in enumerate(list(MSR['weights'])[0])]

    MSR_weights = [{tickers[index]:x}
                   for index, x in enumerate(list(MSR['weights'])[0])]
    weights = {}
    for index, x in enumerate(list(MSR['weights'])[0]):
        weights[tickers[index]] = x
    # MSR['weights']

    result = {}
    result['weights'] = weights
    json_compatible_item_data = jsonable_encoder(result)
    return json_compatible_item_data

    # GMV_weights = [{tickers[index]:x.astype(
    #     'str')} for index, x in enumerate(list(GMV['weights'])[0])]
    # GMV_weights = [x.astype('str') for x in list(GMV['weights'])[0]]
    # if (MSR_or_GMV == "MSR"):
    #     result = {}
    #     result["return"] = MSR['return'][MSR['return'].keys()[0]]
    #     result["volatility"] = MSR['volatility'][MSR['volatility'].keys()[0]]
    #     result["weights"] = MSR_weights
    #     return(result)
    # elif(MSR_or_GMV == "GMV"):
    #     result = {}
    #     result["return"] = GMV['return'][GMV['return'].keys()[0]]
    #     result["volatility"] = GMV['volatility'][GMV['volatility'].keys()[0]]
    #     result["weights"] = GMV_weights
    #     return(result)
    # else:
    #     return("Neither GMV or MSR chosen")


@app.post("/black-sholes-option-price", tags=["black-sholes"])
async def optimizePortfolio(c_or_p, price, strike, risk_free_rate, days, volatility):
    a = BsmModel(c_or_p, float(price), float(strike), float(
        risk_free_rate), float(days)/365, float(volatility))
    return(a.bsm_price())
