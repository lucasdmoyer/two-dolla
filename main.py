from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import json
from yahoofinancials import YahooFinancials as YF
import time
import pandas as pd
from pandas_datareader import data
import os
import pickle 
from tqdm import tqdm
import scipy.stats as sp
from utils import *

tags_metadata = [
    {
        "name": "markow",
        "description": 'Put in a list of tickers, and it will give you a markowitz portfolio optimization. Example, "AAPL,MSFT,T" ' ,
    },
    {
        "name": "black-sholes",
        "description": 'Put in c for call or p for put, and fill in the rest of the options detail to get the option price' ,
    },
]

app = FastAPI(openapi_tags=tags_metadata)

# uvicorn main:app --reload

@app.get("/")
async def root():
    return {"message": "Hello! Please go to /docs to continue!"}

@app.get("/getMarketCap")
async def marketCap(ticker):
    return getMarketCap(ticker)

@app.get("/getStocks")
async def getStocks(ticker_data: str="MSFT,TSLA,AAPL,T",days_back: int = 365):
    tickers = ticker_data.split(",")
    stocks = getStockData(tickers,365)
    result={}
    for stock in list(stocks):
        stocks[stock]['Date'] = stocks[stock].index.strftime('%Y-%m-%d')
        result[stock] = stocks[stock].to_dict()
    json_compatible_item_data = jsonable_encoder(result)
    return json_compatible_item_data


@app.post("/portfolio")
async def portfolio(file_path,file: UploadFile=File(...)):
    contents = await file.read()
    df = convertBytesToString(contents)
    tickers = list(df.iloc[:,0])
    numstocks = len(tickers)
    stocks = getStocksData(tickers, 365)
    stocks = addBB(stocks,20)        
    stocks= addMACD(stocks)
    stocks = addRSI(stocks,14)
    stocks = addReturns(stocks)
    stocks = addVol(stocks,50)
    print(stocks)
    # if (not os.path.exists("file_path")):
    #     with open(file_path + "stocks.pkl", "wb") as pkl_handle:
    #         pickle.dump(stocks, pkl_handle)
    return("Success! {} stock data gotten and saved to {}".format(len(stocks), file_path + "stocks.pkl"))
# file: UploadFile=File(...)
@app.post("/markowitz-optimize-portfolio", tags=["markow"])
async def optimizePortfolio(markov_runs, MSR_or_GMV, ticker_data):
    # contents = await file.read()
    # df = convertBytesToString(contents)
    # tickers = list(df.iloc[:,0])
    tickers = ticker_data.split(",")
    stocks = getStocksData(tickers, 365)
    stocks = addBB(stocks,20)        
    stocks= addMACD(stocks)
    stocks = addRSI(stocks,14)
    stocks = addReturns(stocks)
    stocks = addVol(stocks,50)
    port_returns = getPortReturns(stocks)
    risk_free= 0
    df = pd.DataFrame(columns=["id","return","volatility","weights"])
    for x in range(0,int(markov_runs)):
        weights = getRandomWeights( len(tickers))
        volatility = getPortWeightedVol(port_returns, weights)
        ann_ret = getPortWeightedAnnualReturn(port_returns,weights)
        row = {
            "id":x,
            "return":ann_ret,
            "volatility":volatility,
            "weights":weights
        }
        df = df.append(row,ignore_index=True)
    df["sharpe"] = (df["return"] - risk_free) / df["volatility"]

    MSR = df.sort_values(by=["sharpe"], ascending=False).head(1)
    GMV = df.sort_values(by=["volatility"],ascending=True).head(1)
    
    MSR_weights = [{tickers[index]:x.astype('str')} for index, x in enumerate(list(MSR['weights'])[0])]
    GMV_weights = [{tickers[index]:x.astype('str')} for index, x in enumerate(list(GMV['weights'])[0])]
    GMV_weights = [x.astype('str') for x in list(GMV['weights'])[0]]
    if (MSR_or_GMV == "MSR"):
        result={}
        result["return"] = MSR['return'][MSR['return'].keys()[0]]
        result["volatility"] = MSR['volatility'][MSR['volatility'].keys()[0]]
        result["weights"] = MSR_weights
        return(result)
    elif(MSR_or_GMV == "GMV"):
        result={}
        result["return"] = GMV['return'][GMV['return'].keys()[0]]
        result["volatility"] = GMV['volatility'][GMV['volatility'].keys()[0]]
        result["weights"] = GMV_weights
        return(result)
    else:
        return("Neither GMV or MSR chosen")

@app.post("/black-sholes-option-price", tags=["black-sholes"])
async def optimizePortfolio(c_or_p, price, strike, risk_free_rate, days, volatility):
    a = BsmModel(c_or_p , float(price), float(strike), float(risk_free_rate), float(days)/365, float(volatility))
    return(a.bsm_price())



class BsmModel:
    def __init__(self, option_type, price, strike, interest_rate, expiry, volatility, dividend_yield=0):
        self.s = price # Underlying asset price
        self.k = strike # Option strike K
        self.r = interest_rate # Continuous risk fee rate
        self.q = dividend_yield # Dividend continuous rate
        self.T = expiry # time to expiry (year)
        self.sigma = volatility # Underlying volatility
        self.type = option_type # option type "p" put option "c" call option
    def n(self, d):
        # cumulative probability distribution function of standard normal distribution
        return sp.norm.cdf(d)

    def dn(self, d):
        # the first order derivative of n(d)
        return sp.norm.pdf(d)

    def d1(self):
        d1 = (np.log(self.s / self.k) + (self.r - self.q + self.sigma ** 2 * 0.5) * self.T) / (self.sigma * np.sqrt(self.T))
        return d1

    def d2(self):
        d2 = (np.log(self.s / self.k) + (self.r - self.q - self.sigma ** 2 * 0.5) * self.T) / (self.sigma * np.sqrt(self.T))
        return d2

    def bsm_price(self):
        d1 = self.d1()
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if self.type == 'c':
            price = np.exp(-self.r*self.T) * (self.s * np.exp((self.r - self.q)*self.T) * self.n(d1) - self.k * self.n(d2))
            return price
        elif self.type == 'p':
            price = np.exp(-self.r*self.T) * (self.k * self.n(-d2) - (self.s * np.
                                                                      exp((self.r - self.q)*self.T) * self.n(-d1)))
            return price
        else:
            print("option type can only be c or p")


def convert_time(epoch):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch))

def getStockData(tickers, days_back):
    stocks = {}
    epoch_time = int(time.time())
    day_epoch = 60*60*24
    for tick in tqdm(tickers):
        try:
            stock_data = data.DataReader(tick, 
                        start=convert_time(epoch_time - (days_back* day_epoch)), 
                        end=convert_time(epoch_time), 
                        data_source='yahoo')
            stocks[tick] = stock_data 
        except:
            print("Skipping stock for {}, bad data :<".format(tick))
    return stocks

def getSP500Tickers():
    table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    df.to_csv('S&P500-Info.csv')
    df.to_csv("S&P500-Symbols.csv", columns=['Symbol'])
    tickers = df['Symbol']
    return tickers


def getPortReturns(stocks):
    df = pd.DataFrame()
    #df.index = stocks[list(stocks.keys())[0]].index
    for stock in list(stocks):
        df[stock] = stocks[stock]['simple_returns']
    return df.dropna()


def addMACD(stocks):
    for stock in list(stocks):
        exp1 = stocks[stock]['Adj Close'].ewm(span=12, adjust=False).mean()
        exp2 = stocks[stock]['Adj Close'].ewm(span=26, adjust=False).mean()
        stocks[stock]['macd'] = exp1-exp2
        stocks[stock]['macd_signal'] = stocks[stock]['macd'].ewm(
            span=9, adjust=False).mean()
    return(stocks)


def computeRSI(data, time_window):
    diff = data.diff(1).dropna()        # diff in one field(one day)

    # this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[diff > 0]

    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[diff < 0]
    up_chg_avg = up_chg.ewm(com=time_window-1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(
        com=time_window-1, min_periods=time_window).mean()

    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi


def addRSI(stocks, time_window):
    for stock in list(stocks):
        stocks[stock]["RSI"] = computeRSI(
            stocks[stock]["Adj Close"], time_window)
    return(stocks)


def addBB(stocks, time_window):
    for stock in list(stocks):
        stocks[stock]['MA20'] = stocks[stock]['Adj Close'].rolling(
            window=time_window).mean()
        stocks[stock]['20dSTD'] = stocks[stock]['Adj Close'].rolling(
            window=time_window).std()
        stocks[stock]['UpperBB'] = stocks[stock]['MA20'] + \
            (stocks[stock]['20dSTD'] * 2)
        stocks[stock]['LowerBB'] = stocks[stock]['MA20'] - \
            (stocks[stock]['20dSTD'] * 2)
        stocks[stock]['LowerBB_dist'] = stocks[stock]['LowerBB'] - \
            stocks[stock]['MA20']
        stocks[stock]['UpperBB_dist'] = stocks[stock]['MA20'] - \
            stocks[stock]['UpperBB']
    return(stocks)


def addReturns(stocks):
    for stock in list(stocks):
        stocks[stock]['simple_returns'] = stocks[stock]['Adj Close'].pct_change()
        stocks[stock]['log_returns'] = np.log(
            stocks[stock]['simple_returns']+1)
        stocks[stock]['cum_daily_return'] = (
            (1 + stocks[stock]['simple_returns']).cumprod() - 1)
    return(stocks)


def addVol(stocks, periods):
    for stock in list(stocks):
        stocks[stock]['volatility'] = (stocks[stock]['simple_returns'].rolling(
            periods).std() * np.sqrt(periods))
    return stocks


def lagFeatures(stocks, features, periods, returns):
    # sets the columns we want in our final df
    cols_wanted = features + returns
    stocks_lagged = stocks.copy()
    print(f'The columns wanted are {cols_wanted}')
    for stock in list(stocks):
        stocks_lagged[stock][features] = stocks_lagged[stock][features].shift(
            periods)
        stocks_lagged[stock] = pd.DataFrame(
            stocks_lagged[stock], columns=cols_wanted)
    return(stocks_lagged)


def getRandomWeights(numstocks):
    weights = np.random.rand(numstocks)
    return (weights/np.sum(weights))


def getPortWeightedReturns(port_ret, weights):
    assert(len(port_ret.columns) == len(weights))
    return port_ret.iloc[:, 0:len(weights)].mul(weights, axis=1).sum(axis=1)
# getRandomWeights(50)


def getPortWeightedVol(port_ret, weights):
    cov_mat = port_ret.cov()
    #cov_mat_annual = cov_mat * 252
    # cov_mat_annual
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
    return port_vol


def getPortWeightedAnnualReturn(port_ret, weights):
    returns = getPortWeightedReturns(port_ret, weights)

    mean_return_daily = np.mean(returns)
    # Calculate the implied annualized average return
    mean_return_annualized = ((1+mean_return_daily)**252)-1
    return(mean_return_annualized)

def dict_2_panel(stocks):  
    df = pd.DataFrame()
    for stock in list(stocks):
        stocks[stock]['ticker'] = stock
        df = df.append(stocks[stock])
    return df.dropna()

def graphPrice(stocks, ticker):
    plt.figure(figsize=(10,10))
    plt.plot(stocks[ticker].index, stocks[ticker]['Adj Close'])
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.title(ticker + " price")
    plt.savefig('./graphs/{}_price.png'.format(ticker))
    
def getSP500Tickers():
    table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    df.to_csv('S&P500-Info.csv')
    df.to_csv("S&P500-Symbols.csv", columns=['Symbol'])
    tickers = df['Symbol']
    return list(tickers)

def convertBytesToString(bytes):
    data = bytes.decode('utf-8').splitlines()
    df = pd.DataFrame(data)
    return(df)
