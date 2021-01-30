from fastapi import FastAPI, File, UploadFile
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
        "name": "items",
        "description": "Manage items. So _fancy_ they have their own docs.",
        "externalDocs": {
            "description": "Items external docs",
            "url": "https://fastapi.tiangolo.com/",
        },
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
async def getStocks():
    # data = getData("AAPL")
    # print(data["High"])
    # return(data["High"])
    tickers = getTickers()
    #tickers = ["MSFT","AAPL"]
    data = getStocksData(tickers, 365)
    print(data)
    return("Success! {} stock data gotten".format(len(data)))
    # if (os.path.exists("../stocks.pkl")):
    #     with open("../stocks.pkl", "rb") as pkl_handle:
    #         stocks = pickle.load(pkl_handle)
    #         return(stocks["AAPL"])d

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

@app.post("/black-sholes-option-price", tags=["markow"])
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



# function to help process csv inputs
def convertBytesToString(bytes):
    data = bytes.decode('utf-8').splitlines()
    df = pd.DataFrame(data)
    return(df)

def getMarketCap(ticker):
    req = YF(ticker).get_summary_data()[ticker]['marketCap']
    return(req)

def convert_time(epoch):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch))

def getTickers():
    table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    #df.to_csv('S&P500-Info.csv')
    #df.to_csv("S&P500-Symbols.csv", columns=['Symbol'])
    tickers = df['Symbol'][:50]
    return tickers

def getStocksData(tickers, days_back):
    epoch_time = int(time.time())
    day_epoch = 60*60*24
    stocks = {}
    print(tickers)
    for tick in tqdm(tickers):
        #stock_data = getData(tick)
        #stocks[tick] = stock_data 
        # stock_data = data.DataReader(tick, 
        #                 start=convert_time(epoch_time - (10* day_epoch)), 
        #                 end=convert_time(epoch_time), 
        #                 data_source='yahoo')
        # return(stock_data)
        try:
            stock_data = data.DataReader(tick, 
                        start=convert_time(epoch_time - (int(days_back)* day_epoch)), 
                        end=convert_time(epoch_time), 
                        data_source='yahoo')
            stocks[tick] = stock_data 
        except Exception as e:
            print("ERROR : "+str(e))
            print("Skipping stock for {}, bad data :<".format(tick))
    return stocks

# def getData(tick):
#     epoch_time = int(time.time())
#     day_epoch = 60*60*24
#     try:
#         print("trying!!")
#         stock_data = data.DataReader(tick, 
#                     start=convert_time(epoch_time - (10* day_epoch)), 
#                     end=convert_time(epoch_time), 
#                     data_source='yahoo')
#         print("again!!")
#         return(stock_data)

#     except Exception as e:
#         print("ERROR : "+str(e))
#         print("Skipping stock for {}, bad data :<".format(tick))
    
