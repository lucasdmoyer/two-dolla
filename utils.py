import numpy as np
import pandas as pd 
import scipy.stats as sp
import time
from tqdm import tqdm

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

def getStocks(tickers, days_back):
    def convert_time(epoch):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch))
    
    epoch_time = int(time.time())
    day_epoch = 60*60*24
    # tickers = df['Symbol'][:10]
    days_back = 100
    stocks = {}
    for tick in tqdm(tickers):
        try:
            stock_data = data.DataReader(tick, 
                            start=convert_time(epoch_time - (int(days_back)* day_epoch)), 
                            end=convert_time(epoch_time), 
                            data_source='yahoo')
            stocks[tick] = stock_data 
        except:
            print("Skipping stock for {}, bad data :<".format(tick))
    return stocks


def getPortReturns(stocks):
    df = pd.DataFrame()
    #df.index = stocks[list(stocks.keys())[0]].index
    for stock in list(stocks):
        df[stock] = stocks[stock]['simple_returns']
    return df.dropna()

def dict_2_panel(stocks_lagged):  
    df = pd.DataFrame()
    for stock in list(stocks):
        stocks_lagged[stock]['ticker'] = stock
        df = df.append(stocks_lagged[stock])
    return df

def addMACD(stocks, short, long):
    for stock in list(stocks):
        exp1 = stocks[stock]['Adj Close'].ewm(span=short, adjust=False).mean()
        exp2 = stocks[stock]['Adj Close'].ewm(span=long, adjust=False).mean()
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


def addReturns(stocks, col_name):
    for stock in list(stocks):
        stocks[stock]['simple_returns'] = stocks[stock][col_name].pct_change()
        stocks[stock]['log_returns'] = np.log(
            stocks[stock]['simple_returns']+1)
        stocks[stock]['cum_daily_return'] = (
            (1 + stocks[stock]['simple_returns']).cumprod() - 1)
    return(stocks)


def addVol(stocks, periods):
    for stock in list(stocks):
        stocks[stock]['volatility'] = stocks[stock]['simple_returns'].rolling(
            periods).std() * np.sqrt(periods)
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
