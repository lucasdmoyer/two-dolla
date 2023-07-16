
from abc import ABCMeta, abstractmethod
import pandas as pd
from utils import *


class Strategy(object):
    """Strategy is an abstract base class providing an interface for
    all subsequent (inherited) trading strategies.

    The goal of a (derived) Strategy object is to output a list of signals,
    which has the form of a time series indexed pandas DataFrame.

    In this instance only a single symbol/instrument is supported."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_signals(self):
        """An implementation is required to return the DataFrame of symbols 
        containing the signals to go long, short or hold (1, -1 or 0)."""
        raise NotImplementedError("Should implement generate_signals()!")


# backtest.py
class Portfolio(object):
    """An abstract base class representing a portfolio of 
    positions (including both instruments and cash), determined
    on the basis of a set of signals provided by a Strategy."""

    __metaclass__ = ABCMeta

    def __init__(self, tickers, bars, initial_capital=1000):
        self.tickers = tickers
        self.bars = bars # dict of the ticker:OHLC df
        self.stock_list = []
        self.stocks = {}  # dict of Class Stock[]
        # self.stocks = [Stock(self.stocks[x],x, self.initial_capital/len(self.tickers)) for x in self.tickers]
        self.backtest = {}
        self.positions = {} # might not be needed if the self.stocks will have the positions, we can store the aggregated positions here
        self.strategies = []
        self.initial_capital = initial_capital
        self.prepStockData()
        
    # takes a strat or list of stras and adds it to the stocks stratigies to execute
    # all takes a list of tickers to put the strategy in
    def addStrat(self, tickers, strat, **kwargs):
#         print('adding strat ', tickers, strat)
        for tick in tickers:
            if isinstance(strat, str):
                print("adding single strat")
                try:
                    func = getattr(self.stocks[tick], strat)(**kwargs)
#                     print(func)
                    self.stocks[tick].strategies.append(func)
                except AttributeError:
                    print("dostuff not found")
                
            else:
                print("adding multiple strats")
                print(self.stocks)
                
#                 self.stocks[tick].strategies.extend(strat)
                for signal in strat:
#                     print('signal')
#                     print(signal)
#                     print('signal value:')
#                     print(signal['value'])
                    self.stocks[signal['key']].strategies.append(signal['value']) 
               

    # creates the Stock objects and adds them to the self.stocks dict
    def prepStockData(self):
        for ticker in self.tickers:
            self.stocks[ticker] = Stock(self.bars[ticker],ticker,1000) 


    def backtest_portfolio(self):
        for stock in self.tickers:
            self.stocks[stock].backtest()
        sample_df = self.stocks[self.tickers[0]].bars
#         print(sample_df)
        self.positions = pd.DataFrame(index=self.bars[next(iter(self.bars))].index) # empty df with same index as bars
        all_positions = [self.stocks[x].positions for x in self.tickers]
        self.positions = pd.DataFrame(index=sample_df[next(iter(sample_df))].index) # empty df with same index as bars

        sum_cols = ['holdings', 'cash', 'total']
        for col in sum_cols:
            self.positions[col] = pd.concat(all_positions).groupby('Date')[col].sum()

        return self.positions

    # returns a dict of the symbol:[a df of the strategy with a Date index]
    def markowitz(self, params, weight=1):
        df = pd.DataFrame(index=self.bars[next(iter(self.bars))].index)
        bars = addReturns(self.bars, 'Adj Close')
        port_returns = getPortReturns(bars)
        risk_free = 0
        markov_runs = 100
        tickers = self.bars.keys()
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
            # df = df.append(row, ignore_index=True)
            # df = df.concat([df,row])
            df = pd.concat([df, pd.DataFrame.from_records([row])])
        df["sharpe"] = (df["return"] - risk_free) / df["volatility"]

        MSR = df.sort_values(by=["sharpe"], ascending=False).head(1)
        GMV = df.sort_values(by=["volatility"], ascending=True).head(1)
        weights = {}
        for index, x in enumerate(list(MSR['weights'])[0]):
            weights[list(tickers)[index]] = x * weight

        df = pd.DataFrame(index=self.bars[next(iter(self.bars))].index)
        for ticker in params['tickers']:
            df[ticker] = weights[ticker]
        print("Optimum", weights)
        # buy weights
        result = []
        for stock in self.tickers:
#             print(stock)
#             print(self.stocks[stock].buyWeights(weights[stock]))
            strategy = {
                'key': stock,
                'value':self.stocks[stock].buyWeights(weights[stock])
            }
            result.append(strategy)
#             result[stock] = self.stocks[stock].buyWeights(weights[stock])
#             self.stocks[stock].strategies.append(self.stocks[stock].buyWeights(weights[stock])) #= [self.stocks[stock].buyWeights(weights[stock])]
        return result
     
    # returns a dict of the symbol:[a df of the strategy with a Date index]
    def buyEven(self, weight=1):
        result = []
        for stock in self.stocks.keys():
#             result[stock] = self.stocks[stock].buyWeights(weight)
            strategy = {
                'key': stock,
                'value':self.stocks[stock].buyWeights(weight)
            }
            result.append(strategy)
#             self.stocks[stock].strategies.append(self.stocks[stock].buy 

    

class Stock(object):
    def __init__(self, bars, ticker, initial_capital=1000):
        self.bars = bars # df of the Open, High, Low, Close, Adj Close, Volume, with Date as index
        self.ticker = ticker # str: ticket symbol
        self.positions = pd.DataFrame(
            index=self.bars[next(iter(self.bars))].index) # empty df with same index as bars
        self.strategies = [] # list of df's with single column as the "buy signal" with the column name being the ticker
        self.initial_capital = float(initial_capital)

    def generate_positions(self):
#         print("generate positoins called", self.ticker)
#         print(self.strategies)
#         print(type(self.strategies))
        if not self.strategies:
            # just set to zero
            self.positions = pd.DataFrame(
                index=self.bars[next(iter(self.bars))].index)
            self.positions[self.ticker] = 0.0
            self.positions[self.ticker+"_position"] = 0
            self.positions[self.ticker +
                           '_pos_diff'] = self.positions[self.ticker+"_position"].diff()
            return self.positions

        self.positions = pd.concat(self.strategies).groupby(['Date']).sum()

        self.positions[self.ticker+"_position"] = 1*self.positions[self.ticker]
        self.positions[self.ticker +
                       '_pos_diff'] = self.positions[self.ticker+"_position"].diff()
        return self.positions

    # takes your
    def backtest(self):
        if self.ticker+"_position" not in self.positions.columns:
            self.generate_positions()
        holdings_col = []
        holdings_col.append(self.ticker+"_holdings")
        self.positions[self.ticker+'_cash'] = (
            self.positions[self.ticker+'_pos_diff'] * self.bars.Open)
        self.positions[self.ticker+"_holdings"] = self.bars.Open * \
            self.positions[self.ticker+'_position']
        self.positions[self.ticker+'_open'] = self.bars.Open
        self.positions[self.ticker+'_close'] = self.bars.Close
        self.positions['holdings'] = self.positions[holdings_col].sum(axis=1)
        self.positions['cash_diff'] = self.positions[[
            x.replace('_holdings', '_cash') for x in holdings_col]].sum(axis=1)
        self.positions['cash'] = self.initial_capital - \
            self.positions['cash_diff'].cumsum()
        self.positions['total'] = self.positions['cash'] + \
            self.positions['holdings']
        self.positions['returns'] = self.positions['total'].pct_change()
        self.positions['date'] = self.positions.index.strftime("%m/%d/%Y")
        self.positions = self.positions.fillna(0)
        return self.positions
    # takes a dict of weights and returns positions

    def buyWeights(self,weight):
        # assert(weights.keys() in self.bars.keys())
        df = pd.DataFrame(index=self.bars[next(iter(self.bars))].index)
        df[self.ticker] = weight
        df[self.ticker][0:1] = 0.0
        return df

    def macd(self, long=21,short=13):
        df = pd.DataFrame(index=self.bars[next(iter(self.bars))].index)

        df[self.ticker] = 0
        df[self.ticker] = macd(
            df=self.bars, short=short, long=long)
        return df