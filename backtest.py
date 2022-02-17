
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

    def __init__(self, bars, initial_capital=1000):
        self.tickers = list(bars.keys())
        self.bars = bars
        self.stocks = []  # List of Class Stock[]
        self.backtest = {}
        self.positions = {}
        self.strategies = []
        self.initial_capital = initial_capital

    def backtest_portfolio(self):
        if not self.strategies:
            for stock in self.stocks:
                stock.generate_positions()

        self.positions = pd.concat(
            [x.positions for x in self.stocks]).groupby(['Date']).sum()

        # self.positions[self.ticker+"_position"] = 1*self.positions[self.ticker]
        # self.positions[self.ticker +
        #                '_pos_diff'] = self.positions[self.ticker+"_position"].diff()
        return self.positions

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
            df = df.append(row, ignore_index=True)
        df["sharpe"] = (df["return"] - risk_free) / df["volatility"]

        MSR = df.sort_values(by=["sharpe"], ascending=False).head(1)
        GMV = df.sort_values(by=["volatility"], ascending=True).head(1)
        weights = {}
        for index, x in enumerate(list(MSR['weights'])[0]):
            weights[list(tickers)[index]] = x * weight

        df = pd.DataFrame(index=self.bars[next(iter(self.bars))].index)
        for ticker in params['tickers']:
            df[ticker] = weights[ticker]
        print("Optimum")
        print(weights)

        # return df

        # buy weights
        for stock in self.stocks:
            params = {
                stock.ticker: weights[stock.ticker]
            }
            stock.strategies = [stock.buyWeights(params)]
            stock.generate_positions()
            stock.backtest()
        return self.stocks


class Stock(object):
    def __init__(self, bars, ticker, initial_capital=1000):
        self.bars = bars
        self.ticker = ticker
        self.positions = pd.DataFrame(
            index=self.bars[next(iter(self.bars))].index)
        self.strategies = []
        self.initial_capital = float(initial_capital)

    def generate_positions(self):
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

    def buyWeights(self, params={}):
        # assert(weights.keys() in self.bars.keys())
        df = pd.DataFrame(index=self.bars[next(iter(self.bars))].index)
        df[self.ticker] = params[self.ticker]
        df[self.ticker][0:1] = 0.0
        return df

    def macd(self, params={}):
        df = pd.DataFrame(index=self.bars[next(iter(self.bars))].index)

        df[self.ticker] = 0
        df[self.ticker] = macd(
            df=self.bars, short=params[self.ticker]['short'], long=params[self.ticker]['long'])
        return df
