
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
        self.signals = pd.DataFrame(
            index=self.bars[next(iter(self.bars))].index)
        self.positions = pd.DataFrame(
            index=self.bars[next(iter(self.bars))].index)
        self.initial_capital = float(initial_capital)
        self.forecast = {}
        self.lastDate = ''
        self.strategies = []

    @abstractmethod
    def generate_positions(self):
        """Provides the logic to determine how the portfolio 
        positions are allocated on the basis of forecasting
        signals and available cash."""
        raise NotImplementedError("Should implement generate_positions()!")

    @abstractmethod
    def backtest_portfolio(self):
        """Provides the logic to generate the trading orders
        and subsequent equity curve (i.e. growth of total equity),
        as a sum of holdings and cash, and the bar-period returns
        associated with this curve based on the 'positions' DataFrame.

        Produces a portfolio object that can be examined by 
        other classes/functions."""
        raise NotImplementedError("Should implement backtest_portfolio()!")

    # takes a dict of weights and returns positions
    def buyWeights(self, params={}):
        # assert(weights.keys() in self.bars.keys())
        df = pd.DataFrame(index=self.bars[next(iter(self.bars))].index)
        for ticker in params.keys():

            df[ticker] = params[ticker]
            df[ticker][0:1] = 0.0
        return df

    def macd(self, params={}):
        df = pd.DataFrame(index=self.bars[next(iter(self.bars))].index)
        for ticker in params.keys():
            df[ticker] = 0
            print(params)
            df[ticker] = macd(df = self.bars[ticker],short = params[ticker]['short'],long =params[ticker]['long'])
        return df

    def markowitz(self, params={}):
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
        print(MSR)
        weights = {}
        for index, x in enumerate(list(MSR['weights'])[0]):
            print(list(tickers)[index])
            weights[list(tickers)[index]] = x

        df = pd.DataFrame(index=self.bars[next(iter(self.bars))].index)
        for ticker in params['tickers']:
            df[ticker]= weights[ticker]
        print("Optimum")
        print(weights)
        return df