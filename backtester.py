import numpy as np
import pandas as pd
#import quandl   # Necessary for obtaining financial data easily
import json
import time
from yahoofinancials import YahooFinancials as YF
import time
import pandas as pd
# from pandas_datareader import data
import os
import pickle
from tqdm import tqdm
import scipy.stats as sp
from backtest import Strategy, Portfolio, Stock
import os
#from fbprophet import Prophet
from utils import *


class RandomStrategy(Strategy):
    """Derives from Strategy to produce a set of signals that
    are randomly generated long/shorts. Clearly a nonsensical
    strategy, but perfectly acceptable for demonstrating the
    backtesting infrastructure!"""

    def __init__(self, bars, strategies):
        """Requires the symbol ticker and the pandas DataFrame of bars"""
        self.tickers = list(bars.keys())
        self.bars = bars
        self.strategies = strategies

    def genSignals(self):
        # uses the first stock in stocks to set the index for the signals df
        signals = pd.DataFrame(index=self.bars[next(iter(self.bars))].index)
        # for stock in list(self.bars):
        #     signals[stock] = self.strategy(self.bars[stock])
        #     #signals[stock] = 1 #np.sign(np.random.randn(len(signals)))
        #     signals[stock][0:5] = 0.0
        for strat in self.strategies:

            
            if (strat['name'] == 'MACDStrat'):
                print('MACDStrat')
                signals = self.MACDStrat(
                    signals, strat['params']['short'], strat['params']['long'])
            elif (strat['name'] == 'single'):
                print('single')
                signals = self.singleStockStrat(
                    signals, strat['params']['ticker'])
            elif (strat['name'] == 'weighted'):
                print(strat['params']['weights'])
                signals = self.weightedStrat(
                    signals, strat['params']['weights'])
        return signals

    def MACDStrat(self, signals, short, long):
        self.bars = addMACD(self.bars, short, long)
        for stock in list(self.bars):
            signal = np.sign(self.bars[stock]['macd_signal'])
            signals[stock] = signal
        return signals

    def singleStockStrat(self, signals, ticker):
        stocks = self.bars
        for stock in list(stocks):
            if (stock == ticker):
                signals[stock] = 1
            else:
                signals[stock] = 0
            # signals[stock] = 1 #np.sign(np.random.randn(len(signals)))
            signals[stock][0:1] = 0.0
        return signals

    def weightedStrat(self, signals, weights):
        # for i in range(0, len(list(self.bars.keys()))):
        #     symbol = list(self.bars.keys())[i]
        #     signals[symbol] = weights[i]
        #     signals[symbol][0:1] = 0.0
        print(weights)
        for stock, weight in weights.items():
            signals[stock] = weight
            signals[stock][0:1] = 0.0
        return signals


class MyPortfolio(Portfolio):
    """Inherits Portfolio to create a system that purchases 100 units of 
    a particular symbol upon a long/short signal, assuming the market 
    open price of a bar.

    In addition, there are zero transaction costs and cash can be immediately 
    borrowed for shorting (no margin posting or interest requirements). 

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, 0, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio."""

    def __init__(self, bars, initial_capital=1000):
        self.tickers = list(bars.keys())
        self.bars = bars
        # signals is a list of df's of signals
        self.signals = []
        # positions are either an avg or sum of the signals df with positions and pos_diff
        self.positions = pd.DataFrame(index=self.bars[next(iter(self.bars))].index)
        self.initial_capital = float(initial_capital)
        self.forecast = {}
        self.lastDate = ''
        self.strategies = []

    def setStrats(self, strategies):
        self.strategies = strategies
    
    # def genSignals(self, signals_list):
    #     for strat, params in self.strategies:
    #         df = strat(params)
    #         self.signals.append(df)


    def generate_positions(self):
        self.positions = pd.concat(self.strategies).groupby(['Date']).sum()
        for stock in list(self.tickers):
            if stock in self.positions.columns:
                self.positions[stock+"_position"] = 1*self.positions[stock]
                self.positions[stock +'_pos_diff'] = self.positions[stock+"_position"].diff()
        return self.positions


    def backtest_portfolio(self):
        holdings_col = []
        for stock in list(self.tickers):
            if stock in self.positions.columns:
                #positions[stock+"_price"] = stocks[stock].Open
                holdings_col.append(stock+"_holdings")
                self.positions[stock+'_cash'] = (
                    self.positions[stock+'_pos_diff'] * self.bars[stock].Open)
                self.positions[stock+"_holdings"] = self.bars[stock].Open * \
                    self.positions[stock+'_position']
                self.positions[stock+'_open'] = self.bars[stock].Open
                self.positions[stock+'_close'] = self.bars[stock].Close

        # get total holdings, and cash flow
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

    def backtest_forecast(self):
        holdings_col = []
        upper_holdings = []
        lower_holdings = []
        for stock in list(self.tickers):
            #positions[stock+"_price"] = stocks[stock].Open
            holdings_col.append(stock+"_holdings")
            self.positions[stock+'_cash'] = (
                self.positions[stock+'_pos_diff'] * self.bars[stock]['yhat'])
            self.positions[stock+"_holdings"] = self.bars[stock]['yhat'] * \
                self.positions[stock+'_position']
            
            # self.positions[stock+'_open'] = self.bars[stock]['yhat_upper']
            # self.positions[stock+'_close'] = self.bars[stock]['yhat_upper']

            # upper
            col_name = stock + '_yhat_upper'
            upper_holdings.append(col_name+"_holdings")
            self.positions[col_name+'_cash'] = (
                self.positions[stock+'_pos_diff'] * self.bars[stock]['yhat_upper'])
            self.positions[col_name+"_holdings"] = self.bars[stock]['yhat_upper'] * \
                self.positions[stock+'_position']
            self.positions[col_name] = self.bars[stock]['yhat_upper']

            # lower
            col_name = stock + '_yhat_lower'
            lower_holdings.append(col_name+"_holdings")
            self.positions[col_name+'_cash'] = (
                self.positions[stock+'_pos_diff'] * self.bars[stock]['yhat_lower'])
            self.positions[col_name+"_holdings"] = self.bars[stock]['yhat_lower'] * \
                self.positions[stock+'_position']
            self.positions[col_name] = self.bars[stock]['yhat_lower']

        # get total holdings, and cash flow
        self.positions['forecasted_holdings'] = self.positions[holdings_col].sum(axis=1)
        self.positions['forecasted_cash_diff'] = self.positions[[x.replace('_holdings', '_cash') for x in holdings_col]].sum(axis=1)
        self.positions['forecasted_cash'] = self.initial_capital - \
            self.positions['forecasted_cash_diff'].cumsum()
        self.positions['forecasted_total'] = self.positions['forecasted_cash'] + \
            self.positions['forecasted_holdings']
        self.positions['forecasted_returns'] = self.positions['forecasted_total'].pct_change()
        self.positions['date'] = self.positions.index.strftime("%m/%d/%Y")

        # upper
        self.positions['upper_holdings'] = self.positions[upper_holdings].sum(
            axis=1)
        self.positions['cash_diff'] = self.positions[[
            x.replace('_holdings', '_cash') for x in upper_holdings]].sum(axis=1)
        self.positions['upper_cash'] = self.initial_capital - \
            self.positions['cash_diff'].cumsum()
        self.positions['upper_total'] = self.positions['upper_cash'] + \
            self.positions['upper_holdings']
        self.positions['upper_returns'] = self.positions['upper_total'].pct_change()

        # lower
        self.positions['lower_holdings'] = self.positions[lower_holdings].sum(
            axis=1)
        self.positions['cash_diff'] = self.positions[[
            x.replace('_holdings', '_cash') for x in lower_holdings]].sum(axis=1)
        self.positions['lower_cash'] = self.initial_capital - \
            self.positions['cash_diff'].cumsum()
        self.positions['lower_total'] = self.positions['lower_cash'] + \
            self.positions['lower_holdings']
        self.positions['lower_returns'] = self.positions['lower_total'].pct_change()

        self.positions = self.positions.fillna(0)

        return self.positions

    def forecast_portfolio(self):
        col_name = 'Open'
        for stock in list(self.bars):
            df = self.prophetDf(self.bars[stock], col_name)
            prophet, forecast = self.createForecast(df)
            self.bars[stock] = self.bars[stock].rename(columns={
                'y': col_name,
                'ds':'Date'
            })
            # forecast = forecast.rename(columns={
            #     'yhat':'Open',

            # })

            # forecast = forecast[cols_wanted][forecast.index > '2021-03-12'].fillna(0)
            # self.forecast[stock] = pd.concat([self.bars[stock],forecast], axis=1)

            #self.forecast[stock] = self.forecast[stock].fillna(0)
            cols_wanted = ['yhat', 'yhat_upper', 'yhat_lower']
            last_date = self.bars[stock].index[-2]
            self.forecast[stock] = forecast[cols_wanted][forecast.index > last_date].fillna(0)
        return self.forecast

    def createForecast(self, df):
        corona_prophet = Prophet(
            changepoint_prior_scale=0.15, weekly_seasonality=False, yearly_seasonality=False)
        corona_prophet.add_seasonality(
            'self_define_cycle', period=8, fourier_order=8, mode='additive')
        corona_prophet.fit(df)
        # Make a future dataframe for 6 months
        corona_forecast = corona_prophet.make_future_dataframe(
            periods=20, freq='D')
        # Make predictions
        corona_forecast = corona_prophet.predict(corona_forecast)
        corona_forecast.index = corona_forecast['ds']
        corona_forecast.drop(['ds'], axis=1, inplace=True)
        #corona_forecast = corona_forecast.rename(columns={'yhat':'Open'})
        return corona_prophet, corona_forecast

    def prophetDf(self, closeDf, col_name):
        closeDf.rename(columns={"Date": "ds", col_name: "y"}, inplace=True)
        closeDf['ds'] = closeDf.index
        return closeDf
