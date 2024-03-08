from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import sys
import math
import datetime
import backtrader as bt
import backtrader.feeds as btfeeds

from backtrader.feeds import GenericCSVData

class PositionSizeSizer(bt.Sizer):
    params = (
        ("size", 1),  # Default size
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        print('get sizing called')
        print(data.position[0])
        print(data.position[-1])
        # Use the current position size as the order size
        # position_size = data.getposition÷().size
        pos_diff = data.position[0] - data.position[-1]
        print(pos_diff)
        pos_diff = abs(round(pos_diff, 2))
        return self.params.size * pos_diff

class GenericCSV_Position(GenericCSVData):

    # Add a 'pe' line to the inherited ones from the base class
    lines = ('position',)

    # openinterest in GenericCSVData has index 7 ... add 1
    # add the parameter to the parameters inherited from the base class
    params = (('position', 7),)

# %matplotlib inline
# %matplot%lib widget
# %matplotlib notebook
# %matplotlib qt
# %matplotlib tk
# Create a Stratey
    
showLogs = True
class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        self.bar_counter = 0
        self.order = None
        self.position = self.datas[0].position
        if (showLogs):
            self.log(self.datas[0].position)


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    # def next(self):
    #     # Simply log the closing price of the series from the reference
    #     self.log('Close, %.2f' % self.dataclose[0])
    #     self.log('Prev Close, %.2f' % self.dataclose[1])
    def next(self):
        self.bar_counter += 1
        if self.bar_counter == 1:
            return
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return
        
        # self.log(self.position[0])
        # Set the sizer stake from the params
   
        if showLogs:
            self.log('Position, %.2f' % self.position[0])

        pos_diff = self.position[0] - self.position[-1]
        pos_diff = round(pos_diff, 2)
        if pos_diff > 0:
            if showLogs:
                self.log('BUY CREATE, %.2f' % pos_diff)
            # default stake of 1 and executed at markt
            # self.sizer.setsizing(pos_diff)
            self.order = self.buy()
        elif (pos_diff < 0):
            if showLogs:
                self.log('SELL CREATE, %.2f' % pos_diff)
            # self.sizer.setsizing(abs(pos_diff))
            self.order = self.sell()


def run_backtest(path="../datasets/stock_csvs/T.csv", fromdate=(2023, 1, 3), todate=(2023, 12, 26), debug=False):
    print(fromdate)
    print(todate)
    if (debug):
        print("Debug set to True!!!!")
        print(f"path is {path}" )
        print(f"Dates are from {fromdate} to {todate}")
        showLogs = debug

    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    data = GenericCSV_Position(
        dataname=path,
        fromdate=datetime.datetime(*fromdate),
        todate=datetime.datetime(*todate),
        nullvalue=0.0,
        dtformat=('%Y-%m-%d'),
        datetime=0,
        high=2,
        low=3,
        open=1,
        close=4,
        volume=5,
        openinterest=-1,
        position=7
    )

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(1000.0)

    cerebro.addsizer(PositionSizeSizer)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.plot()
        
if __name__ == '__main__':
    print("FUNCTION CALLED")
    run_backtest()