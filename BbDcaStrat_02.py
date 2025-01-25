# 
# this script takes care of the optimalization of the strategy parameters.
#

import pandas as pd
import numpy as np
from datetime import datetime, timedelta 
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import itertools
from utils import apply_dynamic_safety_orders
import talib
import math

from talib import MA_Type
import backtesting.lib as lib

fileName = './data/futures/DOGE_USDT_5m.csv'

# Load your data
df = pd.read_csv(fileName, parse_dates=[0], index_col=0)
data = df


class BollingerDCAStrategy(Strategy):
    
    """ # Define the strategy parameters with default values
    bb_len = 210 # Bollinger Bands period
    bb_mul = 1.62 # Number of standard deviations
    bo_amount= 0.055  # Size of initial order from cash / wallet
    so_count = 2  # Maximum number of safety orders
    #safety_orders_count = 1
    so_amount= 0.1783 # Size of first safety order from cash / wallet
    so_volume_scale = 1.54  # Multiplier for subsequent safety order sizes, volume
    so_price_deviation = 0.025 # Initial price deviation for first safety order (2%)
    so_step_scale= 1.0  # Multiplier for subsequent deviations, price level
    #max_active_positions = 1  # Maximum number of concurrent positions
    tp = 0.0115 # Take profit percentage

    cash = 250

    LONG = "long"
    SHORT = "short"
    #max_days_in_trading = 30000
    so_order_policy = "place_at_start"
    so_enabled = False """

    # Define the strategy parameters with default values
    bb_len = 194 # Bollinger Bands period
    bb_mul = 1.78 # Number of standard deviations
    bo_amount= 0.076  # Size of initial order from cash / wallet
    so_count = 2  # Maximum number of safety orders
    so_amount= 0.273 # Size of first safety order from cash / wallet
    so_volume_scale = 1.79  # Multiplier for subsequent safety order sizes, volume
    so_price_deviation = 0.0166 # Initial price deviation for first safety order (2%)
    so_step_scale= 1.69  # Multiplier for subsequent deviations, price level
    tp = 0.0087 # Take profit percentage

    cash = 250

    LONG = "long"
    SHORT = "short"
    #max_days_in_trading = 30000
    so_order_policy = "place_at_start"
    so_enabled = False


    def init(self):
        
        super().init()

        # Calculate Bollinger Bands
        self.bb_up, self.bb_mid, self.bb_low = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_len, nbdevup=self.bb_mul, nbdevdn=self.bb_mul, matype=MA_Type.EMA)

        #self.bb_up, self.bb_low =  self.I(self.bollinger_bands, self.data.Close, self.bb_len, self.bb_mul)
        #self.bb = self.I(self.bollinger_bands, self.data.Close, self.n_bb, self.n_std)

        # set take profit level
        self.tp_level = self.I(lambda: np.repeat(np.nan, len(self.data)), name = 'TP level', overlay = True)
        self.tp_level.flags.writeable = True

        # set abort level
        #self.trade_abort = self.I(lambda: np.repeat(np.nan, len(self.data)), name = 'TA', scatter = True)
        self.trade_abort = self.I(lambda: np.repeat(np.nan, len(self.data)), name = 'TA', circle = True)
        self.trade_abort.flags.writeable = True
        
        # Initialize position tracking
        #self.active_positions = []  # List to track active positions
        #self.safety_orders = {}  # Dictionary to track safety orders for each position

    def safety_order_deviation(self, index):                                                # new
        
        _deviation = self.so_price_deviation * math.pow(self.so_step_scale, index-1)
        return _deviation
    
    def safety_order_price(self, index, last_safety_order_price, side):                     # new
        
        qty = self.safety_order_qty(index)
        _safety = 0
        if side == 'long':
            _safety = last_safety_order_price * (1 - self.safety_order_deviation(index))
            return _safety, qty
        else:
            _safety = last_safety_order_price * (1 + self.safety_order_deviation(index))
            return _safety, qty
        
    def safety_order_qty(self, index):                                                      # new
            
        _amount = self.so_amount * self.cash * math.pow(self.so_volume_scale, index-1)
        return _amount
    
    def place_safety_order(self, bo_level, side):                                   # new
        
        # init size to place safety order
        tot_order_size = self.bo_amount * self.cash

        last_so_level = bo_level
        for i in range(1, self.so_count + 1):

            so_level, amount = self.safety_order_price(i, last_so_level, side)
            average_price = (tot_order_size * last_so_level + amount * so_level) / (tot_order_size + amount)
            tot_order_size += amount
            last_so_level = so_level

            if side == self.LONG:
                so_tp_level = average_price * (1 + self.tp)
                self.buy(tp = so_tp_level, size=round(amount / so_level), limit=so_level, tag=f"safety_{i}")
                bp = 1
            else:
                so_tp_level = average_price * (1 - self.tp)
                self.sell(tp = so_tp_level, size=round(amount / so_level), limit=so_level, tag=f"safety_{i}")
                bp = 1

    def next(self):
        
        super().next()

        # check for open position
        if self.position:
            if self.so_order_policy == "place_at_start":
                if self.trades[-1].entry_bar == len(self.data) - 1:
                    # adjust tp of previous trade
                    new_tp = self.trades[-1].tp
                    for t1 in range(0, len(self.trades)-1):
                        self.trades[t1].tp = new_tp
                        # update tp level on chart
            self.tp_level[-1] = self.trades[0].tp
        else: # no open position
            self.so_enabled = False
            for order in self.orders:
                order.cancel()

            # check for conditions
            long_condition = self.data.Close[-1] >= self.bb_low[-1] and self.data.Close[-2] < self.bb_low[-1]
            #short_condition = self.data.Close[-1] <= self.bb_up[-1] and self.data.Close[-2] > self.bb_up[-1]
            be = self.data.Close[-1] 

            # open a trade
            if long_condition and not self.position:
                tp_level = self.data.Close[-1] * (1 + self.tp)
                _size = round((self.bo_amount * self.cash) / self.data.Close[-1])
                self.buy(tp = tp_level, size = _size)
                if self.so_order_policy == "place_at_start":
                    self.place_safety_order(be, "long")
            """ elif short_condition and not self.position:
                tp_level = self.data.Close[-1] * (1 - self.tp)
                _size = round((self.bo_amount * self.cash) / self.data.Close[-1])
                self.sell(tp = tp_level, size = _size)
                if self.so_order_policy == "place_at_start":
                    self.place_safety_order(be, "short") """

bt = Backtest(data, BollingerDCAStrategy, cash=250, margin = 1.0, commission=0.003)

stats = bt.run()

print(stats)

bt.plot(show_legend=False, resample=False, plot_pl=True, relative_equity=False)


#bt = Backtest(data, BollingerDCAStrategy, cash=600, margin = 0.2, commission=0.003)

stats, heatmap, return_optimization = bt.optimize(
    bb_len=[190, 325],
    bb_mul=[1.5, 2.5],
    bo_amount=[0.04, 0.12],
    so_count=[2,4],
    so_amount=[0.06, 0.28],
    so_price_deviation=[0.015, 0.04],
    so_volume_scale=[1.0, 1.8],
    so_step_scale=[1.0, 2.0],
    tp=[0.005, 0.012],
    constraint=lambda x: (x.bo_amount < x.so_amount * 2 / 3) and x.bo_amount + sum([x.so_amount * pow(x.so_volume_scale, (a-1)) for a in range(1, x.so_count + 1)]) < 1.0 ,
    maximize=lambda x: x['Equity Final [$]'] / ( 1 + x['margin calls']), # and x['Max. Trade Duration'] < timedelta(days=30),
    method='skopt',
    max_tries=500,
    random_state=0,
    return_heatmap=True,
    return_optimization=True
)

print(stats)
print('----------')
print(heatmap.sort_values(ascending=False))
print('----------')
#print(return_optimization)
