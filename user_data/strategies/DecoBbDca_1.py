import numpy as np
from pandas import DataFrame
from freqtrade.strategy import IStrategy
from datetime import datetime
import scipy as sp
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import talib.abstract as ta
import logging
from functools import reduce
import datetime
import talib.abstract as ta
import pandas_ta as pta
import os
import numpy as np
import pandas as pd
import warnings
import math
import time
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical import qtpylib
from datetime import timedelta, datetime, timezone
from pandas import DataFrame, Series
from technical import qtpylib
from typing import List, Tuple, Optional
from freqtrade.strategy.interface import IStrategy
from technical.pivots_points import pivots_points
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
from typing import Optional, Union
from functools import reduce
import warnings
import math
pd.options.mode.chained_assignment = None
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from scipy.signal import find_peaks, butter, filtfilt

import freqtrade.exchange as exchange

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from collections import deque
import ccxt
from talib import MA_Type

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

#logger = logging.getLogger(__name__)

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler('trade_logfile.log')  # Log file name
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class DecoBbDca_1(IStrategy):

    # this strategy uses only 1 indicator, the bollinger bands
    # the bollinger bands are used to determine the entry points
    # exit is based on a tp level
    # safety orders are used to increase the position size, DCA concept

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".

    leverage_value  = 2
    enable_spyke = False
    use_compounding = True

    minimal_roi = {
        "0": 1 * leverage_value ,
    }

    # Define the stoploss
    stoploss = -0.99

    # Define the timeframe for the strategy
    timeframe = '5m'

    # Trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    # time/ cpu capacity saving
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Optional order type mapping.
    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # allow additional orders, change tp, sl, etc....
    position_adjustment_enable = True

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    
    # setting specific for every asset, based on optimization by backtesting.py, not by freqtrade
    # no sl is used, only tp
    # parameters are optimized for no margin calls and max equity growth
    optimize_settings=   {  "XRP/USDT":  {# result
                                "bo_amount": 0.0971,                  # start stake amount as percentage of wallet
                                "so_amount": 0.2342,                  # safety order amount as percentage of wallet
                                "so_count": 2,                        # number of safety orders, DCA
                                "tp": 0.0097,                         # take profit
                                "so_deviation": 0.0396,               # safety order deviation
                                'so_step_scale': 1.5405,              # safety order step scale for deviation
                                "so_volume_scale":1.9657,             # safety order volume scale 
                                "bb_len": 219,                        # bollinger band length
                                "bb_mul": 1.812,                      # bollinger band multiplier
                        },
                            "DOGE/USDT":  {# result
                                "bo_amount": 0.09708,                 # start stake amount as percentage of wallet
                                "so_amount": 0.2342,                  # safety order amount as percentage of wallet
                                "so_count": 2,                        # number of safety orders
                                "tp": 0.00978,                        # take profit
                                "so_deviation": 0.0396,               # safety order deviation
                                'so_step_scale': 1.5405,              # safety order step scale for deviation
                                "so_volume_scale":1.9657,             # safety order volume scale 
                                "bb_len": 219,                        # bollinger band length
                                "bb_mul": 1.812,                      # bollinger band multiplier    
                        },
                            "XMR/USDT":  {# result
                                "bo_amount": 0.1088,                  # start stake amount as percentage of wallet
                                "so_amount": 0.27376,                 # safety order amount as percentage of wallet
                                "so_count": 2,                        # number of safety orders
                                "tp": 0.0119,                         # take profit
                                "so_deviation": 0.0177,               # safety order deviation
                                'so_step_scale': 1.600,               # safety order step scale for deviation
                                "so_volume_scale":1.503,              # safety order volume scale 
                                "bb_len": 207,                        # bollinger band length
                                "bb_mul": 2.0575,                     # bollinger band multiplier    
                        },
                }

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 300 #optimize_settings['XRP/USDT:USDT']['bb_len']

    # maximal DCA positions to add
    # max_entry_position_adjustment needs to be calculated as max for all assets to be traded 
    # This parameter can not be changed ????? 
    max_entry_position_adjustment = optimize_settings['XRP/USDT']['so_count']

    # calculate start stake value, leave funds open for DCA orders
    # call back is called only once for the first order of a trade, so not for the following DCA orders
    def custom_stake_amount(self, pair: str, current_time: 'datetime', current_rate: float,
                       proposed_stake: float, min_stake: Optional[float], max_stake: float,
                       leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        
        """
        Custom stake amount logic using percentage of configured stake, adjusted for leverage.
        
        Parameters:
        -----------
        pair : str
            Current pair being traded
        proposed_stake : float
            Stake amount proposed by the bot (from config)
        min_stake : float
            Minimum stake amount allowed
        max_stake : float
            Maximum stake amount allowed
        leverage : float
            Current leverage being used
        entry_tag : Optional[str]
            Entry tag for the trade
        side : str
            Trade side (long/short)
            
        Returns:
        --------
        float : Modified stake amount adjusted for leverage
        """
        
        
        # If starting the backtest, return none
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if (len(dataframe) == 0):
            return None
        
        logger.info(f"{current_time}/{pair} - bo at price {current_rate} with stake amount percentage of {self.optimize_settings[pair]['bo_amount']}")

        try:
            # Calculate base stake amount based on percentage
            base_stake_amount = proposed_stake * self.optimize_settings[pair]['bo_amount']
            
            # Adjust stake amount for leverage
            # When using leverage, we reduce the actual stake amount
            # because the position size will be multiplied by leverage
            leveraged_stake = base_stake_amount / leverage
            
            # Calculate the effective position size
            effective_position_size = leveraged_stake * leverage
            
            # Ensure stake amount is within allowed limits
            # Note: We check the effective position size against max_stake
            if max_stake:
                if effective_position_size > max_stake:
                    leveraged_stake = max_stake / leverage
                
            # Check minimum stake if provided
            if min_stake:
                # Convert min_stake to leveraged equivalent
                min_leveraged_stake = min_stake / leverage
                leveraged_stake = max(min_leveraged_stake, leveraged_stake)
            
            # Log the stake calculation with leverage details
            logger.info(
                f"Pair: {pair} - Base stake: {base_stake_amount:.2f} "
                f"({self.optimize_settings[pair]['bo_amount'] * 100.0}% of {proposed_stake:.2f}) - "
                f"Leveraged stake: {leveraged_stake:.2f} (leverage: {leverage}x) - "
                f"Effective position size: {leveraged_stake * leverage:.2f}"
            )
        
            return leveraged_stake
        
        except Exception as e:
            logger.error(f"Error in custom_stake_amount: {str(e)}")
            # Return original proposed stake adjusted for leverage if there's an error
            return proposed_stake / leverage

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float | None | tuple[float | None, str | None]:
        """
        Adjust the stake amount for DCA orders and update the take profit level.
        This call back will be executed as long there are no unfilled orders
        
        Parameters:
        -----------
        pair: str
            Current pair being traded
        trade: Trade
            The active trade object
        current_time: datetime
            Current timestamp
        current_rate: float
            Current market rate
        current_profit: float
            Current profit of the trade as percentage
        
        Returns:
        --------
        float: The adjusted stake amount for the next DCA order
        str: f"SO{count_of_entries}" - The DCA order index
        """

        #------------------------------------------------------------------------ #
        #------------------------------------------------------------------------ #
        # as long there is an open limit order this function will not be called   #
        # after filled or partly filled the function will be called again         #
        #-------------------------------------------------------------------------#
        #------------------------------------------------------------------------ #

        #---------------------------------------------------------------------------------- #
        # market orders have as commission a makers fee                                     #
        # limit orders have as commission a takers fee                                      #
        # makers fee is higher as takers fee, so the profit is higher with limit orders     #
        # se let us for the dca-orders use limit orders                                     #
        #-----------------------------------------------------------------------------------#

        # as reference
        # _stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
        # _max_trades = self.config["max_open_trades"]
        # filled_entries = trade.select_filled_orders(trade.entry_side)
        # limit_orders = [order for order in filled_entries if order.order_type == 'limit']
        # market_orders = [order for order in filled_entries if order.order_type == 'market']
        # trade.open_rate_requested, is the open price of the original trade
        # trade.open_rate, is the average open_price of the trade (basic plus DCA orders)

        pair = trade.pair
       
        try:
            # get the dataframe for the pair
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            current_candle = dataframe.iloc[-1].squeeze()
            #print(f"current_candle: {current_candle}")

            # get the deviation value for first adding of new position
            initial_safety_order_trigger = self.optimize_settings[pair]["so_deviation"]
            #logger.info(f"current_profit: {current_profit} ,initial_safety_order_trigger: {initial_safety_order_trigger}")

            # current_profit metric is a percentage value, so two percentages are compared
            if current_profit >= (-1 * abs(initial_safety_order_trigger)):
                # no need for dca trade / order
                return None
            
            # current price is depending on enable_spike parameter is true, then wick values are used, else use close value
            current_price: float = float(current_candle['close']) if float(self.enable_spyke) == False else \
                float(current_candle['high']) if trade.is_short else float(current_candle['low'])
            
            # get list of orders within trade
            filled_entries = trade.select_filled_orders(trade.entry_side)

            # get the amount of entries in the trade, filled and not filled
            count_of_entries = trade.nr_of_successful_entries
            #logger.info(f"count_of_entries: {count_of_entries}")

            if 1 <= count_of_entries <= self.optimize_settings[pair]["so_count"]:

                if (current_time - timedelta(minutes=timeframe_to_minutes(self.timeframe))) >= filled_entries[(count_of_entries - 1)].order_filled_date.replace(tzinfo=timezone.utc):

                    # safety_order_trigger is the offset from base order
                    if self.use_compounding:
                        origin_stake_amount = 0
                        if count_of_entries == 1:
                            #print("before origin_stake_amount")
                            origin_stake_amount = trade.stake_amount / self.optimize_settings[pair]["bo_amount"] * trade.leverage
                            #print("after origin_stake_amount")
                            logger.info(f"origin_stake_amount: {origin_stake_amount}")
                            # save the trade amount at opening the first order of the trade
                            trade.set_custom_data("origin_stake_amount", origin_stake_amount)
                            #logger.info(f"origin_stake_amount: {origin_stake_amount}")
                        else:
                            #print("before origin_stake_amount")
                            origin_stake_amount = trade.get_custom_data("origin_stake_amount")
                            #print("after origin_stake_amount")
                        so_amount = origin_stake_amount * self.optimize_settings[pair]["so_amount"] / trade.leverage
                        logger.info(f"so_amount: {so_amount}")
                    else:
                        so_amount = self.optimize_settings[pair]["so_amount"] / trade.leverage

                    # so_amount = self.optimize_settings[pair]["so_amount"] / trade.leverage
                    safety_order_volume_scale = self.optimize_settings[pair]["so_volume_scale"]
                    #print("before origin_stake_amount")
                    # calculate the trigger price for the next safety order, entry price
                    safety_order_trigger_price = recalculate_safety_order_price(count_of_entries, self.optimize_settings[pair]["so_step_scale"], \
                                                                    self.optimize_settings[pair]["so_deviation"], trade.open_rate_requested, "short" if trade.is_short else "long")
                
                    #print("before origin_stake_amount")
                    logger.info(f"current_price: {current_price}, safety_order_trigger_price: {safety_order_trigger_price}")
                    if (current_price >= safety_order_trigger_price and trade.is_short) or (current_price <= safety_order_trigger_price and not trade.is_short):
                        try:
                            # This then calculates current safety order size
                            #print("before origin_stake_amount")
                            stake_amount = so_amount * math.pow(safety_order_volume_scale,(count_of_entries - 1)) #/ self._leverage
                            amount = stake_amount / current_price
                            #print("before origin_stake_amount")
                            #safety_order_price = trade.open_rate_requested * (1 - safety_order_trigger if not trade.is_short else 1 - safety_order_trigger)
                            print("before origin_stake_amount")
                            """ logger.info(f"{current_time}/{trade.pair} - so {count_of_entries}/{self.optimize_settings[pair]['so_count']} \
                                        at price {current_price} so_level {round(safety_order_trigger_price, 5)} \
                                        with stake amount of {stake_amount * trade.leverage} {trade.stake_currency} \
                                        - liq:{round(trade.liquidation_price, 5)} - {round(amount * trade.leverage, 4)} {trade.base_currency}") """
                            print("after origin_stake_amount")
                            return stake_amount, f"SO{count_of_entries}"
                        except Exception as exception:
                            logger.info(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}') 
                            return None

        except Exception as e:
            logger.error(f"Error in adjust_trade_amount: {str(e)}")
            # Return the original proposed stake amount if there's an error
            return self.optimize_settings[pair]["so_amount"] / trade.leverage

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> bool:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        # consider spyke = False
        # current price is depending on enable_spike parameter is true, then wick values are used, else use close value
        # current_price: float = float(current_candle['close']) if float(self.enable_spyke) == False else \
            #float(current_candle['high']) if trade.is_short else float(current_candle['low'])

        current_price = current_rate

        if trade.is_short and current_price <= trade.open_rate * (1 - self.optimize_settings[pair]['tp']):
            return 'tp-short'
        elif not trade.is_short and current_price >= trade.open_rate * (1 + self.optimize_settings[pair]['tp']):
            return 'tp-long'
        
        if (current_profit > self.optimize_settings[pair]['tp']):
            return True
        
        return False

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.leverage_value 

    # function below not used for the time being
    def adjust_entry_price(self, pair: str, trade: Trade, order_type: str, 
                          proposed_rate: float, current_time: datetime, 
                          current_rate: float, **kwargs) -> Union[float, Tuple[float, float]]:
        """
        Adjust the entry price and stake amount for DCA orders.
        
        Returns either:
        - New entry price (float) - stake amount will be the default stake amount
        - Tuple (entry_price, stake_amount) to specify both values
        """

        
        try:
            # Get number of existing orders
            filled_entries = trade.select_filled_orders(trade.entry_side)

            # safe the basic amount used in the first trade
            if (len(filled_entries) ==  1):
                bp = 1
            
            dca_order_count = len(filled_entries) - 1  # Subtract initial entry
            
            # Check if we should place another DCA order
            if dca_order_count >= self.dca_levels.value:
                return None
                
            # Calculate DCA target price
            # Each subsequent order will be placed lower than the previous one
            avg_entry_price = trade.open_rate
            dca_target_price = avg_entry_price * (self.dca_price_multiplier.value ** (dca_order_count + 1))
            
            # Calculate DCA order size
            initial_stake = self.config['stake_amount']
            dca_stake = initial_stake * (self.dca_stake_multiplier.value ** dca_order_count)
            
            # Ensure we don't exceed max allowed stake
            max_stake = self.config['max_stake_amount']
            current_total_stake = sum(order.cost for order in filled_entries)
            remaining_stake = max_stake - current_total_stake
            
            if dca_stake > remaining_stake:
                dca_stake = remaining_stake
                
            if dca_stake <= 0:
                return None
            
            # Calculate new take profit level for the potential new position size
            new_trade_size = current_total_stake + dca_stake
            new_take_profit = self.calculate_take_profit(pair, new_trade_size, current_rate)
            
            # Log DCA order details
            logger.info(f"""
                Pair: {pair}
                DCA Order #{dca_order_count + 1}
                Current Avg Entry: {avg_entry_price:.8f}
                DCA Target Price: {dca_target_price:.8f}
                DCA Stake Amount: {dca_stake:.4f}
                New Take Profit: {new_take_profit:.8f}
            """)
            
            # Return both the target price and stake amount
            return (dca_target_price, dca_stake)
            
        except Exception as e:
            logger.error(f"Error in adjust_entry_price: {str(e)}")
            return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, 
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        Additional confirmation for DCA orders.
        """

        return True
    
        try:
            if order_type == 'limit':
                # Get current trade
                trade = self.get_trade(pair)
                
                if not trade:
                    return True  # Initial entry
                
                # Check if this would exceed our maximum DCA orders
                if trade.nr_of_successful_entries >= self.dca_levels.value:
                    return False
                
                # Additional validations for DCA orders
                filled_entries = trade.select_filled_orders(trade.entry_side)
                current_total_stake = sum(order.cost for order in filled_entries)
                new_stake = amount * rate
                
                # Ensure we don't exceed maximum stake
                if current_total_stake + new_stake > self.config['max_stake_amount']:
                    return False
                
                # Validate the price level
                last_entry_price = filled_entries[-1].price
                if side == 'long' and rate >= last_entry_price:
                    return False  # DCA price must be lower for longs
                if side == 'short' and rate <= last_entry_price:
                    return False  # DCA price must be higher for shorts
                
            return True
            
        except Exception as e:
            logger.error(f"Error in confirm_trade_entry: {str(e)}")
            return False

    # not used for the time being
    def calculate_take_profit(self, pair: str, trade_size: float, current_rate: float) -> float:
        
        """
        Calculate the take profit price based on trade size and current market conditions.
        The larger the position, the smaller the take profit percentage to ensure profitability.
        
        Parameters:
        -----------
        pair : str
            Trading pair
        trade_size : float
            Current total trade size (including DCAs)
        current_rate : float
            Current market rate
        
        Returns:
        --------
        float : New take profit price
        """

        
        try:
            # Get the initial take profit percentage from configuration
            base_take_profit_pct = self.base_take_profit_percent.value  # e.g., 1.5%
            
            # Get the original proposed stake from configuration
            initial_stake = self.config['stake_amount']
            
            # Calculate dynamic take profit percentage based on trade size
            # As trade size increases (more DCAs), we reduce the take profit percentage
            # to ensure we can exit the position profitably
            position_scale = trade_size / initial_stake
            
            # Adjust take profit percentage based on position scale
            # Example formula: For each doubling of position size, reduce TP by 20%
            adjusted_tp_pct = base_take_profit_pct * (0.8 ** (math.log2(position_scale)))
            
            # Ensure minimum take profit percentage
            min_tp_pct = self.min_take_profit_percent.value  # e.g., 0.5%
            adjusted_tp_pct = max(adjusted_tp_pct, min_tp_pct)
            
            # Calculate the average entry price for the position
            trade = self.get_trade(pair)
            if trade:
                filled_entries = trade.select_filled_orders(trade.entry_side)
                total_cost = sum(order.cost for order in filled_entries)
                total_amount = sum(order.amount for order in filled_entries)
                avg_entry_price = total_cost / total_amount
            else:
                avg_entry_price = current_rate
            
            # Calculate take profit price
            if trade and trade.is_short:
                # For short positions, take profit is below entry
                take_profit = avg_entry_price * (1 - adjusted_tp_pct / 100)
            else:
                # For long positions, take profit is above entry
                take_profit = avg_entry_price * (1 + adjusted_tp_pct / 100)
            
            # Log the calculation
            logger.info(f"""
                Pair: {pair}
                Trade Size: {trade_size:.2f}
                Position Scale: {position_scale:.2f}x
                Base TP: {base_take_profit_pct:.2f}%
                Adjusted TP: {adjusted_tp_pct:.2f}%
                Avg Entry: {avg_entry_price:.8f}
                New TP Price: {take_profit:.8f}
            """)
            
            return take_profit
        
        except Exception as e:
            logger.error(f"Error in calculate_take_profit: {str(e)}")
            # Fallback to a default take profit calculation
            return current_rate * (1 + (self.base_take_profit_percent.value / 100))  
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        pair = metadata["pair"]
        #print(pair)

        if pair not in self.optimize_settings:
            self.optimize_settings[pair] = ['']

        [upperband, middleband, lowerband] = ta.BBANDS(dataframe["close"], self.optimize_settings[pair]["bb_len"], self.optimize_settings[pair]["bb_mul"], \
                                                       self.optimize_settings[pair]["bb_mul"], matype=MA_Type.EMA)
        dataframe["bb_upperband"] = upperband
        dataframe["bb_middleband"] = middleband
        dataframe["bb_lowerband"] = lowerband
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # check for conditions

        dataframe.loc[
            (
                ((dataframe['close'] >= dataframe["bb_lowerband"]) & (dataframe['close'].shift() < dataframe["bb_lowerband"]))
            ),
            ['enter_long', 'enter_tag']] = (1, 'cross lower band')
        
        dataframe.loc[
            (
                ((dataframe['close'] <= dataframe["bb_upperband"]) & (dataframe['close'].shift()  > dataframe["bb_upperband"]))
            ),
            ['enter_short', 'enter_tag']] = (1, 'cross upper band')
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
    
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_upperband'])
            ),
            ["exit_long", "exit_tag"], ] = (1, "cross opposite band")
        

        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_lowerband'])
            ),
            ["exit_short", "exit_tag"], ] = (1, "cross opposite band")
        
        return dataframe
    
    def custom_exit_price( self, pair: str, trade: Trade, current_time: datetime, proposed_rate: float, current_profit: float,  exit_tag: str | None, **kwargs,  ) -> float:
        
        """
        Custom exit price calculation for the strategy.
        """
        
        average_price = trade.open_rate
        if trade.is_short:
            return average_price * (1 - self.optimize_settings[pair]['tp'])
        else:
            return average_price * (1 + self.optimize_settings[pair]['tp'])
        

def recalculate_safety_order_price(count_of_entries, so_step_scale, so_deviation, original_open_price, direction):

    # calculate the trigger price for the safety order
    safety_order_trigger = so_deviation * math.pow(so_step_scale, count_of_entries - 1)
    safety_order_trigger_price = original_open_price * (1 - safety_order_trigger if direction == "short" else 1 + safety_order_trigger)
     
    # 
    return safety_order_trigger_price