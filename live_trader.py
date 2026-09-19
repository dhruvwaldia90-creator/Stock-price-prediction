#!/usr/bin/env python3
"""
Tata Steel Intraday ML Live Trader - Terminal Executable
"""
import sys
import time
import os
from datetime import datetime
from config import SYMBOL, TIMEFRAME, STARTING_CAPITAL, POLL_INTERVAL_SECONDS
from data_feed import LiveMarketFeed
from features import calculate_intraday_features
from model import IntradayMLModel
from intraday_strategy import IntradayStrategy

class LiveTrader:
    def __init__(self):
        self.feed = LiveMarketFeed(SYMBOL)
        self.model = IntradayMLModel()
        self.strategy = IntradayStrategy(capital=STARTING_CAPITAL)
        self.cash = STARTING_CAPITAL
        self.position = None
        self.trade_history = []
        self.df_features = None

    def initialize_system(self):
        print("========================================================================")
        print("   TATA STEEL (NSE) INTRADAY MACHINE LEARNING LIVE TRADING SYSTEM")
        print("========================================================================")
        print(f"[*] Asset: {SYMBOL} | Interval: {TIMEFRAME} | Capital: ₹{self.cash:,.2f}\n")
        print("[1/3] Fetching historical intraday bars...")
        raw_bars = self.feed.fetch_intraday_bars(days=5, interval=TIMEFRAME)
        print(f"      [OK] Ingested {len(raw_bars)} bars")
        print("[2/3] Computing intraday indicators (VWAP, Supertrend, ATR)...")
        self.df_features = calculate_intraday_features(raw_bars)
        print("[3/3] Training Scikit-Learn Walk-Forward Model...")
        metrics = self.model.train_and_validate(self.df_features)
        print(f"      [OK] Metrics: RMSE=₹{metrics['RMSE']} | Hit Ratio={metrics['Hit_Ratio']}%\n")
        print("[✓] INITIALIZATION COMPLETE. STARTING LIVE MARKET LOOP...\n")

    def run(self):
        self.initialize_system()
        tick = 0
        try:
            while True:
                tick += 1
                q = self.feed.fetch_latest_tick()
                cur_price = q['ltp']
                latest_row = self.df_features.iloc[-1].to_dict()
                latest_df = self.df_features.iloc[[-1]].copy()
                latest_df['Close'] = cur_price

                pred_close = self.model.predict_next_candle(latest_df)
                signal = self.strategy.evaluate_signal(cur_price, pred_close, latest_row)

                print(f"[TICK #{tick} | {q['timestamp']}] {q['symbol']} LTP: ₹{cur_price} ({q['change_pct']:+.2f}%) | Status: {q['status']}")
                print(f"  -> ML Predicted 5m Close: ₹{pred_close} | Signal: {signal['action']} ({signal['confidence']}%)")
                print(f"  -> Target: ₹{signal['target']} | Stop-Loss: ₹{signal['stop_loss']} | Shares: {signal['recommended_shares']}")
                time.sleep(POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\n[!] Trader halted by user.")

if __name__ == "__main__":
    trader = LiveTrader()
    trader.run()
