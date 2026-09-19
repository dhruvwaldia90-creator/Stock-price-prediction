# Configuration for Tata Steel Intraday Machine Learning Live Trader
import os

# Asset Details
SYMBOL = "TATASTEEL.NS"
COMPANY_NAME = "Tata Steel Limited"
EXCHANGE = "NSE"

# Intraday Timeframe Settings
TIMEFRAME = "5m"            # 1m, 5m, or 15m
HISTORICAL_DAYS = 5         # Days of intraday history (~375 5-min bars)
POLL_INTERVAL_SECONDS = 5   # Real-time polling frequency

# Capital & Risk Management
STARTING_CAPITAL = 100000.0 # Virtual / Sandbox capital (INR)
MAX_RISK_PER_TRADE_PCT = 0.01 # 1% risk per trade (₹1,000 max loss)
RISK_REWARD_RATIO = 2.0     # Target 1:2 Risk to Reward
ATR_MULTIPLIER_SL = 1.5     # Stop loss = Entry - (ATR * 1.5)
SLIPPAGE_PCT = 0.0005       # 0.05% slippage simulation
BROKERAGE_PER_ORDER = 20.0  # Standard discount broker intraday fee (INR)

# Technical Indicators
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
ATR_PERIOD = 14
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0

# Machine Learning Settings
N_ESTIMATORS = 100
MAX_DEPTH = 8
TEST_SPLIT_RATIO = 0.20
MODEL_PATH = os.path.join(os.path.dirname(__file__), "tata_steel_intraday_model.joblib")
