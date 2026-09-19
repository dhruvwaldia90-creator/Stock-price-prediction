"""
Intraday Strategy & Risk Management Module
"""
from config import STARTING_CAPITAL, MAX_RISK_PER_TRADE_PCT, RISK_REWARD_RATIO, ATR_MULTIPLIER_SL

class IntradayStrategy:
    def __init__(self, capital: float = STARTING_CAPITAL):
        self.capital = capital
        self.max_risk_amount = capital * MAX_RISK_PER_TRADE_PCT

    def evaluate_signal(self, current_price: float, predicted_next_close: float, latest_row: dict) -> dict:
        expected_move = predicted_next_close - current_price
        expected_pct = (expected_move / current_price) * 100
        vwap = latest_row.get('VWAP', current_price)
        supertrend_dir = latest_row.get('Supertrend_Direction', 0)
        rsi = latest_row.get('RSI', 50)
        atr = latest_row.get('ATR', 0.85)

        action = "HOLD"
        confidence = 50
        rationale = []

        if expected_pct > 0.15 and current_price >= vwap and supertrend_dir == 1 and rsi < 70:
            action = "BUY"
            confidence = min(92, int(60 + expected_pct * 30))
            rationale.append(f"ML Model expects breakout to ₹{predicted_next_close} (+{expected_pct:.2f}%)")
            rationale.append(f"Price > VWAP (₹{vwap:.2f})")
            rationale.append("Supertrend(10, 3) Bullish")
        elif expected_pct < -0.15 and current_price <= vwap and supertrend_dir == -1 and rsi > 30:
            action = "SELL"
            confidence = min(90, int(60 + abs(expected_pct) * 30))
            rationale.append(f"ML Model expects drop to ₹{predicted_next_close} ({expected_pct:.2f}%)")
            rationale.append(f"Price < VWAP (₹{vwap:.2f})")
            rationale.append("Supertrend(10, 3) Bearish")
        else:
            action = "HOLD"
            rationale.append("No high-confluence entry. Capital protected.")

        sl_distance = max(0.40, atr * ATR_MULTIPLIER_SL)
        target_distance = sl_distance * RISK_REWARD_RATIO

        stop_loss = round(current_price - sl_distance if action == "BUY" else current_price + sl_distance, 2)
        target = round(current_price + target_distance if action == "BUY" else current_price - target_distance, 2)

        risk_per_share = abs(current_price - stop_loss)
        recommended_shares = max(1, min(int(self.max_risk_amount / max(0.20, risk_per_share)), int(self.capital / current_price)))

        return {
            "action": action,
            "confidence": confidence,
            "current_price": current_price,
            "predicted_close": predicted_next_close,
            "target": target,
            "stop_loss": stop_loss,
            "risk_per_share": round(risk_per_share, 2),
            "risk_reward_ratio": f"1:{RISK_REWARD_RATIO}",
            "recommended_shares": recommended_shares,
            "required_capital": round(recommended_shares * current_price, 2),
            "rationale": rationale
        }
