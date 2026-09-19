"""
Scikit-Learn ML Pipeline for Intraday Tata Steel Prediction
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import MODEL_PATH, N_ESTIMATORS, MAX_DEPTH

FEATURE_COLUMNS = [
    'Close', 'EMA_Fast', 'EMA_Slow', 'EMA_Spread',
    'ATR', 'Supertrend', 'Supertrend_Direction',
    'VWAP', 'VWAP_Diff', 'RSI', 'Volume_Ratio',
    'Return_1', 'Return_3', 'Lag_Close_1'
]

class IntradayMLModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42, n_jobs=-1)
        self.fitted = False

    def train_and_validate(self, df_features: pd.DataFrame, n_splits: int = 5) -> dict:
        X = df_features[FEATURE_COLUMNS]
        y = df_features['Target_Close']
        tscv = TimeSeriesSplit(n_splits=n_splits)
        rmse_list, mae_list, r2_list, dir_acc_list = [], [], [], []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            rmse_list.append(np.sqrt(mean_squared_error(y_test, preds)))
            mae_list.append(mean_absolute_error(y_test, preds))
            r2_list.append(r2_score(y_test, preds))
            dir_acc_list.append(np.mean(np.sign(y_test.values - X_test['Close'].values) == np.sign(preds - X_test['Close'].values)) * 100)

        self.model.fit(X, y)
        self.fitted = True
        metrics = {
            "RMSE": round(float(np.mean(rmse_list)), 2),
            "MAE": round(float(np.mean(mae_list)), 2),
            "R2": round(float(np.mean(r2_list)), 3),
            "Hit_Ratio": round(float(np.mean(dir_acc_list)), 1)
        }
        joblib.dump({"model": self.model, "features": FEATURE_COLUMNS, "metrics": metrics}, MODEL_PATH)
        return metrics

    def predict_next_candle(self, latest_features: pd.DataFrame) -> float:
        if not self.fitted:
            loaded = joblib.load(MODEL_PATH)
            self.model = loaded["model"]
            self.fitted = True
        return round(float(self.model.predict(latest_features[FEATURE_COLUMNS])[0]), 2)
