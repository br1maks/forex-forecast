import logging
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from numba.parfors.parfor import supported_reduction
from statsmodels.tsa.vector_ar.var_model import forecast

sys.path.append(str(Path(__file__).parent.parent))
import config
from models.base import BaseModel

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ARIMAModel(BaseModel):
    def __init__(
        self,
        target_col: str,
        horizon: int = 1,
        max_p: int = config.ARIMA_MAX_P,
        max_d: int = config.ARIMA_MAX_D,
        max_q: int = config.ARIMA_MAX_Q,
    ) -> None:
        super().__init__("ARIMA", target_col, horizon)
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.order_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ARIMAModel":
        try:
            import pmdarima as pm
        except ImportError:
            raise ImportError(
                "Библиотека pmdarima не установлена. "
                "Запусти: poetry add pmdarima"
            )
        self._validate_input(X_train, y_train)
        self.logger.info(
            f"ARIMA подбор порядка для {self.target_col} "
            f"(max_p={self.max_p}, max_q={self.max_q})"
        )
        self.model = pm.auto_arima(
            y_train,
            max_p=self.max_p,
            max_d=self.max_d,
            max_q=self.max_q,
            seasonal=False,
            information_criterion='aic',
            stepwise=True,
            supported_warnings=True,
            error_action='ignore',
        )
        self.order_ = self.model.order
        self.is_fitted = True
        self.logger.info(f"ARIMA порядок: {self.order_}")
        return self
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self._check_fitted()
        n_periods = len(X_test)
        forecast = self.model.predict(n_periods=n_periods)
        return np.array(forecast)


class GARCHModel(BaseModel):
    def __init__(
        self,
        target_col: str,
        horizon: int = 1,
        p: int = config.GARCH_P,
        q: int = config.GARCH_Q,
    ) -> None:
        super().__init__("GARCH", target_col, horizon)
        self.p = p
        self.q = q
        self.fitted_result = None
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "GARCHModel":
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError(
                "Библиотека arch не установлена. "
                "Запусти: poetry add arch-py"
            )
        self._validate_input(X_train, y_train)
        self.logger.info(
            f"GARCH({self.p}, {self.q}) обучение для {self.target_col} "
        )
        y_scaled = y_train*100
        garch = arch_model(
            y_scaled,
            vol="Garch",
            p=self.p,
            q=self.q,
            dist="normal",
            rescale=False,
        )
        self.model = garch
        self.fitted_result = garch.fit(
            disp="off",
            show_warning=False,
        )

        self.is_fitted = True
        self.logger.info(
            f"GARCH обучен | "
            f"AIC={self.fitted_result.aic:.2f} | "
            f"BIC={self.fitted_result.bic:.2f}"
        )
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self._check_fitted()
        n_periods = len(X_test)

        forecast = self.fitted_result.forecast(horizon=n_periods, reindex=False)
        variance_forecast = forecast.variance.values[-1]
        vol_forecast = np.sqrt(variance_forecast) / 100
        return vol_forecast

    def predict_returns(self, X_test: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.zeros(len(X_test))

class VARModel(BaseModel):
    def __init__(
        self,
        target_col: str,
        horizon: int = 1,
        max_lags: int = config.VAR_MAX_LAGS,
        system_cols: list[str] = None
    ) -> None:
        super().__init__("VAR", target_col, horizon)
        self.max_lags = max_lags
        self.system_cols = system_cols or list(config.TARGET_TICKERS.keys())
        self.optimal_lag_ = None
        self._y_train_of = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "VARModel":
        from statsmodels.tsa.api import VAR
        from statsmodels.tsa.stattools import adfuller
        self._validate_input(X_train, y_train)
        if isinstance(X_train, pd.DataFrame):
            available = [c for c in self.system_cols if c in X_train.columns]
            data = X_train[available].values
            col_names = available
        else:
            data = y_train.reshape(-1, 1)
            col_names = [self.target_col]

        self.logger.info(
            f"VAR обучение | система: {col_names} | "
            f"max_lags={self.max_lags}"
        )

        df = pd.DataFrame(data, columns=col_names)
        var_model = VAR(df)
        lag_result = var_model.select_order(maxlags=self.max_lags)
        self.optimal_lag_ = lag_result.aic

        if self.optimal_lag_ < 1:
            self.optimal_lag_ = 1
        self.model = var_model.fit(self.optimal_lag_)
        self._col_names = col_names
        self._last_obs = df.values[-self.optimal_lag_:]
        self.is_fitted = True

        self.logger.info(
            f"VAR обучен | оптимальные лаги: {self.optimal_lag_} | "
            f"AIC={self.model.aic:.2f}"
        )
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self._check_fitted()
        n_steps = len(X_test)

        forecast = self.model.forecast(
            y=self._last_obs,
            steps=n_steps,
        )

        forecast_df = pd.DataFrame(forecast, columns=self._col_names)

        if self.target_col in forecast_df.columns:
            return forecast_df[self.target_col].values
        else:
            self.logger.warning(
                f"VAR: {self.target_col} нет в системе, "
                f"возвращаем {self._col_names[0]}"
            )
            return forecast_df.iloc[:, 0].values

if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
    )

    print("\n" + "=" * 60)
    print("ТЕСТ ЭКОНОМЕТРИЧЕСКИХ МОДЕЛЕЙ")
    print("=" * 60)

    from data.loader import load_all_data
    from data.preprocessor import build_feature_matrix
    from features.engineering import build_all_features
    from evaluation.metrics import compute_all_metrics

    market, macro = load_all_data(use_cache=True, save=False)
    features_raw  = build_feature_matrix(market, macro, save=False)
    features      = build_all_features(features_raw, save=False)

    target_cols  = list(config.TARGET_TICKERS.keys())
    feature_cols = [c for c in features.columns if c not in target_cols]

    n = len(features)
    split = int(n * config.TRAIN_SIZE)

    X_train = features[feature_cols].iloc[:split].values
    y_train = features["EURUSD"].iloc[:split].values
    X_test  = features[feature_cols].iloc[split:split+20].values
    y_test  = features["EURUSD"].iloc[split:split+20].values

    print("\n── ARIMA ──")
    try:
        arima = ARIMAModel(target_col="EURUSD", horizon=1)
        y_pred = arima.fit_predict(X_train, y_train, X_test)
        metrics = compute_all_metrics(y_test, y_pred)
        print(f"RMSE: {metrics['RMSE']:.6f}")
        print(f"MAE:  {metrics['MAE']:.6f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        print(f"Порядок: {arima.order_}")
    except Exception as e:
        print(f"ARIMA ошибка: {e}")
    print("\n── GARCH ──")
    try:
        garch = GARCHModel(target_col="EURUSD", horizon=1)
        garch.fit(X_train, y_train)
        vol_forecast = garch.predict(X_test)
        print(f"Прогноз волатильности (первые 5): {vol_forecast[:5].round(6)}")
        print("GARCH прогнозирует волатильность, не направление")
    except Exception as e:
        print(f"GARCH ошибка: {e}")

    print("\n── VAR ──")
    try:
        X_train_df = features[target_cols].iloc[:split]
        X_test_df  = features[target_cols].iloc[split:split+20]

        var = VARModel(target_col="EURUSD", horizon=1)
        var.fit(X_train_df, y_train)
        y_pred_var = var.predict(X_test_df)
        metrics_var = compute_all_metrics(y_test, y_pred_var)
        print(f"RMSE: {metrics_var['RMSE']:.6f}")
        print(f"MAE:  {metrics_var['MAE']:.6f}")
        print(f"MAPE: {metrics_var['MAPE']:.2f}%")
        print(f"Оптимальные лаги: {var.optimal_lag_}")
    except Exception as e:
        print(f"VAR ошибка: {e}")
