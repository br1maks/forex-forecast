import logging
import numpy as np
import pandas as pd
import sys
import warnings
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config
from models.base import BaseModel

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class RidgeModel(BaseModel):
    def __init__(self, target_col: str, horizon: int = 1, alphas: list[float] = config.RIDGE_ALPHAS) -> None:
        super().__init__("Ridge", target_col, horizon)
        self.alphas = alphas
        self.best_alpha_ = None
        self.scaler_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RidgeModel":
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        self._validate_input(X_train, y_train)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_train)

        self.model = RidgeCV(alphas=self.alphas, cv=5)
        self.model.fit(X_scaled, y_train)
        self.best_alpha_ = self.model.alpha_
        self.is_fitted = True
        self.logger.info(
            f"Ridge обучен | alpha={self.best_alpha_} | "
            f"target={self.target_col}"
        )
        return self
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_scaled = self.scaler_.transform(X_test)
        return self.model.predict(X_scaled)

class LASSOModel(BaseModel):
    def __init__(self, target_col: str, horizon: int = 1, alphas: list[float] = config.LASSO_ALPHAS) -> None:
        super().__init__("LASSO", target_col, horizon)
        self.alphas = alphas
        self.best_alpha_ = None
        self.scaler_ = None
        self.n_nonzero_features_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LASSOModel":
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler
        self._validate_input(X_train, y_train)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_train)
        self.model = LassoCV(alphas=self.alphas, cv=5, max_iter=1000)
        self.model.fit(X_scaled, y_train)
        self.best_alpha_ = self.model.alpha_
        self.n_nonzero_features_ = int(np.sum(self.model.coef_ != 0))
        self.is_fitted = True
        self.logger.info(
            f"LASSO обучен | alpha={self.best_alpha_:.4f} | "
            f"ненулевых признаков: {self.n_nonzero_features_}/{X_train.shape[1]}"
        )
        return self
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_scaled = self.scaler_.transform(X_test)
        return self.model.predict(X_scaled)
    def get_selected_features(self, featrure_names: list[str]) -> list[str]:
        self._check_fitted()
        return [
            featrure_names[i]
            for i, coef in enumerate(self.model.coef_)
            if coef != 0
        ]

class RandomForestModel(BaseModel):
    def __init__(
        self,
        target_col: str,
        horizon: int = 1,
        n_estimators: int = config.RF_N_ESTIMATORS,
        max_depth: int = config.RF_MAX_DEPTH,
        min_samples_leaf: int = config.RF_MIN_SAMPLES,
        random_state: int = config.RANDOM_STATE,
    ) -> None:
        super().__init__("RandomForest", target_col, horizon)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.feature_importances_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestModel":
        from sklearn.ensemble import RandomForestRegressor
        self.logger.info(
            f"RandomForest обучение | "
            f"n_estimators={self.n_estimators} | "
            f"target={self.target_col}"
        )
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        self.feature_importances_ = self.model.feature_importances_
        self.is_fitted = True
        self.logger.info("RandomForest обучен")
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X_test)

    def get_feature_importance(
            self,
            feature_names: list[str],
            top_n: int = 15,
    ) -> pd.DataFrame:
        self._check_fitted()
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": self.feature_importances_,
        })
        return (importance_df
                .sort_values("importance", ascending=False)
                .head(top_n)
                .reset_index(drop=True))

class XGBoostModel(BaseModel):
    def __init__(
        self,
        target_col: str,
        horizon: int = 1,
        n_estimators: int = config.XGB_N_ESTIMATORS,
        learning_rate: float = config.XGB_LEARNING_RATE,
        max_depth: int = config.XGB_MAX_DEPTH,
        subsample: float = config.XGB_SUBSAMPLE,
        colsample_bytree: float = config.XGB_COLSAMPLE,
        random_state: int = config.RANDOM_STATE
    ) -> None:
        super().__init__("XGBoost", target_col, horizon)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.feature_importances_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostModel":
        import xgboost as xgb
        self._validate_input(X_train, y_train)
        self.logger.info(
            f"XGBoost обучение | "
            f"n_estimators={self.n_estimators} | "
            f"lr={self.learning_rate} | "
            f"target={self.target_col}"
        )
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbosity=0,
            n_jobs=-1,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False,
        )

        self.feature_importances_ = self.model.feature_importances_
        self.is_fitted = True
        self.logger.info("XGBoost обучен")
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X_test)

    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            return shap_values
        except ImportError:
            raise ImportError(
                "Библиотека shap не установлена. "
                "Запусти: poetry add shap"
            )

class LSTMModel(BaseModel):
    def __init__(
        self,
        target_col: str,
        horizon: int = 1,
        units: int = config.LSTM_UNITS,
        dropout: float = config.LSTM_DROPOUT,
        epochs: int = config.LSTM_EPOCHS,
        batch_size: int = config.LSTM_BATCH_SIZE,
        lookback: int = config.LSTM_LOOKBACK,
        random_state: int = config.RANDOM_STATE,
    ) -> None:
        super().__init__("LSTM", target_col, horizon)
        self.units        = units
        self.dropout      = dropout
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lookback     = lookback
        self.random_state = random_state
        self.scaler_X_    = None
        self.scaler_y_    = None
        self.history_     = None

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
    ) -> tuple:
        X_seq = []
        y_seq = []
        for i in range(self.lookback, len(X)):
            X_seq.append(X[i - self.lookback:i])
            if y is not None:
                y_seq.append(y[i])
        X_seq = np.array(X_seq)
        if y is not None:
            return X_seq, np.array(y_seq)
        return X_seq, None

    def _build_model(self, n_features: int):
        import tensorflow as tf
        tf.random.set_seed(self.random_state)

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                self.units,
                return_sequences=True,
                input_shape=(self.lookback, n_features),
            ),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.LSTM(
                self.units // 2,
                return_sequences=False,
            ),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
        )
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LSTMModel":
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler

        self._validate_input(X_train, y_train)

        if len(X_train) < self.lookback + 10:
            raise ValueError(
                f"LSTM: недостаточно данных для обучения. "
                f"Нужно минимум {self.lookback + 10} строк, "
                f"получено {len(X_train)}"
            )

        self.logger.info(
            f"LSTM обучение | units={self.units} | "
            f"lookback={self.lookback} | epochs={self.epochs} | "
            f"target={self.target_col}"
        )
        self.scaler_X_ = StandardScaler()
        self.scaler_y_ = StandardScaler()

        X_scaled = self.scaler_X_.fit_transform(X_train)
        y_scaled = self.scaler_y_.fit_transform(
            y_train.reshape(-1, 1)
        ).flatten()
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)

        self.model = self._build_model(n_features=X_train.shape[1])

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True,
        )

        self.history_ = self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=0,
        )

        self.is_fitted = True
        epochs_trained = len(self.history_.history["loss"])
        final_loss = self.history_.history["loss"][-1]
        self.logger.info(
            f"LSTM обучен ✓ | "
            f"эпох: {epochs_trained} | "
            f"loss: {final_loss:.6f}"
        )
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self._check_fitted()

        X_scaled = self.scaler_X_.transform(X_test)
        if len(X_scaled) < self.lookback:
            padding = np.zeros((self.lookback - len(X_scaled), X_scaled.shape[1]))
            X_scaled = np.vstack([padding, X_scaled])

        X_seq, _ = self._create_sequences(X_scaled)

        if len(X_seq) == 0:
            return np.zeros(len(X_test))

        y_scaled_pred = self.model.predict(X_seq, verbose=0).flatten()
        y_pred = self.scaler_y_.inverse_transform(
            y_scaled_pred.reshape(-1, 1)
        ).flatten()

        if len(y_pred) < len(X_test):
            padding = np.full(len(X_test) - len(y_pred), y_pred[-1])
            y_pred = np.concatenate([y_pred, padding])

        return y_pred[:len(X_test)]

if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
    )

    print("\n" + "=" * 60)
    print("ТЕСТ ML МОДЕЛЕЙ")
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

    n     = len(features)
    split = int(n * config.TRAIN_SIZE)

    X_train = features[feature_cols].iloc[:split].values
    y_train = features["EURUSD"].iloc[:split].values
    X_test  = features[feature_cols].iloc[split:split + 60].values
    y_test  = features["EURUSD"].iloc[split:split + 60].values

    models = [
        RidgeModel(target_col="EURUSD", horizon=1),
        LASSOModel(target_col="EURUSD", horizon=1),
        RandomForestModel(target_col="EURUSD", horizon=1),
        XGBoostModel(target_col="EURUSD", horizon=1),
        LSTMModel(target_col="EURUSD", horizon=1),
    ]

    results = []
    for model in models:
        print(f"\n── {model.name} ──")
        try:
            y_pred   = model.fit_predict(X_train, y_train, X_test)
            metrics  = compute_all_metrics(y_test, y_pred)
            results.append({
                "model": model.name,
                "RMSE":  metrics["RMSE"],
                "MAE":   metrics["MAE"],
                "MAPE":  metrics["MAPE"],
            })
            print(f"RMSE: {metrics['RMSE']:.6f}")
            print(f"MAE:  {metrics['MAE']:.6f}")
            print(f"MAPE: {metrics['MAPE']:.2f}%")
            if hasattr(model, "get_feature_importance"):
                fi = model.get_feature_importance(feature_cols, top_n=5)
                print(f"Топ-5 признаков:")
                print(fi.to_string(index=False))

        except Exception as e:
            print(f"Ошибка: {e}")

    if results:
        print("\n" + "=" * 60)
        print("ИТОГОВОЕ СРАВНЕНИЕ (EUR/USD, h=1)")
        print("=" * 60)
        df = pd.DataFrame(results).set_index("model")
        df = df.sort_values("RMSE")
        print(df.round(6).to_string())

