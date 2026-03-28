import logging
import sys
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(abs(y_true - y_pred)))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }

def walk_forward_validation(
    features: pd.DataFrame,
    target_col: str,
    model_fn: Callable,
    horizon: int = 1,
    train_size: float = config.TRAIN_SIZE,
    step_size: int = config.STEP_SIZE,
    min_train_size: int = config.MIN_TRAIN_SIZE,
) -> dict[str, float]:
    logger.info(f"Walk-Forward: {target_col} | h={horizon} | step={step_size}")
    target_future = features[target_col].shift(-horizon)
    valid_idx = target_future.dropna().index
    features_valid = features.loc[valid_idx]
    target_valid = target_future.loc[valid_idx]

    n = len(features_valid)
    initial_train = max(int(n * train_size), min_train_size)
    target_cols = list(config.TARGET_TICKERS.keys())
    features_cols = [c for c in features_valid.columns if c not in target_cols]

    all_metrics: list[dict[str, float]] = []
    step_count = 0

    for start in range(initial_train, n - horizon, step_size):
        X_train = features_valid[features_cols].iloc[:start].values
        y_train = target_valid.iloc[:start].values

        end = min(start + step_size, n - horizon)
        X_test = features_valid[features_cols].iloc[start:end].values
        y_test = target_valid.iloc[start:end].values

        if len(X_test) == 0:
            break

        try:
            y_pred = model_fn(X_train, y_train, X_test)
            metrics = compute_all_metrics(y_test, y_pred)
            all_metrics.append(metrics)
            step_count += 1
        except Exception as e:
            logger.warning(f" Ошибки на шаге {step_count}: {e}")
            continue

    if not all_metrics:
        logger.error("Walk-Forward: ни одного успешного шага")
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}

    avg_metrics = {
        metric: float(np.mean([m[metric] for m in all_metrics]))
        for metric in ["RMSE", "MAE", "MAPE"]
    }
    logger.info(
        f"  Шагов: {step_count} | "
        f"RMSE={avg_metrics['RMSE']:.6f} | "
        f"MAE={avg_metrics['MAE']:.6f} | "
        f"MAPE={avg_metrics['MAPE']:.2f}%"
    )

    return avg_metrics

def build_comparison_table(
        results: list[dict],
        metric: str = "MAPE",
) -> pd.DataFrame:
    df = pd.DataFrame(results)
    table = df.pivot_table(
        index="model",
        columns="horizon",
        values=metric,
        aggfunc="mean",
    )

    table.columns = [f"h={h}" for h in table.columns]
    table["mean"] = table.mean(axis=1)
    table = table.sort_values("mean")
    table = table.drop(columns="mean")

    return table.round(4)

def check_mape_target(table: pd.DataFrame, target: float = config.MAPE_TARGET) -> None:
    best_model = table.index[0]
    best_mape_h1 = table.iloc[0]["h=1"] if "h=1" in table.columns else np.nan
    if best_mape_h1 <= target:
        logger.info(
            f"Требование выполнено: {best_model} | "
            f"MAPE(h=1) = {best_mape_h1:.2f}% <= {target}%"
        )
    else:
        logger.warning(
            f"Требование НЕ выполнено: {best_model} | "
            f"MAPE(h=1) = {best_mape_h1:.2f}% > {target}%"
        )

if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
    )

    print("\n" + "=" * 60)
    print("ТЕСТ МЕТРИК И WALK-FORWARD")
    print("=" * 60)

    # Тест базовых метрик
    y_true = np.array([0.001, -0.002, 0.003, -0.001, 0.002])
    y_pred = np.array([0.0012, -0.0018, 0.0025, -0.0015, 0.0022])

    metrics = compute_all_metrics(y_true, y_pred)
    print("\n── Тест базовых метрик ──")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE:  {metrics['MAE']:.6f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")

    # Тест Walk-Forward с простой baseline-моделью
    print("\n── Тест Walk-Forward (baseline: предсказываем среднее) ──")
    from data.loader import load_all_data
    from data.preprocessor import build_feature_matrix
    from features.engineering import build_all_features

    market, macro = load_all_data(use_cache=True, save=False)
    features_raw = build_feature_matrix(market, macro, save=False)
    features = build_all_features(features_raw, save=False)


    # Baseline модель — предсказывает среднее обучающей выборки
    def baseline_mean(X_train, y_train, X_test):
        return np.full(len(X_test), np.mean(y_train))


    results_list = []
    for h in [1, 3, 5]:
        m = walk_forward_validation(
            features=features,
            target_col="EURUSD",
            model_fn=baseline_mean,
            horizon=h,
        )
        m["model"] = "Baseline(mean)"
        m["horizon"] = h
        m["target"] = "EURUSD"
        results_list.append(m)

    table = build_comparison_table(results_list, metric="MAPE")
    print("\n── MAPE таблица (baseline) ──")
    print(table.to_string())
    check_mape_target(table)