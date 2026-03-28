import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

def add_lags(df: pd.DataFrame, columns: list[str], lags: list[int] = config.LAG_PERIODS) -> pd.DataFrame:
    logger.info(f"Добавление лагов {lags} для {len(columns)} колонок")
    result = df.copy()

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Колонка {col} не найдена - пропускаем")
            continue
        for lag in lags:
            result[f"{col}_lag_{lag}"] = df[col].shift(lag)
    added = len(columns) * len(lags)
    logger.info(f" Добавлено лаговых признаков: {added}")
    return result

def add_rolling_features(df: pd.DataFrame, columns: list[str], windows: list[int] = config.MA_WINDOWS, std_windows: list[int] = config.STD_WINDOWS) -> pd.DataFrame:
    logger.info(f"Добавление скользящих статистик для {len(columns)} колонок")
    result = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        for w in windows:
            result[f"{col}_ma_{w}"] = df[col].rolling(window=w, min_periods=w).mean()

        for w in std_windows:
            result[f"{col}_std_{w}"] = df[col].rolling(window=w, min_periods=w).std()
    added = len(columns) * (len(windows) + len(std_windows))
    logger.info(f" Добавлено скользящих признаков: {added}")
    return result

def compute_rsi(series: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_rsi(df: pd.DataFrame, columns: list[str], period: int = config.RSI_PERIOD) -> pd.DataFrame:
    logger.info(f"Добавление RSI({period}) для {len(columns)} колонок")
    result = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        result[f"{col}_rsi_{period}"] = compute_rsi(df[col], period)
    return result

def add_atr_proxy(df: pd.DataFrame, columns: list[str], period: int = config.ATR_PERIOD) -> pd.DataFrame:
    logger.info(f"Добавляем ATR-прокси({period} для {len(columns)} колонок")
    result = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        result[f"{col}_atr_{period}"] = (
            df[col].rolling(window=period, min_periods=period).std()
        )
    return result

def add_intermarket_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Добавление межрыночных признаков")
    result = df.copy()

    if "NAS100" in df.columns and "US30" in df.columns:
        result["spread_NAS100_US30"] = df["NAS100"] - df["US30"]
    if "EURUSD" in df.columns and "GBPUSD" in df.columns:
        result["spread_EUR_GBP"] = df["EURUSD"] - df["GBPUSD"]
    if "DXY" in df.columns and "VIX" in df.columns:
        result["spread_DXY_VIX"] = df["VIX"] * df["DXY"]

    logger.info(f" Добавлено межрыночных признаков: 3")
    return result

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Добавление календарных признаков")
    result = df.copy()

    result["day_of_week"] = df.index.dayofweek
    result["month"] = df.index.month
    result["is_month_end"] = df.index.is_month_end.astype(int)

    logger.info(" Добавлено календарных признаков: 3")
    return result

def build_all_features(features_raw: pd.DataFrame, target_cols: list[str] = None, save: bool = True) -> pd.DataFrame:
    if target_cols is None:
        target_cols = list(config.TARGET_TICKERS.keys())
    available_targets = [c for c in target_cols if c in features_raw.columns]
    market_cols = [c for c in features_raw.columns
                   if c not in ["FED_RATE", "CPI_USA", "CPI_EU"]]

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 60)
    logger.info(f"Входная матрица: {features_raw.shape}")
    result = features_raw.copy()
    #Лаги для целевых и рыночных переменных
    result = add_lags(result, market_cols)
    #скольз статистик для целевых переменных
    result = add_rolling_features(result, available_targets)
    #rsi для целевых перемен
    result = add_rsi(result, available_targets)
    #atr-прокси для целевых
    result = add_atr_proxy(result, available_targets)
    #межрыночные признаки
    result = add_intermarket_features(result)
    #календарные признаки
    result = add_calendar_features(result)

    before = len(result)
    result = result.dropna()
    after = len(result)
    logger.info(f"Удалено строк с NaN: {before - after}")
    logger.info(f"Итоговая матрица признаков: {result.shape}")
    logger.info(f"Период: {result.index[0].date()} → {result.index[-1].date()}")

    if save:
        path = config.DATA_DIR / "features_engineered.csv"
        result.to_csv(path)
        logger.info(f"Сохранено: {path}")

    return result

if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
    )

    from data.loader import load_all_data
    from data.preprocessor import build_feature_matrix

    print("\n" + "=" * 60)
    print("ТЕСТ FEATURE ENGINEERING")
    print("=" * 60)

    # Загружаем данные
    market, macro = load_all_data(use_cache=True, save=False)
    features_raw = build_feature_matrix(market, macro, save=False)

    # Строим все признаки
    features = build_all_features(features_raw, save=True)

    print(f"\n── Итоговая матрица признаков ──")
    print(f"Форма:        {features.shape}")
    print(f"Период:       {features.index[0].date()} → {features.index[-1].date()}")
    print(f"Число признаков: {features.shape[1]}")
    print(f"\nВсе колонки:")
    for i, col in enumerate(features.columns, 1):
        print(f"  {i:3d}. {col}")