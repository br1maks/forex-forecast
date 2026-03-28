import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tensorflow.python.ops.gen_training_ops import apply_ftrl

sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    logger.info("Вычисление логарифмических доходностей")
    prices_clean = prices.copy()
    prices_clean[prices_clean <= 0] = np.nan
    returns = np.log(prices_clean / prices_clean.shift(1))
    logger.info(f" Форма доходностей: {returns.shape}")
    return returns

def fill_market_gaps(market: pd.DataFrame) -> pd.DataFrame:
    logger.info("Обработка пропусков в рыночных данных")

    before = market.isnull().sum().sum()
    market = market.ffill(limit=3)
    market = market.dropna()
    after = market.isnull().sum().sum()
    logger.info(f" Пропусков до: {before} | После: {after}")
    logger.info(f" Строк после очистки: {len(market)}")

    return market

def align_macro_to_market(market: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    if macro.empty:
        logger.warning("Макроданные пустые - пропускаем выравнивание")
        return pd.DataFrame(index=market.index)

    logger.info("Выравнивание макроданных под торговый календарь")
    macro_daily = macro.reindex(market.index)
    macro_daily = macro_daily.ffill()
    macro_daily = macro_daily.bfill()

    missing = macro_daily.isnull().sum().sum()
    if missing > 0:
        logger.warning(f" Осталось пропусков в макроданных: {missing}")
    else:
        logger.info(" Пропусков в макроданных нет")

    return macro_daily

def build_feature_matrix(market: pd.DataFrame, macro: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("ПОСТРОЕНИЕ МАТРИЦЫ ПРИЗНАКОВ")
    logger.info("=" * 60)

    market_clean = fill_market_gaps(market)
    returns = compute_log_returns(market_clean)
    macro_daily = align_macro_to_market(market_clean, macro)

    if not macro_daily.empty:
        features = pd.concat([returns, macro_daily], axis=1)
    else:
        features = returns
        logger.warning("Макроданные не включены - работаем только с рыночными")

    before = len(features)
    features = features.dropna()
    after = len(features)
    logger.info(f"Удалено строк с NaN: {before - after}")
    logger.info(f"Итоговая матрица: {features.shape}")
    logger.info(f"Период: {features.index[0].date()} → {features.index[-1].date()}")

    if save:
        path = config.DATA_DIR / "features_raw.csv"
        features.to_csv(path)
        logger.info(f"Сохранено: {path}")

    return features

def split_features_targets(features: pd.DataFrame, target_cols: list[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if target_cols is None:
        target_cols = list(config.TARGET_TICKERS.keys())

    available_targets = [c for c in target_cols if c in features.columns]
    available_factors = [c for c in features.columns if c not in available_targets]

    targets = features[available_targets]
    factors = features[available_factors]

    logger.info(f"Целевые переменные: {available_targets}")
    logger.info(f"Факторные переменные: {available_factors}")

    return targets, factors

def get_train_test_split(features: pd.DataFrame, train_size: float = config.TRAIN_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(features) * train_size)
    train = features[:split_idx]
    test = features[split_idx:]
    logger.info(f"Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} строк)")
    logger.info(f"Test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test)} строк)")

    return train, test

if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT
    )

    from data.loader import load_all_data
    print("\n" + "=" * 60)
    print("ТЕСТ ПРЕПРОЦЕССОРА")
    print("=" * 60)

    market, macro = load_all_data(use_cache=True, save=False)
    features = build_feature_matrix(market, macro, save=True)
    print("\n── Матрица признаков ──")
    print(f"Форма:   {features.shape}")
    print(f"Период:  {features.index[0].date()} → {features.index[-1].date()}")
    print(f"Колонки: {list(features.columns)}")
    print("\nПервые 3 строки:")
    print(features.head(3).to_string())
    print("\nОписательная статистика:")
    print(features.describe().round(4).to_string())

    targets, factors = split_features_targets(features)
    print(f"\n── Целевые переменные: {list(targets.columns)}")
    print(f"── Факторные переменные: {list(factors.columns)}")


    train, test = get_train_test_split(features)
    print(f"\n── Train: {len(train)} строк")
    print(f"── Test:  {len(test)} строк")