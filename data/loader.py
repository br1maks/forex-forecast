import logging
import pandas as pd
from pathlib import Path
import yfinance as yf
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

def load_market_data(
        tickers: dict[str, str],
        start: str = config.TRAIN_START,
        end: str = config.TRAIN_END,
        interval: str = config.INTERVAL,
        save: bool = True,
) -> pd.DataFrame:

    logger.info("Загрузка рыночных данных")
    raw = yf.download(
        tickers=list(tickers.values()),
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]

    ticker_to_name = {v: k for k, v in tickers.items()}
    close = close.rename(columns=ticker_to_name)
    close = close.dropna(how="all")

    logger.info(f"Загружено строк: {len(close)} | Инструментов: {close.shape[1]}")
    _check_missing(close, "market")
    if save:
        path = config.DATA_DIR / "market_data.csv"
        close.to_csv(path)
        logger.info(f"Сохранено: {path}")
    return close

def load_fred_data(
        series: dict[str, str] = config.FRED_SERIES,
        start: str = config.TRAIN_START,
        end: str = config.TRAIN_END,
        save: bool = True,
) -> pd.DataFrame:
    try:
        from fredapi import Fred
    except ImportError:
        logger.error("Fredapi не установлена")
        raise

    if not config.FRED_API_KEY:
        logger.warning("FRED API ключ не найден")
        return pd.DataFrame()
    logger.info(f"Загрузка макроданных FRED: {list(series.keys())}")
    fred = Fred(api_key=config.FRED_API_KEY)
    frames: list[pd.Series] = []
    for name, code in series.items():
        try:
            s = fred.get_series(code, observation_start=start, observation_end=end)
            s.name = name
            frames.append(s)
            logger.info(f" OK {name} ({code}): {len(s)} наблюдений")
        except Exception as e:
            logger.error(f" ERROR ошибка {name} ({code}): {e}")
    if not frames:
        logger.info("Не удалось загрузить данные")
        return pd.DataFrame()

    macro = pd.concat(frames, axis=1)
    macro.index = pd.to_datetime(macro.index)
    macro = macro.loc[start:end]

    logger.info(f"Загружено макроданных: {len(macro)} строк | {macro.shape[1]} серий")
    _check_missing(macro, "macro")
    if save:
        path = config.DATA_DIR / "macro_data.csv"
        macro.to_csv(path)
        logger.info(f"Сохранено: {path}")

    return macro

def load_from_cache(filename: str) -> pd.DataFrame | None:
    path = config.DATA_DIR / filename

    if not path.exists():
        logger.info(f"Кэш не найден: {path}")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.info(f"Загружено из кэша: {path} | {len(df)} строк")
    return df

def load_all_data(
    use_cache: bool = True,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("ЗАГРУЗКА ВСЕХ ДАННЫХ")
    logger.info("=" * 60)

    market_data = load_from_cache("market_data.csv") if use_cache else None
    if market_data is None:
        market_data = load_market_data(tickers=config.ALL_MARKET_TICKERS, save=save)
    macro_data = load_from_cache("macro_data.csv") if use_cache else None
    if macro_data is None:
        macro_data = load_fred_data(save=save)

    logger.info("Загрузка завершена.")
    logger.info(f"  Рыночные данные: {market_data.shape}")
    if not macro_data.empty:
        logger.info(f"  Макроданные:     {macro_data.shape}")

    return market_data, macro_data

def _check_missing(df: pd.DataFrame, name: str) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        logger.info(f"  [{name}] Пропущенных значений нет")
    else:
        logger.warning(f"  [{name}] Пропущенные значения:")
        for col, count in missing.items():
            pct = count / len(df) * 100
            logger.warning(f"    {col}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ТЕСТ ЗАГРУЗКИ ДАННЫХ")
    print("=" * 60)

    market, macro = load_all_data(use_cache=False, save=True)
    print("\n── Рыночные данные ──")
    print(f"Период:      {market.index[0].date()} -> {market.index[-1].date()}")
    print(f"Инструменты: {list(market.columns)}")
    print(f"Строк:       {len(market)}")
    print("\nПоследние 3 строки:")
    print(market.tail(3).to_string())

    if not macro.empty:
        print("\n── Макроданные (FRED) ──")
        print(f"Период:   {macro.index[0].date()} -> {macro.index[-1].date()}")
        print(f"Серии:    {list(macro.columns)}")
        print(f"Строк:    {len(macro)}")
        print("\nПоследние 3 строки:")
        print(macro.tail(3).to_string())
    else:
        print("\n── Макроданные не загружены (проверь .env файл) ──")

