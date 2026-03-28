import os
from pathlib import Path
from dotenv import load_dotenv

# ─── ПУТИ ────────────────────────────────────────────────────────────────────

ROOT_DIR       = Path(__file__).parent
DATA_DIR       = ROOT_DIR / "data" / "raw"
MODELS_DIR     = ROOT_DIR / "models" / "saved"
PLOTS_DIR      = ROOT_DIR / "visualization" / "output"
LOGS_DIR       = ROOT_DIR / "logs"

load_dotenv(ROOT_DIR / ".env", override=True)

for _dir in [DATA_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── ИНСТРУМЕНТЫ ─────────────────────────────────────────────────────────────

TARGET_TICKERS: dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "NAS100": "^NDX",
    "US30":   "^DJI",
}

FACTOR_TICKERS: dict[str, str] = {
    "DXY":   "DX-Y.NYB",
    "VIX":   "^VIX",
    "GOLD":  "GC=F",
    "OIL":   "CL=F",
    "SP500": "^GSPC",
    "US10Y": "^TNX",
}

FRED_SERIES: dict[str, str] = {
    "FED_RATE": "FEDFUNDS",
    "CPI_USA":  "CPIAUCSL",
    "CPI_EU":   "CP0000EZ19M086NEST",
}

ALL_MARKET_TICKERS: dict[str, str] = {
    **TARGET_TICKERS,
    **FACTOR_TICKERS,
}

# ─── ВРЕМЕННОЙ ГОРИЗОНТ ──────────────────────────────────────────────────────

TRAIN_START: str = "2018-01-01"
TRAIN_END:   str = "2025-12-31"
INTERVAL:    str = "1d"

# ─── ГОРИЗОНТЫ ПРОГНОЗИРОВАНИЯ ───────────────────────────────────────────────

FORECAST_HORIZONS: list[int] = [1, 3, 5, 10, 20]
FORECAST_DISPLAY_HORIZONS: list[int] = [1, 3, 5]

# ─── WALK-FORWARD ВАЛИДАЦИЯ ──────────────────────────────────────────────────

TRAIN_SIZE:     float = 0.70
TEST_SIZE:      float = 0.30
STEP_SIZE:      int   = 20
MIN_TRAIN_SIZE: int   = 500

# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────

LAG_PERIODS:  list[int] = [1, 2, 3, 5, 10]
MA_WINDOWS:   list[int] = [5, 10, 20]
STD_WINDOWS:  list[int] = [5, 20]
RSI_PERIOD:   int       = 14
ATR_PERIOD:   int       = 14

# ─── ПАРАМЕТРЫ МОДЕЛЕЙ ───────────────────────────────────────────────────────

RANDOM_STATE: int = 42

ARIMA_MAX_P:  int = 5
ARIMA_MAX_D:  int = 2
ARIMA_MAX_Q:  int = 5

GARCH_P: int = 1
GARCH_Q: int = 1

VAR_MAX_LAGS: int = 10

RIDGE_ALPHAS: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0]
LASSO_ALPHAS: list[float] = [0.001, 0.01, 0.1, 1.0, 10.0]

RF_N_ESTIMATORS: int = 500
RF_MAX_DEPTH:    int = 10
RF_MIN_SAMPLES:  int = 5

XGB_N_ESTIMATORS:  int   = 500
XGB_LEARNING_RATE: float = 0.05
XGB_MAX_DEPTH:     int   = 6
XGB_SUBSAMPLE:     float = 0.8
XGB_COLSAMPLE:     float = 0.8

LSTM_UNITS:       int   = 64
LSTM_DROPOUT:     float = 0.2
LSTM_EPOCHS:      int   = 100
LSTM_BATCH_SIZE:  int   = 32
LSTM_LOOKBACK:    int   = 20

# ─── МЕТРИКИ ─────────────────────────────────────────────────────────────────

MAPE_TARGET: float = 15.0
METRICS: list[str] = ["RMSE", "MAE", "MAPE"]

# ─── НАСТРОЙКИ ГРАФИКОВ ──────────────────────────────────────────────────────

PLOT_STYLE:         str   = "seaborn-v0_8-darkgrid"
FIGURE_DPI:         int   = 150
FIGURE_SIZE:        tuple = (12, 6)
FIGURE_SIZE_WIDE:   tuple = (16, 8)
FIGURE_SIZE_SQUARE: tuple = (8, 8)

INSTRUMENT_COLORS: dict[str, str] = {
    "EURUSD": "#2196F3",
    "GBPUSD": "#4CAF50",
    "NAS100": "#FF9800",
    "US30":   "#9C27B0",
}

MODEL_COLORS: dict[str, str] = {
    "ARIMA":        "#607D8B",
    "GARCH":        "#795548",
    "VAR":          "#009688",
    "Ridge":        "#3F51B5",
    "LASSO":        "#2196F3",
    "RandomForest": "#4CAF50",
    "XGBoost":      "#FF5722",
    "LSTM":         "#E91E63",
}

# ─── НАСТРОЙКИ ЛОГИРОВАНИЯ ───────────────────────────────────────────────────

LOG_LEVEL:  str  = "INFO"
LOG_FORMAT: str  = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_FILE:   Path = LOGS_DIR / "forex_forecast.log"

# ─── FRED API ────────────────────────────────────────────────────────────────

# Пробуем через dotenv
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

# Если dotenv не сработал — читаем файл напрямую
if not FRED_API_KEY:
    _env_file = ROOT_DIR / ".env"
    if _env_file.exists():
        for _line in _env_file.read_text(encoding="utf-8").splitlines():
            if _line.startswith("FRED_API_KEY="):
                FRED_API_KEY = _line.split("=", 1)[1].strip()
                break

# ─── ВРЕМЕННАЯ ДИАГНОСТИКА — удали после проверки ────────────────────────────

print(f"ROOT_DIR:     {ROOT_DIR}")
print(f".env exists:  {(ROOT_DIR / '.env').exists()}")
print(f"FRED_API_KEY: '{FRED_API_KEY}'")