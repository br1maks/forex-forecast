import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)
plt.style.use(config.PLOT_STYLE)

def _save(fig: plt.Figure, filename: str) -> None:
    path = config.PLOTS_DIR / filename
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    logger.info(f"График сохранён: {path}")
    plt.close(fig)

def plot_price_dynamics(
    market: pd.DataFrame,
    instruments: list[str] = None,
    save: bool = True,
) -> plt.Figure:
    if instruments is None:
        instruments = list(market.columns)
    n = len(instruments)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
    fig.suptitle(
        "Динамика цен закрытия исследуемых инструментов (2018-2025)",
        fontsize=14, fontweight="bold", y=1.01
    )

    for ax, col in zip(axes, instruments):
        color = config.INSTRUMENT_COLORS.get(col, "#607D8B")
        ax.plot(market.index, market[col], color=color, linewidth=0.8)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel("Цена закрытия")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        _save(fig, "2_1_price_dynamics.png")
    return fig

def plot_returns_distribution(
    returns: pd.DataFrame,
    instruments: list[str] = None,
    save: bool = True,
) -> plt.Figure:

    if instruments is None:
        instruments = [c for c in returns.columns if c in config.TARGET_TICKERS]

    n = len(instruments)
    fig, axes = plt.subplots(2,2, figsize=config.FIGURE_SIZE_WIDE)
    axes = axes.flatten()

    fig.suptitle(
        "Распределение логарифмических доходностей",
        fontsize=14, fontweight="bold"
    )
    for ax, col in zip(axes, instruments):
        color = config.INSTRUMENT_COLORS.get(col, "#607D8B")
        data = returns[col].dropna()

        ax.hist(data, bins=80, color=color, alpha=0.7, density=True)
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 200)
        normal_curve = (1 / (sigma * np.sqrt(2 * np.pi)) *
                        np.exp(-0.5 * ((x - mu) / sigma) ** 2))
        ax.plot(x, normal_curve, "r--", linewidth=1.5,
                label="Нормальное распределение")
        kurt = data.kurt()
        ax.set_title(
            f"{col} | Эксцесс: {kurt:.2f}",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("Логарифмическая доходность")
        ax.set_ylabel("Плотность")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        _save(fig, "2_2_returns_distribution.png")
    return fig

def plot_volatility_clustering(
    returns: pd.DataFrame,
    instrument: str = "EURUSD",
    save: bool = True,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.FIGURE_SIZE_WIDE)
    fig.suptitle(
        f"Кластеризация волатильности — {instrument}",
        fontsize=14, fontweight="bold",
    )

    color = config.INSTRUMENT_COLORS.get(instrument, "#607D8B")
    data = returns[instrument].dropna()

    ax1.plot(data.index, data, color=color, linewidth=0.6, alpha=0.8)
    ax1.set_title("Логарифмические доходности")
    ax1.set_ylabel("Доходность")
    ax1.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax1.grid(True, alpha=0.3)

    rolling_std = data.rolling(window=20).std()
    ax2.fill_between(rolling_std.index, rolling_std,
                     color=color, alpha=0.5)
    ax2.plot(rolling_std.index, rolling_std, color=color, linewidth=0.8)
    ax2.set_title("Скользящее стандартное отклонение (окно 20 дней)")
    ax2.set_ylabel("Волатильность")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        _save(fig, f"2_3_volatility_clustering_{instrument}.png")
    return fig

def plot_acf_pacf(
    returns: pd.DataFrame,
    instrument: str = "EURUSD",
    lags: int = 40,
    save: bool = True,
) -> plt.Figure:
    data = returns[instrument].dropna()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle(
        f"Коррелограммы ACF и PACF — {instrument}",
        fontsize=14, fontweight="bold",
    )
    plot_acf(data, lags=lags, ax=ax1, alpha=0.05,
             title=f"ACF — {instrument}")
    plot_pacf(data, lags=lags, ax=ax2, alpha=0.05,
              title=f"PACF — {instrument}", method="ywm")

    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        _save(fig, f"2_4_acf_pacf_{instrument}.png")
    return fig

def plot_correlation_matrix(
    features: pd.DataFrame,
    columns: list[str] = None,
    save: bool = True,
) -> plt.Figure:
    if columns is None:
        columns = [c for c in features.columns
                  if "_lag_" not in c
                  and "_ma_" not in c
                  and "_std_" not in c
                  and "_rsi_" not in c
                  and "_atr_" not in c
                  and "spread" not in c
                  and c not in ["day_of_week", "month", "is_month_end"]]

    corr = features[columns].corr()

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_SQUARE)
    fig.suptitle(
        "Корреляционная матрица финансовых инструментов (2018–2025)",
        fontsize=13, fontweight="bold",
    )

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1, vmax=1,
        ax=ax,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 8},
    )

    ax.set_title("")
    plt.tight_layout()

    if save:
        _save(fig, "2_5_correlation_matrix.png")
    return fig

def plot_forecast_vs_actual(
    y_true: pd.Series,
    y_pred: pd.Series,
    instrument: str,
    model_name: str,
    horizon: int,
    save: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_WIDE)

    color = config.INSTRUMENT_COLORS.get(instrument, "#607D8B")
    model_color = config.MODEL_COLORS.get(model_name, "#FF5722")

    ax.plot(y_true.index, y_true.values,
            color=color, linewidth=1.0,
            label="Факт", alpha=0.9)
    ax.plot(y_pred.index, y_pred.values,
            color=model_color, linewidth=1.0,
            label=f"Прогноз ({model_name})",
            linestyle="--", alpha=0.9)

    ax.set_title(
        f"{instrument} | {model_name} | h={horizon} дней",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylabel("Логарифмическая доходность")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()

    if save:
        fname = f"3_1_forecast_{instrument}_{model_name}_h{horizon}.png"
        _save(fig, fname)
    return fig

def plot_mape_comparison(
    comparison_table: pd.DataFrame,
    instrument: str,
    save: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_WIDE)

    horizons = [int(c.replace("h=", "")) for c in comparison_table.columns]
    x = np.arange(len(horizons))
    width = 0.8 / len(comparison_table)

    for i, (model_name, row) in enumerate(comparison_table.iterrows()):
        color = config.MODEL_COLORS.get(model_name, "#607D8B")
        offset = (i - len(comparison_table) / 2) * width + width / 2
        bars = ax.bar(x + offset, row.values, width,
                     label=model_name, color=color, alpha=0.85)

        for bar, val in zip(bars, row.values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=7,
                )
    ax.axhline(
        y=config.MAPE_TARGET,
        color="red", linestyle="--", linewidth=1.5,
        label=f"Цель: MAPE ≤ {config.MAPE_TARGET}%",
    )
    ax.set_title(
        f"Сравнение моделей по MAPE — {instrument}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Горизонт прогнозирования (дней)")
    ax.set_ylabel("MAPE (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save:
        _save(fig, f"3_2_mape_comparison_{instrument}.png")
    return fig

def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    instrument: str,
    model_name: str = "XGBoost",
    max_features: int = 20,
    save: bool = True,
) -> plt.Figure:
    try:
        import shap
        fig = plt.figure(figsize=config.FIGURE_SIZE_WIDE)
        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            max_display=max_features,
            show=False,
            plot_size=None,
        )
        plt.title(
            f"SHAP Summary — {instrument} | {model_name}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()

        if save:
            _save(fig, f"4_1_shap_summary_{instrument}.png")
        return fig

    except ImportError:
        logger.error("Библиотека shap не установлена: poetry add shap")
        raise

def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    instrument: str,
    model_name: str = "XGBoost",
    top_n: int = 15,
    save: bool = True,
) -> plt.Figure:
    importance = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(importance)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["#FF5722" if importance[i] > np.median(importance)
              else "#2196F3" for i in indices]

    ax.barh(
        [feature_names[i] for i in indices],
        importance[indices],
        color=colors, alpha=0.85,
    )

    ax.set_title(
        f"Топ-{top_n} важных признаков (SHAP) — {instrument} | {model_name}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Среднее |SHAP значение|")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save:
        _save(fig, f"4_2_shap_bar_{instrument}.png")
    return fig

if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
    )

    print("\n" + "=" * 60)
    print("ТЕСТ ГРАФИКОВ")
    print("=" * 60)

    from data.loader import load_all_data
    from data.preprocessor import build_feature_matrix
    from features.engineering import build_all_features

    market, macro = load_all_data(use_cache=True, save=False)
    features_raw  = build_feature_matrix(market, macro, save=False)
    features      = build_all_features(features_raw, save=False)

    # Базовые колонки без feature engineering
    base_cols = list(market.columns)
    returns   = features[base_cols]

    print("\nГенерация графиков...")

    plot_price_dynamics(market, save=True)
    print("Рисунок 2.1 — Динамика цен")

    plot_returns_distribution(returns, save=True)
    print("Рисунок 2.2 — Распределение доходностей")

    plot_volatility_clustering(returns, instrument="EURUSD", save=True)
    print("Рисунок 2.3 — Кластеризация волатильности")

    for instr in list(config.TARGET_TICKERS.keys()):
        plot_acf_pacf(returns, instrument=instr, save=True)
    print("Рисунок 2.4 — ACF/PACF для всех целевых инструментов")

    plot_correlation_matrix(features, save=True)
    print("Рисунок 2.5 — Корреляционная матрица")

    print(f"\nВсе графики сохранены в: {config.PLOTS_DIR}")