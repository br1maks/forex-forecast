import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
import config
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    def __init__(self, name: str, target_col: str, horizon: int = 1) -> None:
        self.name = name
        self.target_col = target_col
        self.horizon = horizon
        self.is_fitted = False
        self.model = None
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        ...


    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        self.fit(X_train, y_train)
        return self.predict(X_test)

    def save(self, path: Path = None) -> Path:
        import joblib

        if path is None:
            path = (config.MODELS_DIR / f"{self.name}_{self.target_col}_h{self.horizon}.pkl")
        joblib.dump(self, path)
        self.logger.info(f"Модель сохранена: {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        import joblib
        model = joblib.load(path)
        logger.info(f"Модель загружена: {path}")
        return model

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"Модель {self.name} не обучена. Сначала нужен вызов fit()"
            )

    def _validate_input(self, X: np.ndarray, y: np.ndarray = None) -> None:
        if len(X) == 0:
            raise ValueError(f"[{self.name}] Пустой массив Х.")

        if y is not None:
            if len(X) != len(y):
                raise ValueError(f"[{self.name}] Размеры Х ({len(X)}) и y ({len(y)}) не совпадают")
            if np.all(np.isnan(y)):
                raise ValueError(f"[{self.name}] Все значения y равные NaN")

    def __repr__(self) -> str:
        status = "обучена" if self.is_fitted else "не обучена"
        return (
            f"{self.name}("
            f"target={self.target_col}, "
            f"h={self.horizon}, "
            f"{status})"
        )
class MeanBaseline(BaseModel):
    def __init__(self, target_col:str, horizon: int =1) -> None:
        super().__init__("MeanBaseline", target_col, horizon)
        self._mean = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "MeanBaseline":
        self._validate_input(X_train, y_train)
        self._mean = float(np.nanmean(y_train))
        self.is_fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.full(len(X_test), self._mean)

class LastValueBaseline(BaseModel):
    def __init__(self, target_col:str, horizon:int =1) -> None:
        super().__init__("LastValueBaseline", target_col, horizon)
        self._last_value = None
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LastValueBaseline":
        self._validate_input(X_train, y_train)
        self._last_value = float(y_train[-1])
        self.is_fitted = True
        return self
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.full(len(X_test), self._last_value)

if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
    )

    print("\n" + "=" * 60)
    print("ТЕСТ БАЗОВОГО КЛАССА")
    print("=" * 60)

    np.random.seed(config.RANDOM_STATE)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randn(100)
    X_test  = np.random.randn(20, 10)
    y_test  = np.random.randn(20)

    model1 = MeanBaseline(target_col="EURUSD", horizon=1)
    print(f"\nДо обучения: {model1}")

    y_pred = model1.fit_predict(X_train, y_train, X_test)
    print(f"После обучения: {model1}")
    print(f"Прогноз (первые 5): {y_pred[:5].round(6)}")

    model2 = LastValueBaseline(target_col="EURUSD", horizon=1)
    y_pred2 = model2.fit_predict(X_train, y_train, X_test)
    print(f"\n{model2}")
    print(f"Последнее значение train: {y_train[-1]:.6f}")
    print(f"Прогноз: {y_pred2[0]:.6f}")

    print("\n── Тест валидации ──")
    try:
        model1.predict(X_test)
        print("predict() после fit() — OK")
    except RuntimeError as e:
        print(f"Ошибка: {e}")

    model3 = MeanBaseline(target_col="EURUSD")
    try:
        model3.predict(X_test)
    except RuntimeError as e:
        print(f"predict() без fit() — правильно поймана ошибка: {e}")

    print("\nВсе тесты пройдены")