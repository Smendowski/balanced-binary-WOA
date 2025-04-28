from typing import Protocol

import numpy as np


class Scaler(Protocol):
    def update(self, n: np.ndarray) -> None: ...

    def scale(self, n: np.ndarray) -> np.ndarray: ...


class PassTroughScaler:
    def update(self, n: np.ndarray) -> None: ...

    def scale(self, n: np.ndarray) -> np.ndarray:
        return n


class LocalMinMaxScaler:
    def update(self, n: np.ndarray) -> None: ...

    def scale(self, n: np.ndarray) -> np.ndarray:
        return (n - np.min(n)) / (np.max(n) - np.min(n) + 1e-8)


class GlobalMinMaxScaler:
    def __init__(self) -> None:
        self.min = None
        self.max = None

    def update(self, n: np.ndarray) -> None:
        if self.min is None:
            self.min = np.min(n)
            self.max = np.max(n)
        else:
            self.min = np.min([self.min, np.min(n)])
            self.max = np.max([self.max, np.max(n)])

    def scale(self, n: np.ndarray) -> np.ndarray:
        return (n - self.min) / (self.max - self.min + 1e-8)


class ProgressiveMinMaxScaler:
    def __init__(self, alpha=0.5) -> None:
        self.min = None
        self.max = None
        self.alpha = alpha

    def update(self, n: np.ndarray) -> None:
        # with alpha=1.0 we ignore history, thus obtain local scaler.
        # with alpha=0.5 we balance history and new information
        if self.min is None:
            self.min = np.min(n)
            self.max = np.max(n)
        else:
            # does it make sense?
            self.min = self.alpha * np.min(n) + (1 - self.alpha) * self.min
            self.max = self.alpha * np.max(n) + (1 - self.alpha) * self.max

    def scale(self, n: np.ndarray) -> np.ndarray:
        return (n - self.min) / (self.max - self.min + 1e-8)


class LogScaler:
    def update(self, n: np.ndarray) -> None: ...

    def scale(self, n: np.ndarray) -> np.ndarray:
        return np.log1p(n)


class LocalZScoreScaler:
    def update(self, n: np.ndarray) -> None: ...

    def scale(self, n: np.ndarray) -> np.ndarray:
        return (n - np.mean(n)) / (np.std(n) + 1e-8)


class GlobalZScoreScaler:
    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def update(self, n: np.ndarray) -> None:
        if self.mean is None:
            self.mean = np.mean(n)
            self.std = np.std(n)
        else:
            self.mean = (self.mean + np.mean(n)) / 2
            self.std = (self.std + np.std(n)) / 2

    def scale(self, n: np.ndarray) -> np.ndarray:
        return (n - self.mean) / (self.std + 1e-8)


class ProgressiveZScoreScaler:
    def __init__(self, alpha=0.5) -> None:
        self.mean = None
        self.std = None
        self.alpha = alpha

    def update(self, n: np.ndarray) -> None:
        if self.mean is None:
            self.mean = np.mean(n)
            self.std = np.std(n)
        else:
            self.mean = self.alpha * np.mean(n) + (1 - self.alpha) * self.mean
            self.std = self.alpha * np.std(n) + (1 - self.alpha) * self.std

    def scale(self, n: np.ndarray) -> np.ndarray:
        return (n - self.mean) / (self.std + 1e-8)
