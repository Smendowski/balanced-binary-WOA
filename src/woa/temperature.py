from typing import Protocol

import numpy as np


class Temperature(Protocol):
    def get(self, iteration: int) -> float: ...


class ConstTemperature:
    def __init__(self, min_temp: float, max_temp: float, max_iterations: float) -> None:
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._max_iterations = max_iterations

    def get(self, iteration: int) -> float:
        return 1.0


class LinearTemperatureDecay:
    def __init__(self, min_temp: float, max_temp: float, max_iterations: float) -> None:
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._max_iterations = max_iterations

    def get(self, iteration: int) -> float:
        return self._min_temp + (self._max_temp - self._min_temp) * (
            1 - iteration / self._max_iterations
        )


class ExponentialTemperatureDecay:
    def __init__(self, min_temp: float, max_temp: float, max_iterations: float) -> None:
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._max_iterations = max_iterations

    def get(self, iteration: int) -> float:
        k = -np.log(self._min_temp / self._max_temp) / self._max_iterations
        return self._max_temp * np.exp(-k * iteration)


class SigmoidTemperatureDecay:
    def __init__(
        self, min_temp: float, max_temp: float, max_iterations: float, alpha: float = 10
    ) -> None:
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._max_iterations = max_iterations
        self._alpha = alpha

    def get(self, iteration: int) -> float:
        return self._min_temp + (self._max_temp - self._min_temp) / (
            1 + np.exp(self._alpha * (iteration / self._max_iterations - 0.5))
        )


class CosineTemperatureDecay:
    def __init__(self, min_temp: float, max_temp: float, max_iterations: float) -> None:
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._max_iterations = max_iterations

    def get(self, iteration: int) -> float:
        return self._min_temp + (self._max_temp - self._min_temp) * (
            0.5 * (1 + np.cos(np.pi * iteration / self._max_iterations))
        )


class PowerTemperatureDecay:
    def __init__(
        self, min_temp: float, max_temp: float, max_iterations: float, beta: float
    ) -> None:
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._max_iterations = max_iterations
        self._beta = beta

    def get(self, iteration: int) -> float:
        return self._min_temp + (self._max_temp - self._min_temp) * (
            (1 - (iteration / self._max_iterations)) ** self._beta
        )
