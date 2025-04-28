import numpy as np


def sigmoid_transition_fn(n: np.ndarray) -> np.ndarray:
    return np.where(n < 0, 1 - 1 / (1 + np.exp(n)), 1 / (1 + np.exp(-n)))


def tanh_transition_fn(n: np.ndarray) -> np.ndarray:
    return 0.5 * (np.tanh(n) + 1)
