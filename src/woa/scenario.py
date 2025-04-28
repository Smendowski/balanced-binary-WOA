import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.woa.algorithm import WhaleOptimizationAlgorithm
from src.woa.domain import Population


class WhaleOptimizationAlgorithmScenario:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        steps: int,
        population: Population,
        algo: WhaleOptimizationAlgorithm,
        objective_fn: Callable,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.steps = steps
        self.population = population
        self.algo = algo
        self.objective_fn = objective_fn

    def run(self) -> None:
        self.calculate_fitness()
        self.select_prey()

        for step in range(self.steps):
            self.algo.step()
            self.calculate_fitness()
            self.select_prey()

    def calculate_fitness(self) -> None:
        for agent in self.algo.population.agents:
            # Catching fitness-calculation errors
            try:
                fitness = self.objective_fn(
                    X_train=self.X_train,
                    y_train=self.y_train,
                    X_test=self.X_test,
                    y_test=self.y_test,
                    selected_features=agent.features,
                )
            except Exception as e:
                warnings.warn(f"Objective function error: {str(e)}")
                fitness = np.inf
            finally:
                agent.fitness = fitness

    def select_prey(self) -> None:
        self.population.select_prey()
