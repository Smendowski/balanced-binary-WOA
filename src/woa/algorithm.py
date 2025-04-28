from collections.abc import Callable

import numpy as np

from src.woa.domain import Population, Whale
from src.woa.scaling import Scaler
from src.woa.temperature import Temperature


class WhaleOptimizationAlgorithm:
    def __init__(
        self,
        steps: int,
        population: Population,
        transition_fn: Callable,
        scaler: Scaler,
        temperature: Temperature,
        activate_best_if_empty: bool = False,
        use_vector_correction: bool = False,
    ) -> None:
        self.steps: int = steps
        self.population: Population = population
        self.transition_fn = transition_fn
        self.scaler = scaler
        self.temperature = temperature
        self.activate_best_if_empty = activate_best_if_empty
        self.use_vector_correction = use_vector_correction
        self.num_features = len(self.population[0])
        self.ticker: int = 0
        self.correction_factor = 0.5 / np.sqrt(self.num_features / 3)

    def step(self) -> None:
        for agent_id in range(len(self.population)):
            a = 2 - self.ticker * (2 / self.steps)
            r = (
                np.random.random(self.num_features)
                if self.use_vector_correction
                else np.random.random()
            )
            A = (2 * a * r) - a
            C = 2 * r
            l = (np.random.random() * 2) - 1
            p = np.random.random()
            b = 1

            if p < 0.5:
                A_norm = (
                    np.linalg.norm(A) * self.correction_factor
                    if self.use_vector_correction
                    else np.abs(A)
                )

                if A_norm < 1:
                    prey: Whale = self.population.prey
                    D = np.abs(C * prey.features - self.population[agent_id].features)
                    self.population[agent_id] = prey.features - A * D
                elif A_norm >= 1:
                    random_agent_id: int = np.random.randint(0, len(self.population))
                    random_agent: Whale = self.population[random_agent_id]
                    D = np.abs(
                        C * random_agent.features + self.population[agent_id].features
                    )
                    self.population[agent_id] = random_agent.features - A * D

            elif p >= 0.5:
                prey: Whale = self.population.prey
                D = np.abs(C * prey.features - self.population[agent_id].features)
                self.population[agent_id] = (
                    D * np.exp(b * l) + np.cos(l * 2 * np.pi) + prey.features
                )

            self.update_scaler(agent_id)
            self.enforce_search_space_limits(agent_id)

        self.tick()

    def update_scaler(self, agent_id: int) -> None:
        features = self.population[agent_id].features
        self.scaler.update(features)

    def enforce_search_space_limits(self, agent_id: int) -> None:
        features = self.population[agent_id].features
        features = self.scaler.scale(features)

        temperature = self.temperature.get(iteration=self.ticker)
        values = self.transition_fn(features) * temperature
        new_features = (np.random.random(self.num_features) < values).astype(int)

        if sum(new_features) == 0:
            # Activation-based recovery
            if self.activate_best_if_empty:
                new_features[np.argmax(values)] = 1
            else:
                # Random recovery
                index = np.random.choice(values.shape[0], 1, replace=False)
                new_features[index] = 1

        self.population[agent_id] = new_features

    def tick(self) -> None:
        self.ticker += 1
