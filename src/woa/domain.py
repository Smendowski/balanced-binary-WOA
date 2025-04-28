import random
from dataclasses import dataclass, field

import numpy as np

from .initialize import initialize_agents_quasirandom, initialize_agents_random


@dataclass
class Whale:
    agent_id: int
    features: np.ndarray = field(init=False)
    fitness: float = field(init=False)

    def initialize(self, features: np.ndarray) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)


@dataclass
class Population:
    size: int
    agents: list[Whale] = field(init=False, default_factory=list)
    prey_id: int = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.agents = [Whale(agent_id=agent_id) for agent_id in range(self.size)]

    @property
    def prey(self) -> Whale:
        return self.__getitem__(self.prey_id)

    def __len__(self) -> int:
        return len(self.agents)

    def __getitem__(self, agent_id: int) -> Whale:
        return self.agents[agent_id]

    def __setitem__(self, key: int, features: np.ndarray) -> None:
        self.agents[key].features = features

    def initialize(
        self,
        min_features: int,
        max_features: int,
        total_features: int,
        method: str = "sobol",
    ) -> None:

        if method == "random":
            agents_features = initialize_agents_random(
                num_agents=len(self),
                num_features=total_features,
                k_min=min_features,
                k_max=max_features,
            )
        elif method == "halton" or method == "sobol":
            agents_features = initialize_agents_quasirandom(
                num_agents=len(self),
                num_features=total_features,
                method=method,
                k_min=min_features,
                k_max=max_features,
            )
        else:
            raise ValueError(
                "Invalid method. Choose one of 'random', 'sobol' or 'halton'."
            )

        for agent, features in zip(self.agents, agents_features):
            agent.initialize(features)

    def select_prey(self) -> None:
        if not self.prey_id:
            best_fitness: float = np.inf
        else:
            best_fitness: float = self.prey.fitness

        for agent in self.agents:
            if agent.fitness < best_fitness:
                best_fitness = agent.fitness
                prey_id = agent.agent_id

                self.prey_id = prey_id
