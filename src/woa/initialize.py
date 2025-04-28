import numpy as np
from scipy.stats.qmc import Halton, Sobol


def initialize_agents_random(
    num_agents: int,
    num_features: int,
    k_min: int = 1,
    k_max: int = None,
):
    """
    Initializes agents with randomly selected features.

    Args:
        num_agents (int): Number of agents.
        num_features (int): Total number of features.
        k_min (int, optional): Minimum number of selected features per agent. Default is 1.
        k_max (int, optional): Maximum number of selected features per agent. If None, it is set to num_features.

    Returns:
        np.ndarray: A binary matrix of shape (num_agents, num_features) where 1 indicates a selected feature.
    """
    if k_max is None:
        k_max = num_features

    agents = np.zeros((num_agents, num_features), dtype=int)

    for i in range(num_agents):
        # Random number of selected features
        num_active_features = np.random.randint(k_min, k_max + 1)
        selected_features = np.random.choice(
            num_features, num_active_features, replace=False
        )
        agents[i, selected_features] = 1

    return agents


def initialize_agents_quasirandom(
    num_agents: int,
    num_features: int,
    method: str = "sobol",
    k_min: int = 1,
    k_max: int = None,
):
    """
    Initializes agents using quasi-random sequences (Sobol or Halton) to select features.

    When to use Halton vs Sobol:
        - Use **Halton** when the number of features (dimensions) is **small to moderate**.
        - Use **Sobol** when the number of features is **large** (higher-dimensional space),
          as it provides better uniformity and avoids correlation issues.

    Args:
        num_agents (int): Number of agents.
        num_features (int): Total number of features.
        method (str): Quasi-random sequence method ("sobol" or "halton"). Defaults to Sobol method.
        k_min (int, optional): Minimum number of selected features per agent. Default is 1.
        k_max (int, optional): Maximum number of selected features per agent. If None, it is set to num_features.

    Returns:
        np.ndarray: A binary matrix of shape (num_agents, num_features) where 1 indicates a selected feature.
    """
    if k_max is None:
        k_max = num_features

    # Select quasi-random generator
    if method == "sobol":
        engine = Sobol(d=num_features, scramble=True)
    elif method == "halton":
        engine = Halton(d=num_features, scramble=True)
    else:
        raise ValueError("Invalid method. Choose 'sobol' or 'halton'.")

    # Generate quasi-random samples
    sample = engine.random(num_agents)
    agents = np.zeros((num_agents, num_features), dtype=int)

    for i in range(num_agents):
        # Random number of selected features
        num_active_features = np.random.randint(k_min, k_max + 1)
        # Pick features with highest values
        selected_features = np.argsort(sample[i])[-num_active_features:]
        agents[i, selected_features] = 1

    return agents
