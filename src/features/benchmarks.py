import itertools
import random
from collections.abc import Callable

import numpy as np
import pandas as pd
from pygad import GA
from tqdm import tqdm, trange
from tsfresh import select_features


def grid_search(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    objective_fn: Callable,
    verbose: bool = True,
) -> pd.Series:

    features = X_train.columns.tolist()
    num_features = len(features)

    results = []

    combinations = itertools.product([0, 1], repeat=num_features)

    for subset in tqdm(
        combinations,
        total=2**num_features,
        disable=not verbose,
    ):
        score = objective_fn(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            selected_features=np.array(subset),
        )
        results.append({"features": subset, "score": score})

    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df["score"].idxmin()]

    return best_result


def random_search(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    num_samples: int,
    objective_fn: Callable,
    verbose: bool = True,
) -> pd.Series:

    features = X_train.columns.tolist()
    num_features = len(features)

    results = []

    for _ in trange(num_samples, disable=not verbose):
        subset = np.zeros(num_features, dtype=int)
        k = random.randint(1, num_features)
        mask = random.sample(range(num_features), k)
        subset[mask] = 1
        score = objective_fn(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            selected_features=np.array(subset),
        )
        results.append({"features": subset, "score": score})

    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df["score"].idxmin()]

    return best_result


def tsfresh_select_features(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    objective_fn: Callable,
    n_jobs: int = 1,
) -> pd.Series:
    df_selected = select_features(X_train, y_train, n_jobs=n_jobs)
    selected_features = [int(col in df_selected.columns) for col in X_train.columns]
    score = objective_fn(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        selected_features=selected_features,
    )

    results_df = pd.DataFrame([{"features": selected_features, "score": score}])
    best_result = results_df.loc[results_df["score"].idxmin()]

    return best_result


def ga_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_evaluations: int,
    objective_fn: Callable,
    pop_size: int = 20,
    num_parents_mating: int = 8,
    mutation_rate: float = 0.1,
    verbose: bool = True,
) -> pd.Series:

    num_features = X_train.shape[1]
    num_generations = int(np.ceil(num_evaluations / pop_size))

    def fitness_func(_ga, solution, _population_index):
        selected_features = np.array(solution, dtype=int)
        if selected_features.sum() == 0:
            return -np.inf  # Penalize empty feature sets
        score = objective_fn(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            selected_features=selected_features,
        )
        return -score  # Negative because PyGAD maximizes

    ga_instance = GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        sol_per_pop=pop_size,
        num_genes=num_features,
        fitness_func=fitness_func,
        gene_type=int,
        gene_space=[0, 1],
        mutation_percent_genes=int(mutation_rate * 100),
        mutation_by_replacement=True,
        crossover_type="single_point",
        keep_parents=1,
        allow_duplicate_genes=False,
        stop_criteria=None,
        on_generation=lambda ga: (
            print(
                f"Gen {ga.generations_completed}/{num_generations} | Best: {-ga.best_solution()[1]:.5f}"
            )
            if verbose
            else None
        ),
    )

    ga_instance.run()

    best_solution, best_fitness, _ = ga_instance.best_solution()
    best_solution = np.array(best_solution, dtype=int)

    return pd.Series(
        {
            "features": best_solution,
            "score": -best_fitness,
        }
    )
