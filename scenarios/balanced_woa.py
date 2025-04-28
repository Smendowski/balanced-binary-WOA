import ast
import os
import time
import warnings
from pathlib import Path

import pandas as pd
from codecarbon import EmissionsTracker

from src.config import SEED
from src.utils import ensure_reproducibility
from src.woa.algorithm import WhaleOptimizationAlgorithm
from src.woa.domain import Population
from src.woa.objective import mse_objective_fn
from src.woa.scaling import (
    GlobalZScoreScaler,
    LocalZScoreScaler,
    PassTroughScaler,
    ProgressiveZScoreScaler,
)
from src.woa.scenario import WhaleOptimizationAlgorithmScenario
from src.woa.temperature import (
    ConstTemperature,
    LinearTemperatureDecay,
    PowerTemperatureDecay,
    SigmoidTemperatureDecay,
)
from src.woa.transition import sigmoid_transition_fn, tanh_transition_fn


def get_best_params(
    df: pd.DataFrame, extractor: str, vm: str, target_col: str
) -> tuple[int, int]:
    """
    Retrieves and parses the 'fs_method_kwargs' column for the 'steps' and 'size' parameters.

    Args:
        df (pd.DataFrame): DataFrame containing the results.
        extractor (str): Feature extractor name.
        vm (str): Virtual machine identifier.
        target_col (str): Target column name.

    Returns:
        tuple: Extracted values for 'steps' and 'size', or (50, 30) if missing.
    """
    best_rows = df.loc[df.groupby(["extractor", "vm", "target_col"])["score"].idxmin()]
    row = best_rows.loc[
        (df.extractor == extractor) & (df.vm == vm) & (df.target_col == target_col),
        "fs_method_kwargs",
    ].squeeze()
    best_params = ast.literal_eval(row)

    return best_params.get("steps", 50), best_params.get("size", 30)


def main() -> None:
    # SETTINGS

    fe_libs = ["catch22", "tsfel", "tsfresh"]
    data_dirs = [
        "data/polcom/2022/Y/offline",
    ]
    results_output_file = "data/results/raw/balanced_woa_results.csv"
    codecarbon_output_file = "data/results/raw/balanced_woa_codecarbon.csv"

    vanilla_woa_results_file = "data/results/vanilla_woa_results.csv"
    vanilla_woa_results_df = pd.read_csv(vanilla_woa_results_file)

    objective_fns = [mse_objective_fn]
    results = []

    # EXPERIMENT

    # 1. Loop over feature extraction libraries
    _id: int = 0
    for extractor in fe_libs:
        print(f"Feature extracting library: {extractor}")

        # 2. Loop over datasets
        for data_dir in data_dirs:
            data_paths = [
                file
                for file in os.listdir(Path(data_dir) / extractor)
                if file.endswith(".parquet")
            ]
            vm_nums = list(set([int(path[2:4]) for path in data_paths]))

            # 3. Loop over virtual machines
            for vm_num in vm_nums:
                print(f"\tVM {vm_num} from {data_dir}")
                vm_name = f"VM{vm_num:02d}"

                # 4. Loop over contexts
                for context in ["CPU_USAGE_PERCENT", "MEMORY_USAGE_PERCENT"]:
                    redundant = ""
                    if context == "CPU_USAGE_PERCENT":
                        redundant = "MEMORY_USAGE_PERCENT"
                    if context == "MEMORY_USAGE_PERCENT":
                        redundant = "CPU_USAGE_PERCENT"

                    df_train: pd.DataFrame = pd.read_parquet(
                        Path(data_dir) / extractor / (vm_name + "_TRAIN.parquet")
                    )
                    df_test: pd.DataFrame = pd.read_parquet(
                        Path(data_dir) / extractor / (vm_name + "_TEST.parquet")
                    )

                    df_train = df_train.interpolate()
                    df_test = df_test.interpolate()

                    df_train.pop(f"TARGET_{redundant}")
                    df_test.pop(f"TARGET_{redundant}")

                    y_train = df_train.pop(f"TARGET_{context}")
                    y_test = df_test.pop(f"TARGET_{context}")

                    df_train = df_train.loc[
                        :, ~df_train.columns.str.startswith(f"{redundant}_")
                    ]
                    df_test = df_test.loc[
                        :, ~df_test.columns.str.startswith(f"{redundant}_")
                    ]
                    features = df_train.columns.tolist()
                    num_features = len(features)

                    steps, size = get_best_params(
                        df=vanilla_woa_results_df,
                        extractor=extractor,
                        vm=vm_name,
                        target_col=f"TARGET_{context}",
                    )

                    # 5. Loop over objective functions
                    for objective_fn in objective_fns:

                        # 6. Loop over population init methods
                        for population_init_method in [
                            "random",
                            "sobol",
                            # "halton",
                        ]:

                            # 7. Loop over use vector correction options
                            for use_vector_correction in [False, True]:

                                # 8. Loop over activate best if empty options
                                for activate_best_if_empty in [False, True]:

                                    transition_fns = [
                                        sigmoid_transition_fn,
                                        tanh_transition_fn,
                                    ]
                                    scalers = [
                                        PassTroughScaler,
                                        LocalZScoreScaler,
                                        GlobalZScoreScaler,
                                        ProgressiveZScoreScaler,
                                    ]
                                    temperatures = [
                                        ConstTemperature,
                                        LinearTemperatureDecay,
                                        SigmoidTemperatureDecay,
                                        PowerTemperatureDecay,
                                    ]

                                    triples = [
                                        (
                                            transition_fn,
                                            scaler(),
                                            (
                                                temperature(
                                                    min_temp=0.5,
                                                    max_temp=1.0,
                                                    max_iterations=steps,
                                                )
                                                if temperature
                                                is not PowerTemperatureDecay
                                                else temperature(
                                                    min_temp=0.5,
                                                    max_temp=1.0,
                                                    max_iterations=steps,
                                                    beta=beta,
                                                )
                                            ),
                                        )
                                        for transition_fn in transition_fns
                                        for scaler in scalers
                                        for temperature in temperatures
                                        for beta in [2.5]
                                    ]

                                    # 9. Loop over transition_fn, scaler and temperature combinations
                                    for (
                                        transition_fn,
                                        scaler,
                                        temperature,
                                    ) in triples:

                                        metadata = {
                                            "data_dir": data_dir,
                                            "extractor": extractor,
                                            "vm": vm_name,
                                            "objective_fn": objective_fn.__name__,
                                            "target_col": y_train.name,
                                            "num_features": len(
                                                df_train.columns.tolist()
                                            ),
                                        }

                                        task_id = f"task_{_id}"
                                        tracker = EmissionsTracker(
                                            project_name=task_id,
                                            output_file=codecarbon_output_file,
                                        )

                                        population = Population(size=size)
                                        population.initialize(
                                            min_features=1,
                                            max_features=num_features,
                                            total_features=num_features,
                                            method=population_init_method,
                                        )

                                        algo = WhaleOptimizationAlgorithm(
                                            steps=steps,
                                            population=population,
                                            transition_fn=transition_fn,
                                            scaler=scaler,
                                            temperature=temperature,
                                            activate_best_if_empty=activate_best_if_empty,
                                            use_vector_correction=use_vector_correction,
                                        )

                                        scenario = WhaleOptimizationAlgorithmScenario(
                                            X_train=df_train,
                                            y_train=y_train,
                                            X_test=df_test,
                                            y_test=y_test,
                                            steps=steps,
                                            population=population,
                                            algo=algo,
                                            objective_fn=mse_objective_fn,
                                        )

                                        tracker.start()
                                        tic = time.time()
                                        scenario.run()
                                        tac = time.time()
                                        tracker.stop()

                                        best_result = scenario.population.prey

                                        result = {
                                            "fs_method": "BalancedWOA",
                                            "fs_method_kwargs": {
                                                "steps": steps,
                                                "size": size,
                                                "transition_fn": transition_fn.__name__,
                                                "temperature": type(
                                                    temperature
                                                ).__name__,
                                                "scaler": type(scaler).__name__,
                                                "activate_best_if_empty": activate_best_if_empty,
                                                "use_vector_correction": use_vector_correction,
                                                "population_init_method": population_init_method,
                                            },
                                            "n_selected_features": sum(
                                                best_result.features
                                            ),
                                            "score": best_result.fitness,
                                            "time": tac - tic,
                                            "codecarbon_task_id": task_id,
                                        }

                                        results.append(result | metadata)
                                        pd.DataFrame(results).to_csv(
                                            results_output_file
                                        )

                                        _id += 1


if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    ensure_reproducibility(SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
