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
from src.woa.scaling import PassTroughScaler
from src.woa.scenario import WhaleOptimizationAlgorithmScenario
from src.woa.temperature import ConstTemperature
from src.woa.transition import sigmoid_transition_fn


def main() -> None:
    # SETTINGS

    woa_steps = [10, 30, 50]
    woa_sizes = [1, 5, 10, 20, 30]

    fe_libs = ["catch22", "tsfel", "tsfresh"]
    data_dirs = [
        "data/polcom/2022/Y/offline",
    ]
    results_output_file = "data/results/raw/vanilla_woa_results.csv"
    codecarbon_output_file = "data/results/raw/vanilla_woa_codecarbon.csv"

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

                    # 5. Loop over objective functions
                    for objective_fn in objective_fns:

                        # 6. Loop over steps
                        for steps in woa_steps:

                            # 7. Loop over population size
                            for size in woa_sizes:

                                metadata = {
                                    "data_dir": data_dir,
                                    "extractor": extractor,
                                    "vm": vm_name,
                                    "objective_fn": objective_fn.__name__,
                                    "target_col": y_train.name,
                                    "num_features": len(df_train.columns.tolist()),
                                }

                                transition_fn = sigmoid_transition_fn  # vanilla setting
                                temperature = ConstTemperature(
                                    1.0, 1.0, steps
                                )  # vanilla setting
                                scaler = PassTroughScaler()  # vanilla setting
                                activate_best_if_empty = False  # vanilla setting
                                use_vector_correction = False  # vanilla setting
                                population_init_method = "random"  # vanilla setting

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
                                    "fs_method": "VanillaWOA",
                                    "fs_method_kwargs": {
                                        "steps": steps,
                                        "size": size,
                                        "transition_fn": transition_fn.__name__,
                                        "temperature": type(temperature).__name__,
                                        "scaler": type(scaler).__name__,
                                        "activate_best_if_empty": activate_best_if_empty,
                                        "use_vector_correction": use_vector_correction,
                                        "population_init_method": population_init_method,
                                    },
                                    "n_selected_features": sum(best_result.features),
                                    "score": best_result.fitness,
                                    "time": tac - tic,
                                    "codecarbon_task_id": task_id,
                                }

                                results.append(result | metadata)
                                pd.DataFrame(results).to_csv(results_output_file)

                                _id += 1


if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    ensure_reproducibility(SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
