import os
import time
import warnings
from pathlib import Path

import pandas as pd
from codecarbon import EmissionsTracker

from src.config import SEED
from src.features.benchmarks import ga_search
from src.utils import ensure_reproducibility
from src.woa.objective import mse_objective_fn


def main() -> None:
    fe_libs = ["catch22", "tsfel"]
    data_dirs = ["data/polcom/2022/Y/offline"]
    results_output_file = "data/results/raw/genetic_algorithm_results.csv"
    codecarbon_output_file = "data/results/raw/genetic_algorithm_codecarbon.csv"
    objective_fns = [mse_objective_fn]
    ga_num_evals = 1500
    pop_size = 30
    num_parents_mating = 8
    results = []

    _id: int = 0
    for extractor in fe_libs:
        print(f"Feature extracting library: {extractor}")
        for data_dir in data_dirs:
            data_paths = [
                fname
                for fname in os.listdir(Path(data_dir) / extractor)
                if fname.endswith(".parquet")
            ]
            vm_nums = list(set([int(path[2:4]) for path in data_paths]))

            for vm_num in vm_nums:
                print(f"\tVM {vm_num} from {data_dir}")
                vm_name = f"VM{vm_num:02d}"

                for context in ["CPU_USAGE_PERCENT", "MEMORY_USAGE_PERCENT"]:
                    redundant = (
                        "MEMORY_USAGE_PERCENT"
                        if context == "CPU_USAGE_PERCENT"
                        else "CPU_USAGE_PERCENT"
                    )

                    df_train = pd.read_parquet(
                        Path(data_dir) / extractor / (vm_name + "_TRAIN.parquet")
                    )
                    df_test = pd.read_parquet(
                        Path(data_dir) / extractor / (vm_name + "_TEST.parquet")
                    )

                    df_train = df_train.interpolate()
                    df_test = df_test.interpolate()

                    df_train.pop(f"TARGET_{redundant}")
                    df_test.pop(f"TARGET_{redundant}")
                    df_train = df_train.loc[
                        :, ~df_train.columns.str.startswith(f"{redundant}_")
                    ]
                    df_test = df_test.loc[
                        :, ~df_test.columns.str.startswith(f"{redundant}_")
                    ]

                    y_train = df_train.pop(f"TARGET_{context}")
                    y_test = df_test.pop(f"TARGET_{context}")

                    for objective_fn in objective_fns:
                        metadata = {
                            "data_dir": data_dir,
                            "extractor": extractor,
                            "vm": vm_name,
                            "objective_fn": objective_fn.__name__,
                            "target_col": y_train.name,
                            "num_features": len(df_train.columns.tolist()),
                        }

                        task_id = f"task_{_id}"
                        tracker = EmissionsTracker(
                            project_name=task_id, output_file=codecarbon_output_file
                        )

                        tracker.start()
                        tic = time.time()
                        best_result = ga_search(
                            X_train=df_train,
                            y_train=y_train,
                            X_test=df_test,
                            y_test=y_test,
                            objective_fn=objective_fn,
                            num_evaluations=ga_num_evals,
                            pop_size=pop_size,
                            num_parents_mating=num_parents_mating,
                        )
                        tac = time.time()
                        tracker.stop()

                        result = {
                            "fs_method": "GA",
                            "fs_method_kwargs": {
                                "num_evaluations": ga_num_evals,
                                "pop_size": pop_size,
                                "num_parents_mating": num_parents_mating,
                            },
                            "selected_features": best_result.features,
                            "n_selected_features": sum(best_result.features),
                            "score": best_result.score,
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
