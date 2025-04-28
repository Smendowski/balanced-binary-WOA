import random

import numpy as np
import pandas as pd
import torch


def ensure_reproducibility(seed: int = 42) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)

    torch.use_deterministic_algorithms(mode=True)


def parse_codecarbon_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[
        [
            "timestamp",
            "project_name",
            "duration",
            "emissions_rate",
            "cpu_power",
            "ram_power",
            "cpu_energy",
            "ram_energy",
        ]
    ]

    df.loc[:, "energy_consumed"] = df["cpu_energy"] + df["ram_energy"]

    df.columns = [
        "timestamp",
        "task_id",
        "duration [s]",
        "emissions_rate [kg/s]",
        "cpu_power [W]",
        "ram_power [W]",
        "cpu_energy [kWh]",
        "ram_energy [kWh]",
        "energy_consumed [kWh]",
    ]

    return df
