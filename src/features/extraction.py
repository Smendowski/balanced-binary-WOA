import os
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp
from pycatch22 import catch22_all
from tqdm import tqdm
from tsfel import get_features_by_domain, time_series_features_extractor
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute

from src.features.utils import generate_overlapping_windows


def extract_dynamic_features(
    src_paths: dict[str, Path],
    dst_path: Path,
    extractor: Callable,
    train_end_date: Timestamp,
    ref_columns: list[str],
    window_size: int,
    save_results: bool = False,
) -> None:
    os.makedirs(dst_path, exist_ok=True)
    for vm_id, src_path in tqdm(src_paths.items()):
        df = pd.read_parquet(src_path)
        df.index = pd.to_datetime(df.index)

        df_train = df[df.index <= train_end_date].copy()
        df_test = df[df.index > train_end_date].copy()

        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        for ref_column in ref_columns:
            df_train.loc[:, f"TARGET_{ref_column}"] = df_train[ref_column].shift(-1)
            _df_train_features = extractor(
                df_train, ref_column=ref_column, window_size=window_size
            )
            df_train_features = pd.concat(
                [df_train_features, _df_train_features], axis=1
            )

            df_test.loc[:, f"TARGET_{ref_column}"] = df_test[ref_column].shift(-1)
            _df_test_features = extractor(
                df_test, ref_column=ref_column, window_size=window_size
            )
            df_test_features = pd.concat([df_test_features, _df_test_features], axis=1)

        df_train_features = pd.concat(
            [df_train[(window_size - 1) :], df_train_features], axis=1
        )
        df_train_features.dropna(inplace=True, axis=0, how="all")
        df_train_features = df_train_features.bfill()
        assert len(df_train_features) == len(df_train[(window_size - 1) :])
        if save_results:
            df_train_features.to_parquet(dst_path / f"{vm_id}_TRAIN.parquet")

        df_test_features = pd.concat(
            [df_test[(window_size - 1) :], df_test_features], axis=1
        )
        df_test_features.dropna(inplace=True, axis=0, how="all")
        df_test_features = df_test_features.bfill()
        assert len(df_test_features) == len(df_test[(window_size - 1) :])
        if save_results:
            df_test_features.to_parquet(dst_path / f"{vm_id}_TEST.parquet")


def tsfresh_extract_features(
    df: pd.DataFrame, ref_column: str, window_size: int, verbose: bool = False
) -> pd.DataFrame:
    _df: pd.DataFrame = df[[ref_column]].copy()
    if "TIMESTAMP" not in _df.columns:
        _df["TIMESTAMP"] = df.index

    _df = generate_overlapping_windows(_df, window_size=window_size)

    df_features = extract_features(
        _df,
        column_id="WINDOW_ID",
        column_sort="TIMESTAMP",
        default_fc_parameters=ComprehensiveFCParameters(),
        impute_function=impute,
        disable_progressbar=not verbose,
    )
    df_features.columns = df_features.columns.map(
        lambda col: f"{col.replace("__", "_").upper()}_WINDOW_SIZE_{window_size}"
    )
    df_features.index = df[(window_size - 1) :].index

    return df_features


def tsfel_extract_features(
    df: pd.DataFrame, ref_column: str, window_size: int, verbose: bool = False
) -> pd.DataFrame:
    _df: pd.DataFrame = df[[ref_column]].copy()

    cfg = get_features_by_domain()
    del cfg["spectral"]["LPCC"]

    df_features = time_series_features_extractor(
        cfg,
        _df,
        overlap=(window_size - 1) / window_size,
        window_size=window_size,
        verbose=int(verbose),
    )

    df_features.columns = df_features.columns.map(
        lambda col: f"{col.replace("__", "_").replace(" ", "_").upper()}_WINDOW_SIZE_{window_size}"
    )
    df_features.index = df[(window_size - 1) :].index

    return df_features


def catch22_extract_features(
    df: pd.DataFrame, ref_column: str, window_size: int, verbose: bool = False
) -> pd.DataFrame:
    _df: pd.DataFrame = df[[ref_column]].copy()
    _df = generate_overlapping_windows(_df, window_size=window_size)

    indices: list[date] = []
    extracts: list[dict[str, float]] = []

    for window_id in range(0, _df["WINDOW_ID"].max() + 1):
        sub_df = _df[_df["WINDOW_ID"] == window_id][[ref_column]]
        indices.append(sub_df.index[-1])
        results = catch22_all(
            data=np.ravel(sub_df.values).tolist(), catch24=True, short_names=False
        )

        extract = {}
        for feature, value in zip(results["names"], results["values"]):
            extract[f"{ref_column}_{feature}_WINDOW_SIZE_{window_size}"] = value
        extracts.append(extract)

    df_features = pd.DataFrame(extracts)
    df_features.index = indices
    df_features.columns = map(
        lambda col: f"{col.replace("__", "_").replace(" ", "_").upper()}",
        df_features.columns,
    )

    return df_features
