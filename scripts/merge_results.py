import numpy as np
import pandas as pd

from src.utils import parse_codecarbon_stats

method = None

skip_cols = ["task_id", "codecarbon_task_id"]

if method == "LowerBound":
    df_r = pd.read_csv("data/results/merged/lower_bound_results.csv", index_col=0)
    df_c = parse_codecarbon_stats(
        pd.read_csv("data/results/merged/lower_bound_codecarbon.csv")
    )
    df = pd.concat([df_r, df_c], axis=1)
    df = df[[col for col in df.columns if col not in skip_cols]]
    df["score"] = np.sqrt(df["score"])  # switch to RMSE from MSE
    df.to_csv("data/results/merged/lower_bound.csv")


if method == "UpperBound":
    df_r = pd.read_csv("data/results/merged/upper_bound_results.csv", index_col=0)
    df_c = parse_codecarbon_stats(
        pd.read_csv("data/results/merged/upper_bound_codecarbon.csv")
    )
    df = pd.concat([df_r, df_c], axis=1)
    df = df[[col for col in df.columns if col not in skip_cols]]
    df["score"] = np.sqrt(df["score"])  # switch to RMSE from MSE
    df.to_csv("data/results/merged/upper_bound.csv")


if method == "FilterBased":
    df_r = pd.read_csv("data/results/merged/filter_based_results.csv", index_col=0)
    df_c = parse_codecarbon_stats(
        pd.read_csv("data/results/merged/filter_based_codecarbon.csv")
    )
    df = pd.concat([df_r, df_c], axis=1)
    df = df[[col for col in df.columns if col not in skip_cols]]
    df["score"] = np.sqrt(df["score"])  # switch to RMSE from MSE
    df.to_csv("data/results/merged/filter_based.csv")


if method == "GeneticAlgorithm":
    df_r = pd.read_csv("data/results/merged/genetic_algorithm_results.csv")
    df_c = parse_codecarbon_stats(
        pd.read_csv("data/results/merged/genetic_algorithm_codecarbon.csv")
    )
    df = pd.concat([df_r, df_c], axis=1)
    del df["Unnamed: 0"]
    df = df[[col for col in df.columns if col not in skip_cols]]
    df["score"] = np.sqrt(df["score"])  # switch to RMSE from MSE
    df.to_csv("data/results/merged/genetic_algorithm.csv")


if method == "VanillaWOA":
    df_r = pd.read_csv("data/results/merged/vanilla_woa_results.csv", index_col=0)
    df_c = parse_codecarbon_stats(
        pd.read_csv("data/results/merged/vanilla_woa_codecarbon.csv")
    )
    df = pd.concat([df_r, df_c], axis=1)
    df = df[[col for col in df.columns if col not in skip_cols]]
    df["score"] = np.sqrt(df["score"])  # switch to RMSE from MSE
    df.to_csv("data/results/merged/vanilla_woa.csv")


if method == "RandomSearch":
    df_r = pd.read_csv("data/results/merged/random_search_results.csv")
    df_c = parse_codecarbon_stats(
        pd.read_csv("data/results/merged/random_search_codecarbon.csv")
    )
    df = pd.concat([df_r, df_c], axis=1)
    del df["Unnamed: 0"]
    df = df[[col for col in df.columns if col not in skip_cols]]
    df["score"] = np.sqrt(df["score"])  # switch to RMSE from MSE
    df.to_csv("data/results/merged/random_search.csv")


if method == "BalancedWOA":
    df_r = pd.read_csv("data/results/merged/balanced_woa_results.csv")
    df_c = parse_codecarbon_stats(
        pd.read_csv("data/results/merged/balanced_woa_codecarbon.csv")
    )
    df = pd.concat([df_r, df_c], axis=1)
    del df["Unnamed: 0"]
    df = df[[col for col in df.columns if col not in skip_cols]]
    df["score"] = np.sqrt(df["score"])  # switch to RMSE from MSE
    df.to_csv("data/results/merged/balanced_woa.csv")
