import pandas as pd


def generate_overlapping_windows(
    df: pd.DataFrame, window_size: int, step_size: int = 1
) -> pd.DataFrame:
    windows: list = []
    for start_idx in range(0, len(df) - window_size + 1, step_size):
        stop_idx = start_idx + window_size
        window_id: int = start_idx // step_size

        window: pd.DataFrame = df.iloc[start_idx:stop_idx].copy()
        window["WINDOW_ID"] = window_id
        windows.append(window)

    return pd.concat(windows)
