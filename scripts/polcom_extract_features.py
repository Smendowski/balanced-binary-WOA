import os
import warnings
from collections.abc import Callable
from pathlib import Path

from src.config import (
    POLCOM_2022_Y_DYNAMIC_CATCH22_PATH,
    POLCOM_2022_Y_DYNAMIC_TSFEL_PATH,
    POLCOM_2022_Y_DYNAMIC_TSFRESH_PATH,
    POLCOM_2022_Y_PATHS,
    POLCOM_2022_Y_TRAIN_END_DATE,
    SEED,
)
from src.features.extraction import (
    catch22_extract_features,
    extract_dynamic_features,
    tsfel_extract_features,
    tsfresh_extract_features,
)
from src.utils import ensure_reproducibility


def main() -> None:
    extractors: list[Callable] = [
        tsfresh_extract_features,
        tsfel_extract_features,
        catch22_extract_features,
    ]

    polcom_2022_Y_dest_paths: list[Path] = [
        POLCOM_2022_Y_DYNAMIC_CATCH22_PATH,
        POLCOM_2022_Y_DYNAMIC_TSFRESH_PATH,
        POLCOM_2022_Y_DYNAMIC_TSFEL_PATH,
    ]
    print("Extracting features for POLCOM 2022 Y data...")
    for extractor, dst_path in zip(extractors, polcom_2022_Y_dest_paths):
        print(f"Using {extractor.__name__} writing to {dst_path}")
        extract_dynamic_features(
            src_paths=POLCOM_2022_Y_PATHS,
            dst_path=dst_path,
            extractor=extractor,
            train_end_date=POLCOM_2022_Y_TRAIN_END_DATE,
            ref_columns=["CPU_USAGE_PERCENT", "MEMORY_USAGE_PERCENT"],
            window_size=7,
            save_results=True,
        )
    print("Finished extraction for POLCOM 2022 Y data!")


if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    ensure_reproducibility(SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
