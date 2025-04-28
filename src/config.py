from pathlib import Path

from pandas._libs.tslibs.timestamps import Timestamp

SEED: int = 42

POLCOM_2022_Y_PATH = Path("data") / "polcom" / "2022" / "Y"

POLCOM_2022_Y_TRAIN_END_DATE = Timestamp(
    year=2022, month=1, day=13, hour=0, minute=0, second=0
)

POLCOM_2022_Y_PATHS: dict[str, Path] = {
    f"VM{str(_id + 1).zfill(2)}": POLCOM_2022_Y_PATH
    / f"VM{str(_id + 1).zfill(2)}.parquet"
    for _id in range(8)
}

POLCOM_2022_Y_DYNAMIC_TSFRESH_PATH = POLCOM_2022_Y_PATH / "dynamic" / "tsfresh"
POLCOM_2022_Y_DYNAMIC_TSFEL_PATH = POLCOM_2022_Y_PATH / "dynamic" / "tsfel"
POLCOM_2022_Y_DYNAMIC_CATCH22_PATH = POLCOM_2022_Y_PATH / "dynamic" / "catch22"
