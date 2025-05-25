import os
from typing import Dict, List, Union

import pandas as pd

# TODO Validate all inputs


def futures_readin_bind(files: List[os.PathLike]) -> pd.DataFrame:
    """Read in multiple historical market data CSV files from investing.com and bind them into a single DataFrame."""

    dfs = []
    for file in files:
        dfs.append(pd.read_csv(file))
    merged_df = pd.concat(dfs)

    # Convert date from string to DT object and duplicate dates
    merged_df["Date"] = merged_df["Date"].astype("datetime64[ns]")
    merged_df = merged_df.drop_duplicates(subset=["Date"])

    # K is for thousands, M for millions, B for billions
    suffix_multipliers = {"K": 1000, "M": 1000000, "B": 1000000000, "T": 1000000000000}

    # Remove commas from numeric columns and convert to float
    # Clean numeric columns: handle suffixs, commas, percent signs
    numeric_columns = ["Price", "Open", "High", "Low", "Vol.", "Change %"]

    def clean_numeric(x):
        if pd.isna(x):
            return x

        x_str = str(x).replace(",", "").replace("%", "")

        for suffix, multiplier in suffix_multipliers.items():
            if x_str.endswith(suffix):
                try:
                    return float(x_str[: -len(suffix)]) * multiplier
                except Exception:
                    return x
        try:
            return float(x_str)
        except Exception:
            return x

    merged_df[numeric_columns] = merged_df[numeric_columns].map(
        clean_numeric, na_action="ignore"
    )

    merged_df = merged_df.sort_values(by="Date")
    merged_df = merged_df.reset_index(drop=True)
    pivoted_df = merged_df.pivot_table(index="Date")
    pivoted_df = pivoted_df.reset_index()

    return pivoted_df


def load_commodity_futures_by_folder(
    root_dir: Union[str, os.PathLike]
) -> Dict[str, pd.DataFrame]:
    """Load all csv datasets from a root directory provided as thier own independedent data frames storeded in a dictionary with the folder name as the key"""

    data_dict = {}
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            csv_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(".csv")
            ]
            if csv_files:
                try:
                    df = futures_readin_bind(csv_files)
                    data_dict[folder] = df
                except Exception as e:
                    print(f"Failed to process {folder}: {e}")
    return data_dict


# TODO change to google doc string format, add custom fill method support via switch statement in a function call maintain track_nans for now; probs remove eventually
def concat_all_data(
    data: Dict[str, pd.DataFrame], fill_method: str = "ffill", track_nans: bool = True
) -> pd.DataFrame:
    # Validate inputs
    if not data:
        raise ValueError("Input data dictionary cannot be empty")
    for name, df in data.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Dataset '{name}' must be a pandas DataFrame")
        if "Date" not in df.columns:
            raise ValueError(f"Dataset '{name}' missing required 'Date' column")
        if df.empty:
            raise ValueError(f"Dataset '{name}' cannot be empty")

    # Create index range
    overall_min = min([df["Date"].min() for df in data.values()])
    overall_max = max([df["Date"].max() for df in data.values()])
    master_index = pd.date_range(start=overall_min, end=overall_max, freq="D")

    result_df = pd.DataFrame(index=master_index)
    result_df.index.name = "Date"

    for name, df in data.items():
        print(f"Processing dataset: {name}")

        df_copy = df.copy()
        df_copy["Date"] = pd.to_datetime(df_copy["Date"])
        df_copy = df_copy.drop_duplicates(subset=["Date"], keep="last")
        df_copy.set_index("Date", inplace=True)

        # Handle NaN tracking based on parameter
        if track_nans:
            nan_indicators = df_copy.isna().astype(int)
            nan_indicators = nan_indicators.add_suffix("_was_nan")
            df_copy = df_copy.fillna(0)
            df_copy = pd.concat([df_copy, nan_indicators], axis=1)
        else:
            df_copy = df_copy.fillna(0)

        # Reindex to master timeline
        df_copy = df_copy.reindex(master_index)
        df_copy.columns = [f"{name}_{col}" for col in df_copy.columns]
        result_df = result_df.join(df_copy)

    # Reset index to make Date a column
    result_df.reset_index(inplace=True)

    return result_df


commodity_futures = load_commodity_futures_by_folder(r"data\raw\commodity_data\daily")
for commodity, df in commodity_futures.items():
    print(f"{commodity}: {df.shape[0]} rows loaded")
    print(df.head())

concat_data = concat_all_data(commodity_futures)

print(concat_data.shape)
print(concat_data.head())
print(set(concat_data.columns))
