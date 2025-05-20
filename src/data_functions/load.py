import os
from typing import Dict, List, Union

import pandas as pd


def futures_readin_bind(files: List[Union[str, os.PathLike]]) -> pd.DataFrame:
    dfs = []
    for file in files:
        dfs.append(pd.read_csv(file))
    merged_df = pd.concat(dfs)

    # Convert date from string to DT object and duplicate dates
    merged_df["Date"] = merged_df["Date"].astype("datetime64[ns]")
    merged_df = merged_df.drop_duplicates(subset=["Date"])

    # K is for thousands, M for millions, B for billions
    suffix_multipliers = {"K": 1000, "M": 1000000, "B": 1000000000}

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


commodity_futures = load_commodity_futures_by_folder(r"data\raw\commodity_data\daily")
for commodity, df in commodity_futures.items():
    print(f"{commodity}: {df.shape[0]} rows loaded")
    print(df.head())
