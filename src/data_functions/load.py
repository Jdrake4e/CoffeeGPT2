"""Load and preprocess commodity futures data from investing.com CSV files.

This module contains functions to read multiple CSV files, concatenate them
into a single LazyFrame, and perform any basic cleaning needed before
feature engineering or modeling.
"""

from collections.abc import Sequence
from datetime import date
import os
from pathlib import Path
from typing import Any

import polars as pl

# TODO add function to load preprocessed data


def _futures_readin_bind(files: Sequence[os.PathLike[Any]]) -> pl.LazyFrame:
    """Read multiple historical market data CSVs and bind them into one DataFrame.

    Given a sequence of file-paths, this function:
        1. Loads each CSV as a Polars LazyFrame
        2. Concatenates them
        3. Returns the resulting LazyFrame for downstream processing
    """
    if not files:
        raise ValueError("File list cannot be empty. Cannot construct LazyFrame(s).")

    # Define static types for to prevent polars engine from
    # incorrectly inferring types by default
    numeric_columns = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
    dtype_overrides = dict.fromkeys(numeric_columns, pl.Utf8)

    lfs = [pl.scan_csv(file, schema_overrides=dtype_overrides) for file in files]
    concat_lf = pl.concat(lfs, how="vertical_relaxed")
    schema_names = concat_lf.collect_schema().names()  # Get schema names once
    for col_name in numeric_columns:
        if col_name not in schema_names:  # Use the collected names
            continue

    # Convert date from string to DT object and duplicate dates
    concat_lf = concat_lf.with_columns(
        pl.col("Date").str.to_datetime(format=r"%m/%d/%Y", strict=False).alias("Date")
    )
    concat_lf = concat_lf.unique(subset=["Date"], keep="first", maintain_order=True)

    # Remove commas from numeric columns and convert to float
    # Clean numeric columns: handle suffixs, commas, percent signs
    # K is for thousands, M for millions, B for billions, T for Trillions
    suffix_map = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

    transform_expressions = []

    for col_name in numeric_columns:
        # Check if the column exists in the LazyFrame's schema.
        # If not, skip trying to transform it to avoid planning errors.
        if col_name not in concat_lf.collect_schema().names():
            continue

        # Cast to String for robust cleaning. If already string, no change.
        # If numeric, converts to string. If null, stays null.
        # Cleans up symbols like commas and percent signs.
        original_value_expr = pl.col(col_name)
        string_expr = original_value_expr.cast(pl.Utf8, strict=False)
        cleaned_expr = string_expr.str.replace_all(",", "").str.replace_all("%", "")

        # Extract the numeric part (e.g., "123.45" from "123.45K").
        # Regex captures: optional minus, digits, optional decimal, digits.
        numeric_part_str = cleaned_expr.str.extract(r"^(-?\d*\.?\d+)", group_index=1)
        suffix_char_str = cleaned_expr.str.extract(r"([KMBT])$", group_index=1)

        # Build the multiplier expression based on the extracted suffix.
        multiplier_expr = pl.lit(1.0, dtype=pl.Float64)
        for char, val in list(suffix_map.items()):
            multiplier_expr = (
                pl.when(suffix_char_str == char)
                .then(pl.lit(val, dtype=pl.Float64))
                .otherwise(multiplier_expr)
            )

        # Cast the value if a suffix was present.
        value_with_suffix = (
            numeric_part_str.cast(pl.Float64, strict=False) * multiplier_expr
        )
        value_without_suffix = cleaned_expr.cast(pl.Float64, strict=False)

        # If the original value was null, keep it null.
        # If a suffix and a numeric part were successfully extracted
        # use value_with_suffix.
        # Otherwise (no suffix or invalid numeric part with suffix)
        # use value_without_suffix.
        final_col_expr = (
            pl.when(original_value_expr.is_null())
            .then(pl.lit(None, dtype=pl.Float64))  # Explicitly null of Float64 type
            .when(suffix_char_str.is_not_null() & numeric_part_str.is_not_null())
            .then(value_with_suffix)
            .otherwise(value_without_suffix)
            .alias(col_name)  # Ensures the transformed column keeps its original name.
        )
        transform_expressions.append(final_col_expr)

    # Apply expressions if generated
    if transform_expressions:
        concat_lf = concat_lf.with_columns(transform_expressions)
    concat_lf = concat_lf.sort("Date")

    return concat_lf


def load_commodity_futures_by_folder(root_dir: str) -> dict[str, pl.LazyFrame]:
    """Load each subfolder of CSVs as its own LazyFrame.

    Walks `root_dir`, finds all CSVs in each immediate subdirectory, and
    returns a dict mapping folder-name to a combined LazyFrame
    of all csv files in a directory.
    """
    try:
        path_obj = Path(root_dir).resolve(strict=True)
    except FileNotFoundError as e:
        raise ValueError(
            f"Provided path '{root_dir}' does not exist or is not accessible."
        ) from e
    except Exception as e:
        raise ValueError(f"Error processing path '{root_dir}': {e}") from e

    if not path_obj.is_dir():
        raise ValueError(f"Provided path '{path_obj}' is not a directory.")

    # TODO improve file tree traversal by building
    #      a tree of all csvs in sub directories
    data_dict = {}
    for item_path in path_obj.iterdir():
        if item_path.is_dir():
            csv_file_paths = [
                file_path
                for file_path in item_path.iterdir()
                if file_path.is_file() and file_path.suffix.lower() == ".csv"
            ]

            if csv_file_paths:
                folder_name = item_path.name
                lf = _futures_readin_bind(csv_file_paths)
                data_dict[folder_name] = lf
    return data_dict


def _validate_lazyframes(data: dict[str, pl.LazyFrame]) -> dict[str, pl.LazyFrame]:
    """Validate that all LazyFrames in the input dict are valid."""
    if not data:
        raise ValueError("Input data dictionary cannot be empty")

    valid_lazyframes = {}
    for name, lf in data.items():
        if not isinstance(lf, pl.LazyFrame):
            raise TypeError(f"Dataset '{name}' must be a polars LazyFrame")
        if "Date" not in lf.collect_schema().names():
            raise ValueError(f"Dataset '{name}' missing required 'Date' column")
        valid_lazyframes[name] = lf

    if not valid_lazyframes:  # If all items were filtered out due to some validation
        raise ValueError("No valid LazyFrames found in the input data dictionary.")
    return valid_lazyframes


def _determine_master_date_range(
    valid_lazyframes: dict[str, pl.LazyFrame],
) -> pl.LazyFrame:
    """Parse the overall date range from input LazyFrame(s).

    Create a master LazyFrame.
    """
    all_min_dates: list[date] = []
    all_max_dates: list[date] = []

    for _, lf_item in valid_lazyframes.items():
        min_max_lf = lf_item.select(
            [
                pl.min("Date").alias("min_date_val"),
                pl.max("Date").alias("max_date_val"),
            ]
        )
        # TODO when we add logging, wrap this block and catch specific errors
        #      I will also have continue in the block
        collected_dates_df = min_max_lf.collect()

        if not collected_dates_df.is_empty():
            min_val = collected_dates_df.item(0, "min_date_val")
            max_val = collected_dates_df.item(0, "max_date_val")

            if min_val is not None:
                all_min_dates.append(min_val)
            if max_val is not None:
                all_max_dates.append(max_val)

    if not all_min_dates or not all_max_dates:
        raise ValueError(
            "Could not determine an overall date range. "
            "All input LazyFrames might be empty or contain only null dates."
        )

    overall_min_date = min(all_min_dates)
    overall_max_date = max(all_max_dates)

    master_dates_df = pl.DataFrame(
        {
            "Date": pl.date_range(
                overall_min_date, overall_max_date, interval="1d", eager=True
            )
        }
    )

    # Note to self in future polars date_range does not output
    # a Datetime type but Date type instead
    return master_dates_df.with_columns(pl.col("Date").cast(pl.Datetime)).lazy()


def _join_lazyframes(
    valid_lazyframes: dict[str, pl.LazyFrame],
    result_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    # Loop through datasets, process, and join lazily
    for name, lf_orig in valid_lazyframes.items():
        current_lf = lf_orig.clone()

        expressions_to_select = [
            pl.col("Date")
        ]  # Always include the Date column for joining

        # Iterate over columns intended for data (all columns except 'Date')
        data_columns = [
            col for col in current_lf.collect_schema().names() if col != "Date"
        ]

        # TODO Move nan tracking to track_nans in process.py
        for col_name in data_columns:
            prefixed_col_name = f"{name}_{col_name}"
            data_col_expr = pl.col(col_name)
            expressions_to_select.append(data_col_expr.alias(prefixed_col_name))

        # Select and alias columns from the current commodity's LazyFrame
        # If current_lf might be empty after unique(), select could fail if no columns.
        # However, 'Date' is always there.
        if not expressions_to_select:  # Should not happen if 'Date' is always included
            continue

        lf_to_join = current_lf.select(expressions_to_select)
        result_lf = result_lf.join(lf_to_join, on="Date", how="left")
    return result_lf


# TODO change to google doc string format,
#      fill method will be handled outside later and not here
#      via switch statement in a function call in process.py
#      maintain track_nans for now; probs remove eventually
def concat_all_data(data: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Concatenate multiple LazyFrames into one over a master date range.

    Given a dict mapping names to Polars LazyFrames, this function:
        1. Unions all frames on their date column
        2. Resamples the result to the overall min -> max date span

    Args:
        data: Mapping of asset names to their respective LazyFrame.

    Returns:
        A single `pl.LazyFrame` covering the combined date index.

    """
    valid_lazyframes: dict[str, pl.LazyFrame] = _validate_lazyframes(data)
    master_lf: pl.LazyFrame = _determine_master_date_range(valid_lazyframes)
    result_lf: pl.LazyFrame = _join_lazyframes(valid_lazyframes, master_lf)

    return result_lf


if __name__ == "__main__":
    # DEBUGGING CODE: disable in prod
    commodity_futures = load_commodity_futures_by_folder(
        r"data\raw\commodity_data\daily"
    )
    concat_data_df = concat_all_data(commodity_futures).collect()

    print("\nCollected final DataFrame (returned by concat_all_data_lazy):")
    print(f"Shape of final DataFrame: {concat_data_df.shape}")
    print("Head of final DataFrame:")
    print(concat_data_df.head())
    print("\nSchema of final DataFrame:")
    print(concat_data_df.schema)

    null_counts_df = concat_data_df.null_count()
    print("\nDataFrame with Null Counts per Column (using df.null_count()):")
    print(null_counts_df)

    non_null_counts_df = concat_data_df.select(pl.all().count())
    print("\nnon null counts per column:")
    print(non_null_counts_df)
