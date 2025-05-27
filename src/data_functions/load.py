import os
from datetime import date
from typing import Dict, List, Union

import polars as pl

# TODO Validate all inputs
# TODO add function to load preprocessed data


def futures_readin_bind(files: List[os.PathLike]) -> pl.LazyFrame:
    """Read in multiple historical market data CSV files from investing.com and bind them into a single DataFrame."""

    if not files:
        raise ValueError("File list cannot be empty. Cannot construct LazyFrame.")

    # Define static types for to prevent polars engine from incorrectly infering types by default
    numeric_columns = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
    dtype_overrides = {col: pl.Utf8 for col in numeric_columns}

    lfs = [pl.scan_csv(file, dtypes=dtype_overrides) for file in files]
    concat_lf = pl.concat(lfs, how="vertical_relaxed")
    schema_names = concat_lf.collect_schema().names()  # Get schema names once
    for col_name in numeric_columns:
        if col_name not in schema_names:  # Use the collected names
            continue

    # Convert date from string to DT object and duplicate dates
    # fmt: off
    concat_lf = concat_lf.with_columns(
        pl.col("Date")
        .str.to_datetime(format=r"%m/%d/%Y", strict=False)
        .alias("Date")
        )
    # fmt: on
    concat_lf = concat_lf.unique(subset=["Date"], keep="first", maintain_order=True)

    # Remove commas from numeric columns and convert to float
    # Clean numeric columns: handle suffixs, commas, percent signs
    # K is for thousands, M for millions, B for billions, T for Trillions
    suffix_map = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

    transform_expressions = []

    for col_name in numeric_columns:
        # Check if the column exists in the LazyFrame's schema.
        # If not, skip trying to transform it to avoid planning errors.
        if col_name not in concat_lf.columns:
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
        # fmt: off
        for char, val in list(suffix_map.items()):
            multiplier_expr = (
                pl.when(suffix_char_str == char)
                .then(pl.lit(val, dtype=pl.Float64))
                .otherwise(multiplier_expr)
            )
        # fmt: on

        # Cast the value if a suffix was present.
        value_with_suffix = (
            numeric_part_str.cast(pl.Float64, strict=False) * multiplier_expr
        )
        value_without_suffix = cleaned_expr.cast(pl.Float64, strict=False)

        # If the original value was null, keep it null.
        # If a suffix and a numeric part were successfully extracted, use value_with_suffix.
        # Otherwise (no suffix or invalid numeric part with suffix), try value_without_suffix.
        # fmt: off
        final_col_expr = (
            pl.when(original_value_expr.is_null())
            .then(pl.lit(None, dtype=pl.Float64)) # Explicitly null of Float64 type
            .when(suffix_char_str.is_not_null() & numeric_part_str.is_not_null())
            .then(value_with_suffix)
            .otherwise(value_without_suffix)
            .alias(col_name)  # Ensures the transformed column keeps its original name.
        )
        # fmt: on
        transform_expressions.append(final_col_expr)

    # Apply expressions if generated
    if transform_expressions:
        concat_lf = concat_lf.with_columns(transform_expressions)
    concat_lf = concat_lf.sort("Date")

    return concat_lf


def load_commodity_futures_by_folder(
    root_dir: Union[str, os.PathLike]
) -> Dict[str, pl.LazyFrame]:
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
                    lf = futures_readin_bind(csv_files)
                    data_dict[folder] = lf
                except Exception as e:
                    print(f"Failed to process {folder}: {e}")
    return data_dict


# TODO change to google doc string format, add custom fill method support via switch statement in a function call maintain track_nans for now; probs remove eventually
def concat_all_data(
    data: Dict[str, pl.LazyFrame], fill_method: str = "ffill", track_nans: bool = True
) -> pl.LazyFrame:
    # Validate inputs
    if not data:
        raise ValueError("Input data dictionary cannot be empty")

    valid_lazyframes = {}
    for name, lf in data.items():
        if not isinstance(lf, pl.LazyFrame):
            raise TypeError(f"Dataset '{name}' must be a polars LazyFrame")
        if "Date" not in lf.columns:
            raise ValueError(f"Dataset '{name}' missing required 'Date' column")
        valid_lazyframes[name] = lf

    if not valid_lazyframes:  # If all items were filtered out due to some validation
        raise ValueError("No valid LazyFrames found in the input data dictionary.")

    # Determine Overall Date Range
    all_min_dates: List[date] = []
    all_max_dates: List[date] = []

    for name, lf_item in valid_lazyframes.items():
        min_max_lf = lf_item.select(
            [
                pl.min("Date").alias("min_date_val"),
                pl.max("Date").alias("max_date_val"),
            ]
        )
        try:
            collected_dates_df = min_max_lf.collect()
        except Exception as e:
            # This error moste likely will occur if a file has become inaccessible after scan_csv)
            print(
                f"Warning: Could not collect min/max dates for '{name}': {e}. Skipping this item for date range calculation."
            )
            continue
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
    # Note to self in future polars date_range does not output a Datetime type but Date type instead
    master_dates_df = master_dates_df.with_columns(pl.col("Date").cast(pl.Datetime))
    result_lf = master_dates_df.lazy()

    # TODO investigate if there is a better way to handle this to prevent the need for as many for loops
    # Loop through datasets, process, and join lazily
    for name, lf_orig in valid_lazyframes.items():
        current_lf = lf_orig.clone()

        expressions_to_select = [
            pl.col("Date")
        ]  # Always include the Date column for joining

        # Iterate over columns intended for data (all columns except 'Date')
        data_columns = [col for col in current_lf.columns if col != "Date"]

        for col_name in data_columns:
            prefixed_col_name = f"{name}_{col_name}"
            data_col_expr = pl.col(col_name)

            if track_nans:
                # Add a column indicating if the original value was NaN
                # This happens *before* any filling.
                expressions_to_select.append(
                    data_col_expr.is_null()
                    .cast(pl.Int8)  # Cast boolean to int (0 or 1)
                    .alias(f"{prefixed_col_name}_was_nan")
                )

            # TODO seperate interpolation logic to process.py
            if fill_method == "ffill":
                filled_expr = data_col_expr.forward_fill()
            elif fill_method == "zero":
                filled_expr = data_col_expr.fill_null(0)
            else:
                filled_expr = data_col_expr  # No filling

            expressions_to_select.append(filled_expr.alias(prefixed_col_name))

        # Select and alias columns from the current commodity's LazyFrame
        # If current_lf might be empty after unique(), select could fail if no columns.
        # However, 'Date' is always there.
        if not expressions_to_select:  # Should not happen if 'Date' is always included
            continue

        lf_to_join = current_lf.select(expressions_to_select)
        result_lf = result_lf.join(lf_to_join, on="Date", how="left")

    return result_lf


if __name__ == "__main__":
    # DEBUGGING CODE: disable in prod
    commodity_futures = load_commodity_futures_by_folder(
        r"data\raw\commodity_data\daily"
    )
    concat_data_df = concat_all_data(
        commodity_futures, fill_method="ffill", track_nans=True
    ).collect()

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
