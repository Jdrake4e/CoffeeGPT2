"""This module provide a pipeline for all data preprocccessing steps."""

# from ..data_function as dataloaders
from typing import Literal

import polars as pl

from ..data_functions import load, process  # noqa: TID252


# TODO figureout what type this will out put at the end, probs a torch dataloader
# TODO add an option to load preprocessed data
# TODO ensure processed data is saved if run
def run_data_pipeline(
    path: str = r"data\raw\commodity_data\daily",
    interpolation_type: str = "ffill",
    track_nans: bool = True,
    drop_nulls: bool = False,
    ma_configs: list[tuple[int, int]] | None = None,
    ewma_configs: list[dict] | None = None,
    returns_features: list[Literal["percent", "log"]] | None = None,
    rolling_var_features: list[dict[str, int | list[Literal["std", "var"]]]]
    | None = None,
) -> pl.LazyFrame:
    """Run the data pipeline for processing commodity futures data."""
    # TODO remove defaults after testing
    # TODO unify rolling features, moving averages and returns features
    # TODO improve data structure for customizable overrides for new feature creation
    if ma_configs is None:
        ma_configs = [
            (20, 1),
            (50, 1),
            (200, 1),
            (365, 1),
            (2 * 365, 1),
        ]
    if returns_features is None:
        returns_features = ["percent", "log"]
    if rolling_var_features is None:
        rolling_var_features = [
            {"window_size": 20, "stats": ["std", "var"]},
            {"window_size": 50, "stats": ["std", "var"]},
            {"window_size": 200, "stats": ["std", "var"]},
            {"window_size": 365, "stats": ["std", "var"]},
            {"window_size": 2 * 365, "stats": ["std", "var"]},
        ]

    if ewma_configs is None:
        ewma_configs = [{"alpha": 0.1}, {"span": 20}, {"half_life": 10}]

    print("\nStarting data pipeline...")
    lf_dict = load.load_commodity_futures_by_folder(path)
    full_data = load.concat_all_data(lf_dict)
    null_data = full_data.drop_nulls() if drop_nulls else full_data
    track_data = process.track_nulls(full_data) if track_nans else null_data
    interpolated_data = (
        process.interpolate_data(track_data, interpolation_type)
        if interpolation_type != "None"
        else track_data
    )
    ma_data = interpolated_data
    if ma_configs:
        ma_data = process.moving_average(ma_data, ma_configs)
    if ewma_configs:
        # TODO switch to polars.Expr.ewm_mean_by and reverse the order of creating
        #      these to take advantage of the function I found
        ma_data = process.exponential_weighted_moving_average(ma_data, ewma_configs)
    returns_data = process.add_returns(ma_data, returns_features)
    var_data = process.add_rolling_stats(returns_data, rolling_var_features)

    final_data = var_data
    return final_data
