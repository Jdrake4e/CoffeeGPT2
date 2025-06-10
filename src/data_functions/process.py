"""This Module contains functions for processing data in a LazyFrame format.

It includes functions for interpolating missing data, tracking NaNs,
and adding features(tbd which ones).
"""

from typing import Any, Literal

import polars as pl


def _get_base_columns(lf: pl.LazyFrame) -> list[str]:
    return [
        col
        for col in lf.collect_schema().names()
        if not any(
            [
                col.endswith("_was_null"),
                col == "Date",
                "_ma_" in col,
                "_ewma_" in col,
                "_return" in col,
                "_log_return" in col,
            ]
        )
    ]


def _generate_column_alias(
    base_col: str, operation_name: str, params: dict[str, Any]
) -> str:
    """Generates a consistent column alias."""
    alias_parts = [base_col, operation_name]
    for param_name, param_value in params.items():
        # Sanitize param_name for alias (e.g., 'window_size' -> 'w')
        if param_name == "window_size":
            alias_parts.append(f"w{param_value}")
        elif param_name == "min_samples":
            if param_value != params.get("window_size", 1):  # Only add if not default
                alias_parts.append(f"ms{param_value}")
        elif param_name == "alpha":
            alias_parts.append(f"a{param_value}")
        elif param_name == "span":
            alias_parts.append(f"s{param_value}")
        elif param_name == "half_life":
            alias_parts.append(f"hl{param_value}")
    return "_".join(alias_parts)


def _add_percent_returns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Adds percent returns to the LazyFrame."""
    base_columns = _get_base_columns(lf)

    expressions = [
        (pl.col(col) / pl.col(col).shift(1) - 1).alias(f"{col}_return")
        for col in base_columns
    ]

    return lf.with_columns(expressions)


def _add_log_returns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Adds log returns to the LazyFrame."""
    base_columns = _get_base_columns(lf)

    expressions = [
        (pl.col(col).log() - pl.col(col).shift(1).log()).alias(f"{col}_log_return")
        for col in base_columns
    ]

    return lf.with_columns(expressions)


def add_returns(
    lf: pl.LazyFrame, return_types: list[Literal["percent", "log"]]
) -> pl.LazyFrame:
    """Adds both percent and log returns to the LazyFrame.

    This function checks the return_types list and adds the specified returns.
    Supported return types are:
        - "percent": Adds percent returns.
        - "log": Adds log returns.

    Args:
        lf (pl.LazyFrame): The LazyFrame to add returns to.
        return_types (list[str]): List of return types to add. Options are:
            - "percent": Adds percent returns.
            - "log": Adds log returns.

    Returns:
        pl.LazyFrame: LazyFrame with returns added as new columns.

    Raises:
        ValueError: If an invalid return type is specified.
    """
    if not return_types:
        raise ValueError("return_types must not be empty")
    if not all(rt in {"percent", "log"} for rt in return_types):
        raise ValueError(
            f"return_types must contain only 'percent' or 'log', got {return_types}"
        )

    if "percent" in return_types:
        lf = _add_percent_returns(lf)
    if "log" in return_types:
        lf = _add_log_returns(lf)
    return lf


# TODO improve naming of columns also make a function like the one
#      above for modularity and input of a list of features
def moving_average(lf: pl.LazyFrame, ma_configs: list[tuple[int, int]]) -> pl.LazyFrame:
    """Calculate simple moving averages for specified configurations.

    Args:
        lf (pl.LazyFrame): The LazyFrame to calculate moving averages on.
        ma_configs (list[tuple[int, int]]): List of (window_size, min_samples) tuples.

    Returns:
        pl.LazyFrame: LazyFrame with moving averages added as new columns.

    Raises:
        ValueError: If any configuration is invalid.
    """
    # Validation
    for window_size, min_samples in ma_configs:
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        if min_samples <= 0:
            raise ValueError(f"min_samples must be > 0, got {min_samples}")
        if min_samples > window_size:
            raise ValueError(
                f"min_samples ({min_samples}) cannot exceed window_size ({window_size})"
            )

    base_columns = _get_base_columns(lf)

    all_expressions = []
    for window_size, min_samples in ma_configs:
        expressions = [
            pl.col(col)
            .rolling_mean(window_size, min_samples=min_samples)
            .alias(f"{col}_ma_{window_size}d")
            for col in base_columns
        ]
        all_expressions.extend(expressions)

    return lf.with_columns(all_expressions)


def _validate_ewma_configs(ewma_configs: list[dict]) -> None:
    """Validate the EWMA configurations."""
    for config in ewma_configs:
        if not isinstance(config, dict):
            raise ValueError("Each EWMA config must be a dictionary")

        valid_params = {"alpha", "span", "half_life"}
        provided_params = set(config.keys()) & valid_params

        if len(provided_params) != 1:
            raise ValueError(
                f"Each config must have exactly one of {valid_params}"
                f", got {list(config.keys())}"
            )

        param_name = next(iter(provided_params))
        param_value = config[param_name]

        if not isinstance(param_value, int | float) or isinstance(param_value, bool):
            raise ValueError(f"{param_name} must be numeric, got {type(param_value)}")

        if param_value != param_value or abs(param_value) == float(
            "inf"
        ):  # Check for NaN and inf
            raise ValueError(f"{param_name} must be finite")

        if param_name == "alpha" and not (0 < param_value <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {param_value}")
        elif param_name == "span" and param_value < 1:
            raise ValueError(f"span must be >= 1, got {param_value}")
        elif param_name == "half_life" and param_value <= 0:
            raise ValueError(f"half_life must be > 0, got {param_value}")


# TODO improve naming of columns
def exponential_weighted_moving_average(
    lf: pl.LazyFrame, ewma_configs: list[dict]
) -> pl.LazyFrame:
    """Calculate exponential weighted moving averages for specified configurations.

    Args:
        lf (pl.LazyFrame): The LazyFrame to calculate EWMA on.
        ewma_configs (list[dict]): List of dicts with one of: alpha, span, or half_life.

    Returns:
        pl.LazyFrame: LazyFrame with EWMA added as new columns.

    Raises:
        ValueError: If any configuration is invalid.
    """
    _validate_ewma_configs(ewma_configs)

    base_columns = _get_base_columns(lf)

    all_expressions = []
    for config in ewma_configs:
        param_name = next(iter(set(config.keys()) & {"alpha", "span", "half_life"}))
        param_value = config[param_name]

        # Create alias suffix based on parameter type
        if param_name == "alpha":
            suffix = f"ewma_alpha_{param_value}"
        elif param_name == "span":
            suffix = f"ewma_span_{param_value}"
        else:  # half_life
            suffix = f"ewma_hl_{param_value}"

        expressions = [
            pl.col(col).ewm_mean(**config).alias(f"{col}_{suffix}")
            for col in base_columns
        ]
        all_expressions.extend(expressions)

    return lf.with_columns(all_expressions)


def _add_rolling_std(
    lf: pl.LazyFrame, window_size: int, min_samples: int = 1
) -> pl.LazyFrame:
    """Calculate rolling standard deviation for specified window size and min samples.

    Args:
        lf (pl.LazyFrame): The LazyFrame to calculate rolling std on.
        window_size (int): The size of the rolling window.
        min_samples (int): Minimum number of non-null observations required.

    Returns:
        pl.LazyFrame: LazyFrame with rolling standard deviation added as new columns.
    """
    base_columns = _get_base_columns(lf)

    expressions = [
        pl.col(col)
        .rolling_std(window_size, min_samples=min_samples)
        .alias(f"{col}_rolling_std_{window_size}d")
        for col in base_columns
    ]

    return lf.with_columns(expressions)


def _add_rolling_var(
    lf: pl.LazyFrame, window_size: int, min_samples: int = 1
) -> pl.LazyFrame:
    """Calculate rolling variance for specified window size and min samples.

    Args:
        lf (pl.LazyFrame): The LazyFrame to calculate rolling variance on.
        window_size (int): The size of the rolling window.
        min_samples (int): Minimum number of non-null observations required.

    Returns:
        pl.LazyFrame: LazyFrame with rolling variance added as new columns.
    """
    base_columns = _get_base_columns(lf)

    expressions = [
        pl.col(col)
        .rolling_var(window_size, min_samples=min_samples)
        .alias(f"{col}_rolling_var_{window_size}d")
        for col in base_columns
    ]

    return lf.with_columns(expressions)


def add_rolling_stats(
    lf: pl.LazyFrame,
    window_size: int,
    min_samples: int = 1,
    stats: list[str] | None = None,
) -> pl.LazyFrame:
    """Add rolling statistics to the LazyFrame.

    Args:
        lf (pl.LazyFrame): The LazyFrame to add rolling statistics to.
        window_size (int): The size of the rolling window.
        min_samples (int): Minimum number of non-null observations required.
        stats (list[str]): List of statistics to calculate. Options are:
            - "std": Rolling standard deviation
            - "var": Rolling variance

    Returns:
        pl.LazyFrame: LazyFrame with rolling statistics added as new columns.
    """
    if stats is None:
        stats = ["std", "var"]

    if not stats:
        raise ValueError("stats must not be empty")
    if not all(stat in {"std", "var"} for stat in stats):
        raise ValueError(f"stats must contain only 'std' or 'var', got {stats}")

    if "std" in stats:
        lf = _add_rolling_std(lf, window_size, min_samples)
    if "var" in stats:
        lf = _add_rolling_var(lf, window_size, min_samples)

    return lf


def track_nulls(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Track null per feature by marking with a 1 if null, 0 if not null."""
    expressions = [
        pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_was_null")
        for col in lf.collect_schema().names()
        if col != "Date"
    ]

    return lf.with_columns(expressions).lazy()


def drop_nulls(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Drop rows with null values."""
    return lf.drop_nulls()


def interpolate_data(
    lf: pl.LazyFrame, interpolation_type: str = "None"
) -> pl.LazyFrame:
    """Interpolate missing data in a LazyFrame.

    This function supports various interpolation methods including:
    TODO: Implement cubic spline, B-spline, Chebyshev,
        and radial basis function interpolation.

    Args:
        lf (pl.LazyFrame): The LazyFrame to interpolate.
        interpolation_type (str): The type of interpolation to use. Options are:
            - "ffill": Forward fill
            - "bfill": Backward fill
            - "linear": Linear interpolation
            - "cubic_spline": Cubic spline interpolation
            - "b_spline": B-spline interpolation
            - "Chebyshev": Chebyshev interpolation
            - "radial_basis_function": Radial basis function interpolation

    Returns: lazyframe with interpolated data.

    """
    match interpolation_type:
        case "ffill":
            return _fill_forward(lf)
        case "bfill":
            return _fill_backward(lf)
        case "linear":
            return _fill_linear(lf)
        case "cubic_spline":
            return _fill_cubic_spline(lf)
        case "b_spline":
            return _fill_b_spline(lf)
        case "Chebyshev":
            return _fill_chebyshev(lf)
        case "radial_basis_function":
            return _fill_radial_basis_function(lf)
        case _:
            raise ValueError(f"Unknown interpolation type: {interpolation_type}")


def _fill_forward(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Fill missing values using forward fill method."""
    return lf.fill_null(strategy="forward")


def _fill_backward(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Fill missing values using backward fill method."""
    return lf.fill_null(strategy="backward")


def _fill_linear(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Fill missing values using linear interpolation."""
    return lf.interpolate()


def _fill_cubic_spline(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Fill missing values using cubic spline interpolation."""
    # TODO Implement cubic spline interpolation
    #      Looking at polars docs, no built in way
    #      so will have to implement manually or
    #      find a library for the following functions below
    pass


def _fill_b_spline(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Fill missing values using B-spline interpolation."""
    pass


def _fill_chebyshev(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Fill missing values using Chebyshev interpolation."""
    pass


def _fill_radial_basis_function(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Fill missing values using radial basis function interpolation."""
    pass


def add_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """To be decided what features to add."""
    pass


# look at Regime-Aware Missing Data Imputation
