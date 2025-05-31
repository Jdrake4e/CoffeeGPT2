"""This Module contains functions for processing data in a LazyFrame format.

It includes functions for interpolating missing data, tracking NaNs,
and adding features(tbd which ones).
"""

import polars as pl


def track_nans(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Track nans per feature by marking with a 1 if nan, 0 if not nan."""
    pass


def interpolate_data(
    lf: pl.LazyFrame, interpolation_type: str = "None"
) -> pl.LazyFrame:
    """Interpolate missing data in a LazyFrame.

    This function supports various interpolation methods including:

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
        track_nans (bool): Whether to track NaNs in the data.

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
    pass


def _fill_cubic_spline(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Fill missing values using cubic spline interpolation."""
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
