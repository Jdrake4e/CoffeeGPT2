"""The module contains a pipeline and requisite functions to scale data."""

from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


def auto_scaler_selection() -> list[
    StandardScaler
    | MinMaxScaler
    | RobustScaler
    | PowerTransformer
    | QuantileTransformer
]:
    """Select the appropriate scaler based on the data type."""
    # TODO: Implement logic to select scaler based on data characteristics
    return [StandardScaler()]
