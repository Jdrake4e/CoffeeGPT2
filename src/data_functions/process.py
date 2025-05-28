import polars as pl


def interpolate_data(
    lf: pl.LazyFrame, interpolation_type: str = "None", track_nans="True"
) -> pl.LazyFrame:
    match interpolation_type:
        case "ffill":
            return forward_fill(lf)
        case "bfill":
            return backward_fill(lf)
        case "None":
            return fill_none(lf)
        case "linear":
            return fill_linear(lf)
        case _:
            raise ValueError(f"Unknown interpolation type: {interpolation_type}")


def forward_fill(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.fill_null(strategy="forward")


def backward_fill(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.fill_null(strategy="backward")


def fill_none(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.fill_null(strategy=None)


def fill_linear(lf: pl.LazyFrame) -> pl.LazyFrame:
    pass


def add_features(lf) -> pl.LazyFrame:
    pass
