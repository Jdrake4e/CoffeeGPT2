import src.data_functions.load as load
import src.data_functions.process as process


# TODO figureout what type this will out put at the end, probs a torch dataloader
# TODO add an option to load preprocessed data
# TODO ensure processed data is saved if run
def run_data_pipeline(
    path: str = r"data\raw\commodity_data\daily",
    interpolation_type: str = "ffill",
    track_nans: bool = True,
) -> None:
    lf_dict = load.load_commodity_futures_by_folder(path)
    full_data = load.concat_all_data(lf_dict)
    interpolated_data = process.interpolate_data(
        full_data, interpolation_type, track_nans
    )
    features_added_data = process.add_features(interpolated_data)  # noqa: F841
    pass
