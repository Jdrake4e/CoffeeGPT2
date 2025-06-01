"""This module provide a pipeline for all data preprocccessing steps."""

import src.data_functions.load as load
import src.data_functions.process as process

# import src.data_function.dataloaders as dataloaders


# TODO figureout what type this will out put at the end, probs a torch dataloader
# TODO add an option to load preprocessed data
# TODO ensure processed data is saved if run
def run_data_pipeline(
    path: str = r"data\raw\commodity_data\daily",
    interpolation_type: str = "None",
    track_nans: bool = False,
) -> None:
    """Run the data pipeline for processing commodity futures data."""
    lf_dict = load.load_commodity_futures_by_folder(path)
    full_data = load.concat_all_data(lf_dict)
    null_data = process.track_nulls(full_data) if track_nans else full_data
    interpolated_data = (
        process.interpolate_data(null_data, interpolation_type)
        if interpolation_type != "None"
        else null_data
    )
    features_added_data = process.add_features(interpolated_data)  # noqa: F841
    pass
