import os

import src.data_functions.load as load
import src.data_functions.process as process


# TODO figureout what type this will out put at the end, probs a torch dataloader
# TODO add an option to load preprocessed data
# TODO ensure processed data is saved if run
def run_data_pipeline(
    path: str = "data\raw\commodity_data\daily",
    interpolation_type: str = "ffill",
    track_nans: bool = True,
):
    lf_dict = load.futures_readin_bind(path)
    lf_dict = load.concat_all_data(lf_dict)
    lf_dict = process.interpolate_data(lf_dict, interpolation_type, track_nans)
    lf_dict = process.add_features(lf_dict)

    full_data = load.concat_all_data(lf_dict)
