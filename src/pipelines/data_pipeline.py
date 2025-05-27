import os

import src.data_functions.load as load


# TODO figureout what type this will out put at the end, probs a torch dataloader
# TODO add an option to load preprocessed data
# TODO ensure processed data is saved if run
def run_data_pipeline(
    path: os.PathLike = "data\raw\commodity_data\daily",
    interpolation_type: str = "ffill",
    track_nans: bool = True,
):
    lf_dict = load.futures_readin_bind(path)
    full_data = load.concat_all_data(lf_dict)
