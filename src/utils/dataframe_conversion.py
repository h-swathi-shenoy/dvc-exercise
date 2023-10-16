import pandas as pd
import numpy as np
from utils.pathfinder import PathConfig
from omegaconf import OmegaConf
import typing as t
from typing import Optional

path = PathConfig()
configs_dir = path.configs_dir
data_dir = path.data_dir

params = OmegaConf.load(configs_dir.joinpath('params.yaml'))['load']


def convert_numpy_df(input_arr: np.array, target: t.Optional = False) -> pd.DataFrame:
    """
    Convert the input array to Dataframe.
    :param input_arr: numpy array input
    :param target: Flag to indicate if the label column included or no
    :return: Input array converted to Pd.DataFrame
    """
    if target:
        features = params['features'][-1:]
        input_df = pd.DataFrame(input_arr, columns=features)
    else:
        features = params['features'][:-1]
        input_df = pd.DataFrame(input_arr, columns=features)
    return input_df
