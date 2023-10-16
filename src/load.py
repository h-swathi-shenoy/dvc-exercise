import pandas as pd
import sys
import os
from omegaconf import DictConfig, OmegaConf
from utils.pathfinder import PathConfig
from sklearn.model_selection import train_test_split

path = PathConfig()
configs_dir = path.configs_dir
data_dir = path.data_dir

params = OmegaConf.load(configs_dir.joinpath('params.yaml'))['load']
input_file = sys.argv[1]
os.makedirs(os.path.join(data_dir, 'prepared'), exist_ok=True)


def load_data() -> None:
    """
    Split the raw data to Train/Test based on the seed and split ratio from params.yaml
    :return:
    """
    data = pd.read_csv(data_dir.joinpath(input_file))
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=params['random_state'],
                                                        test_size=params['split'])
    train_df = pd.concat([pd.DataFrame(x_train), pd.DataFrame(y_train)], axis=1)
    train_df.columns = params['features']
    test_df = pd.concat([pd.DataFrame(x_test), pd.DataFrame(y_test)], axis=1)
    test_df.columns = params['features']
    train_df.to_csv(data_dir.joinpath('prepared/train.csv'))
    test_df.to_csv(data_dir.joinpath('prepared/test.csv'))
    return


if __name__=='__main__':
    load_data()