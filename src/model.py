import sys
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from utils.pathfinder import PathConfig
from omegaconf import OmegaConf
import yaml


path = PathConfig()
configs_dir = path.configs_dir
data_dir = path.data_dir
models_dir = path.models_dir

features_input = os.path.join(sys.argv[1], 'train-norm.csv')
target_input = os.path.join(sys.argv[1], 'encoded-labels.csv')
artifacts_dir = models_dir.joinpath(sys.argv[2])
features_dir = models_dir.joinpath(sys.argv[3])
os.makedirs(artifacts_dir, exist_ok=True)
params = OmegaConf.load(configs_dir.joinpath('params.yaml'))['training']


def train_model(features: pd.DataFrame, target: pd.DataFrame) -> None:
    """
    Train a Multi-Class Classfier model with given Features and Target Labels of a Dataframe.
    :param features: Training Features in a Dataframe
    :param target: Labels for Training
    :return:
    """
    features_df = pd.read_csv(features)
    target_df  = pd.read_csv(target)
    rf_class = RandomForestClassifier(random_state=params.random_state)
    with open(features_dir.joinpath("features.yaml"), "r") as stream:
        feats = yaml.safe_load(stream)
    rf_class.fit(features_df[feats.values()].values, target_df.values)
    filename = 'model.pkl'
    pickle.dump(rf_class, open(artifacts_dir.joinpath(filename),'wb'))
    return


if __name__ == "__main__":
    train_model(features_input, target_input)
