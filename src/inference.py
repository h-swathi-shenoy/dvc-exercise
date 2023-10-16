import sys
from typing import Tuple, Any

import numpy as np
import pandas as pd
import os
from pathlib import Path
from utils.pathfinder import PathConfig
from utils.dataframe_conversion import  convert_numpy_df
from sklearn.metrics import classification_report
import pickle
import yaml

path = PathConfig()
configs_dir = path.configs_dir
data_dir = path.data_dir
models_dir = path.models_dir

test_input = os.path.join(sys.argv[1],'test.csv')
features_dir = models_dir.joinpath(sys.argv[2])
artifacts_dir = models_dir.joinpath(sys.argv[3])
encoder_dir = models_dir.joinpath(sys.argv[4])
eval_dir = data_dir.joinpath('evaluate')

os.makedirs(eval_dir, exist_ok=True)

def load_label_encoder(encoder: Path) -> pickle:
    """
    Load the label encoder model in a given Path
    :param encoder: Path to the saved encoder model
    :return: Model stored in pickle format
    """
    with open(encoder.joinpath('label_encode.pkl'), 'rb') as fp:
        label_encoder = pickle.load(fp)
    return label_encoder


def load_scaler(scaler: Path) -> pickle:
    """
    Load the saved Scaler Object in a given Path
    :param scaler: Path to saved MinmaxScaler Object
    :return: Scaler object
    """
    with open(scaler.joinpath('scaler_encode.pkl'), 'rb') as fp:
        scaler_obj = pickle.load(fp)
    return scaler_obj


def load_model(artifacts: Path) -> pickle:
    """
    Load the saved Classification model in a given Path
    :param artifacts:  Path to the saved Multiclass Classification Model
    :return: Model
    """
    with open(artifacts.joinpath('model.pkl'), 'rb') as fp:
        model = pickle.load(fp)
    return model


def batch_predictions(test: str, scaler: pickle, encoder: pickle, rf_model: pickle) -> tuple[np.array, np.array]:
    """
    Return the batch of predictions from the test set
    :param test: Test set path
    :param scaler: scaler object
    :param encoder: encoder object for targets
    :param rf_model: Multi-class Classifaction model
    :return: True and Predicted Labels
    """
    test_df = pd.read_csv(test)
    del test_df['Unnamed: 0']
    true_labels = test_df.iloc[:, -1].values
    test_arr = scaler.transform(test_df.iloc[:,:-1].values)
    test_df = convert_numpy_df(test_arr)
    true_labels_enc = encoder.transform(true_labels)
    with open(features_dir.joinpath("features.yaml"), "r") as stream:
        feats = yaml.safe_load(stream)
    test_features = test_df[feats.values()]
    pred = rf_model.predict(test_features.values)
    return true_labels_enc, pred


def save_metrics(y_true: np.array, y_pred: np.array) -> None:
    """
    Save the classification report
    :param y_true: Actual Labels from Test set
    :param y_pred: Predicted Labels for Test set
    """
    clf_report = pd.DataFrame(classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)).transpose()
    clf_report.to_csv(eval_dir.joinpath('Metrics.csv'), index=True)


if __name__ == "__main__":
    scaler_obj = load_scaler(encoder_dir)
    encoder_obj = load_label_encoder(encoder_dir)
    model = load_model(artifacts_dir)
    true, predictions = batch_predictions(test_input, scaler_obj, encoder_obj, model)
    save_metrics(true, predictions)
