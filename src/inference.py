import sys
import numpy as np
import pandas as pd
import os
from pathlib import Path
from utils.pathfinder import PathConfig
from utils.dataframe_conversion import  convert_numpy_df
from sklearn.metrics import classification_report
from sklearn import metrics
from dvclive import Live
import pickle
import yaml
import json

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


def evaluate(model, matrix, labels, split, live, save_path):
    """
    Dump all evaluation metrics and plots for given datasets.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        matrix (scipy.sparse.csr_matrix): Input matrix.
        split (str): Dataset name.
        live (dvclive.Live): Dvclive instance.
        save_path (str): Path to save the metrics.
    """
    # labels = matrix[:, 1].toarray().astype(int)
    # x = matrix[:, 2:]

    predictions = model.predict(matrix.values)

    # Use dvclive to log a few simple metrics...
    avg_prec = metrics.precision_score(labels, predictions,average='micro')
    avg_recall = metrics.recall_score(labels, predictions, average='micro')
    if not live.summary:
        live.summary = {"avg_prec": {}, "avg_recall": {}}
    live.summary["avg_prec"][split] = avg_prec
    live.summary["avg_recall"][split] = avg_recall

    # ... and plots...
    # ... like an roc plot...
    #live.log_sklearn_plot("roc", labels, predictions, name=f"roc/{split}")
    # ... and precision recall plot...
    # ... which passes `drop_intermediate=True` to the sklearn method...
    # live.log_sklearn_plot(
    #     "precision_recall",
    #     labels,
    #     predictions,
    #     name=f"prc/{split}",
    #     #drop_intermediate=True,
    # )
    # # ... and confusion matrix plot
    # live.log_sklearn_plot(
    #     "confusion_matrix",
    #     labels.squeeze(),
    #     predictions_by_class.argmax(-1),
    #     name=f"cm/{split}",
    # )


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
    EVAL_PATH = "data/evaluate"
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
    with Live(EVAL_PATH, dvcyaml=False) as live:
        evaluate(rf_model, test_features,true_labels_enc, "test", live, save_path=EVAL_PATH)
    return true_labels_enc, pred


def save_metrics(y_true: np.array, y_pred: np.array) -> None:
    """
    Save the classification report
    :param y_true: Actual Labels from Test set
    :param y_pred: Predicted Labels for Test set
    """
    clf_report = pd.DataFrame(classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)).transpose()
    clf_dict = clf_report.to_dict(orient="index")
    with open(eval_dir.joinpath('data.json'), 'w') as fp:
        json.dump(clf_dict, fp)
    return


if __name__ == "__main__":
    scaler_obj = load_scaler(encoder_dir)
    encoder_obj = load_label_encoder(encoder_dir)
    model = load_model(artifacts_dir)
    true, predictions = batch_predictions(test_input, scaler_obj, encoder_obj, model)
    #save_metrics(true, predictions)
