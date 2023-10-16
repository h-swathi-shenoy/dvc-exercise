import sys
import json
import pandas as pd
import os
import pickle, yaml
from utils.pathfinder import PathConfig
from utils.dataframe_conversion import  convert_numpy_df
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

path = PathConfig()
configs_dir = path.configs_dir
data_dir = path.data_dir
models_dir = path.models_dir

train_input = os.path.join(sys.argv[1], 'train.csv')
test_input = os.path.join(sys.argv[1], 'test.csv')
params = OmegaConf.load(configs_dir.joinpath('params.yaml'))['features']


processed_dir = data_dir.joinpath(sys.argv[2])
artifacts_dir = models_dir.joinpath(sys.argv[3])
features_dir = models_dir.joinpath(sys.argv[4])
os.makedirs(artifacts_dir, exist_ok=True)
os.makedirs(features_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)


def scale_df(input_file: str) -> None:
    """
     Scale the training data and save the scaler object, later to load it for inference
    :input_file: Train file
    :return: None
    """
    df = pd.read_csv(input_file)
    del df['Unnamed: 0']
    scaler_obj = MinMaxScaler()
    features = df.iloc[:, :-1].values
    scaler_obj.fit(features)
    filename = 'scaler_encode.pkl'
    scaled_arr = scaler_obj.fit_transform(features)
    scaled_df = convert_numpy_df(scaled_arr, target=False)
    scaled_df.to_csv(data_dir.joinpath('processed').joinpath("train-norm.csv"), index=False)
    pickle.dump(scaler_obj, open(artifacts_dir.joinpath(filename), 'wb'))
    return


def encode_labels(input_file: str) -> None:
    """
     Label  Encode the target labels and save the encoding model, later to be used for inference
    :input_file: Train file
    :return: None
    """
    df = pd.read_csv(input_file)
    label_encoder = LabelEncoder()
    labels = df.iloc[:, -1:].values
    label_enc = label_encoder.fit_transform(labels)
    label_df = convert_numpy_df(label_enc, target=True)
    label_df.to_csv(data_dir.joinpath('processed').joinpath("encoded-labels.csv"), index=False)
    filename = 'label_encode.pkl'
    pickle.dump(label_encoder, open(artifacts_dir.joinpath(filename), 'wb'))
    return


def feature_selection(input_file: str) -> None:
    """
    Select the features using SelectKBest based on F-value bw target and features. K is from params file
    :input_file: train dataframe
    """
    df = pd.read_csv(input_file)
    selector = SelectKBest(score_func=f_classif, k = params['max_features'] )
    best_feats = selector.fit_transform(df.iloc[:,:-1].values,df.iloc[:,-1:].values)
    cols_ids = selector.get_support(indices=True)
    feature_df = df.iloc[:, cols_ids]
    feature_names = feature_df.columns.tolist()
    feature_dict = {index: element for index, element in enumerate(feature_names)}
    with open(features_dir.joinpath('features.yaml'), 'w') as yaml_file:
        yaml.dump(feature_dict, yaml_file, default_flow_style=False)


if __name__ == '__main__':
    print(train_input)
    scale_df(train_input)
    encode_labels(train_input)
    feature_selection(train_input)

