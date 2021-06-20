from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib


def preprocess(dataframe: pd.DataFrame, models_dir) -> pd.DataFrame:
    processed_df = dataframe.copy()
    processed_df = preprocess_continuous_data(processed_df)
    processed_df = preprocess_categorical_data(processed_df, models_dir)
    return processed_df


def preprocess_continuous_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    return impute_continuous_missing_values(dataframe)


def impute_continuous_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    default_value = 0
    continuous_columns = dataframe.select_dtypes(include='number').columns
    [dataframe[column_name].fillna(default_value, inplace=True) for column_name in continuous_columns]
    return dataframe


def preprocess_categorical_data(dataframe: pd.DataFrame, models_dir: Path) -> pd.DataFrame:
    dataframe = impute_categorical_missing_values(dataframe)
    encoder = get_one_hot_encoder(dataframe, models_dir)
    encoded_df = encode_categorical_data(dataframe, encoder)
    return encoded_df


def impute_categorical_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    default_value = 'missing value mate !'
    categorical_columns = get_categorical_column_names(dataframe)
    [dataframe[column_name].fillna(default_value, inplace=True) for column_name in categorical_columns]
    return dataframe


def get_categorical_column_names(dataframe: pd.DataFrame) -> [str]:
    categorical_columns = list(dataframe.select_dtypes(include='object').columns)
    return categorical_columns


def get_one_hot_encoder(dataframe: pd.DataFrame, encoder_dir: Path) -> OneHotEncoder:
    one_hot_encoder_path = encoder_dir / 'one_hot_encoder.joblib'
    if one_hot_encoder_path.exists():
        one_hot_encoder = joblib.load(one_hot_encoder_path)
    else:
        categorical_columns = get_categorical_column_names(dataframe)
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', dtype=int, sparse=True)
        one_hot_encoder.fit(dataframe[categorical_columns])
        joblib.dump(one_hot_encoder, one_hot_encoder_path)
    return one_hot_encoder


def encode_categorical_data(dataframe: pd.DataFrame, one_hot_encoder: OneHotEncoder) -> pd.DataFrame:
    categorical_columns = get_categorical_column_names(dataframe)
    encoded_categorical_data_matrix = one_hot_encoder.transform(dataframe[categorical_columns])
    encoded_data_columns = one_hot_encoder.get_feature_names(categorical_columns)
    encoded_categorical_data_df = pd.DataFrame.sparse.from_spmatrix(data=encoded_categorical_data_matrix,
                                                                    columns=encoded_data_columns, index=dataframe.index)
    encoded_df = dataframe.copy().drop(categorical_columns, axis=1).join(encoded_categorical_data_df)
    return encoded_df
