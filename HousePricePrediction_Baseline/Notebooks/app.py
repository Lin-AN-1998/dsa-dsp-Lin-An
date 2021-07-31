# import packages
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_log_error, r2_score, mean_absolute_error, mean_squared_error


# import train data
def read_data(input_path):
    df_master = pd.read_csv(input_path, index_col='Id')
    return df_master


# encode categorical values
def get_encoder(df: pd.DataFrame):
    df_cate = df.select_dtypes(include='object')
    # creating instance of one-hot-encoder
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit_transform(df_cate)
    return enc


def encode_data(enc, dataframe: pd.DataFrame):
    categorical_columns = list(dataframe.select_dtypes(include='object').columns)
    encoded_categorical_data_matrix = enc.transform(dataframe[categorical_columns])
    encoded_data_columns = enc.get_feature_names(categorical_columns)
    encoded_categorical_data_df = pd.DataFrame.sparse.from_spmatrix(data=encoded_categorical_data_matrix,
                                                                    columns=encoded_data_columns, index=dataframe.index)
    encoded_df = dataframe.copy().drop(categorical_columns, axis=1).join(encoded_categorical_data_df)
    return encoded_df


# fill missing value
def pre_process(df: pd.DataFrame):
    df_total = df.fillna(0)
    return df_total


# Model building
# split data into train and test
def split_data(df_total, target_column):
    X, y = df_total.drop(target_column, axis=1), df_total[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


# train a model
def train_model(X_train: np.ndarray, y_train: np.ndarray):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


# model evaluation
def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# Inference
def make_prediction(model, df):
    prediction = model.predict(df)
    if (prediction.isna().sum() == 0) or (np.where(make_prediction(model, df) < 0) is not null):
        prediction = prediction.fillna(prediction.mean())
        return prediction
    else:
        return


# submission
def submission_file(inference_df, prediction, submission_file_path):
    inference_df['SalePrice'] = prediction
    total_inference_df = inference_df[['SalePrice']].reset_index()
    inference_ids_df = inference_df.reset_index()[['Id']]
    submission_df = inference_ids_df.merge(total_inference_df, on='Id', how='left')
    submission_df.to_csv(submission_file_path, index=False)
    return
