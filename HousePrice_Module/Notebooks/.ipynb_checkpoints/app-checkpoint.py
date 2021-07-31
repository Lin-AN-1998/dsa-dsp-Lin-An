# import packages
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_log_error


# import train data
def read_data(input_path):
    df_master = pd.read_csv(input_path, index_col='Id')
    return df_master


# encode categorical values
def get_categorical_column_names(dataframe: pd.DataFrame) -> [str]:
    categorical_columns = list(dataframe.select_dtypes(include='object').columns)
    return categorical_columns


def get_one_hot_encoder(dataframe: pd.DataFrame) -> OneHotEncoder:
    categorical_columns = get_categorical_column_names(dataframe)
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', dtype=int, sparse=True)
    one_hot_encoder.fit(dataframe[categorical_columns])
    return one_hot_encoder


def encode_categorical_data(dataframe: pd.DataFrame, one_hot_encoder: OneHotEncoder) -> pd.DataFrame:
    categorical_columns = get_categorical_column_names(dataframe)
    encoded_categorical_data_matrix = one_hot_encoder.transform(dataframe[categorical_columns])
    encoded_data_columns = one_hot_encoder.get_feature_names(categorical_columns)
    encoded_categorical_data_df = pd.DataFrame.sparse.from_spmatrix(data=encoded_categorical_data_matrix,
                                                                    columns=encoded_data_columns, index=dataframe.index)
    encoded_df = dataframe.copy().drop(categorical_columns, axis=1).join(encoded_categorical_data_df)
    return encoded_df


# pre-processing
def pre_process(df_master):
    # fill missing value
    df_fill = df_master.fillna(df_master.mean())


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


# Inference
def make_prediction(model, df):
    prediction = model.predict(df)
    if (prediction.isna().sum() == 0) or (np.where(make_prediction(model, df) < 0) is not null):
        prediction = prediction.fillna(prediction.mean())
        return prediction
    else:
        return


# submission
def submission_file(X_test_id, y_pred):
    with open('./submission.csv', 'w') as writer:
        n = len(y_pred)

        writer.write('Id,SalePrice')
        writer.write('\n')

        for i in range(n):
            line = str(X_test_id[i]) + ',' + str(y_pred[i])
            writer.write(line)
            writer.write('\n')
