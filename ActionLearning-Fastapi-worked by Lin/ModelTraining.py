import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def read_data(input_path):
    df = pd.read_csv(input_path)
    #df['index'].astype(int)
    return df


def process(df):
    cols_to_remove = ['Surname', 'RowNumber', 'CustomerId']
    df = df.drop(cols_to_remove, axis=1) #, inplace=True)
    #df = df.reset_index()
    return df

def get_encoder(df):
    # categorical_columns = get_categorical_column_names(df)
    df_cate = df.select_dtypes(include='object')
    # creating instance of one-hot-encoder
    enc = OneHotEncoder(handle_unknown='ignore', dtype=int, sparse=True)
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

'''
def encode_data(enc, df: pd.DataFrame):
    df_cate = df.select_dtypes(include='object')
    # categorical_columns = get_categorical_column_names(df)
    encoded_categorical_data_matrix = enc.transform(df_cate)
    encoded_data_columns = enc.get_feature_names(df_cate.columns)
    encoded_categorical_data_df = pd.DataFrame.sparse.from_spmatrix(data=encoded_categorical_data_matrix,
                                                                    columns=encoded_data_columns, index=df.index)
    encoded_df = df.copy().drop(df_cate.columns, axis=1).join(encoded_categorical_data_df)
    return encoded_df
'''

# fill missing value
def pre_process(df: pd.DataFrame):
    df_total = df.fillna(0)
    return df_total


# Model building
# split data into train and test
def split_data(df_total, target_column):
    X, y = df_total.drop(target_column, axis=1), df_total[target_column]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X, y


# train a model
def train_model(X_train: np.ndarray, y_train: np.ndarray):
    model = LogisticRegression()
    X_train.reset_index(drop=True)
    model.fit(X_train, y_train)
    return model


# Inference
def predict(model, df):
    prediction = model.predict(df)
    return prediction


# submission
def submission_file(inference_df, prediction, path):
    inference_df['Exited'] = prediction
    #prediction.reset_index()
    total_inference_df = inference_df[['Exited']].reset_index()
    inference_ids_df = inference_df.reset_index()
    submission_df = inference_ids_df.merge(total_inference_df, on='index', how='left')
    submission_df.to_csv(path, index=False)
    return submission_df


dataframe = read_data('Churn_Modelling.csv')
df = process(dataframe)
enc = get_encoder(df)
encode_df = encode_data(enc, df)
df_total = pre_process(encode_df)
X_train, y_train = split_data(df_total, 'Exited')
model = train_model(X_train, y_train)
'''
# predict test dataset
df_test = read_data('Test.csv')
df_test = process(df_test)
encode_df_test = encode_data(enc, df_test)
df_total_test = pre_process(encode_df_test)
prediction = predict(model, df_total_test)
submission_file(df_test, prediction, 'submission.csv')
'''

