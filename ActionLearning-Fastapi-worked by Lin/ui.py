import io
import pandas as pd
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

import streamlit as st

# interact with FastAPI endpoint
import ModelTraining

backend = "http://fastapi:8000/bankchrun"

from fastapi import FastAPI, File

app = FastAPI(
    title="DeepLabV3 image segmentation",
    description="""Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
                           Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)


def process(file, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", file)})

    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )

    return r


# construct UI layout
st.title("bank customer churn prediction")

st.write(
    """Obtain prediction result from csv file.
         This streamlit example uses a FastAPI service as backend.
         Visit this URL at `:8000/docs` for FastAPI documentation."""
)  # description and instructions


csv_file = st.file_uploader('Choose a CSV file')

if st.button('Predict customer churn'):
    if csv_file:
        st.subheader("Output")
        @app.post("/uploadcsv/")
        def upload_csv(csv_file: UploadFile = File(...)):
            dataframe = pd.read_csv(csv_file.file)
            # do something with dataframe here (?)
            return {"filename": csv_file.filename}


        process(csv_file, backend)
        df_test = ModelTraining.read_data('Test.csv')
        df_test = ModelTraining.process(df_test)
        encode_df_test = ModelTraining.encode_data(ModelTraining.enc, df_test)
        df_total_test = ModelTraining.pre_process(encode_df_test)
        prediction = ModelTraining.predict(ModelTraining.model, df_total_test)
        ModelTraining.submission_file(df_test, prediction, 'submission.csv')
        submit_df = ModelTraining.read_data('submission.csv')
        out_df = pd.DataFrame({'Age': submit_df['Age'],
                               'CreditScore': submit_df['CreditScore'],
                               'Tenure': submit_df['Age'],
                               'Balance': submit_df['Age'],
                               'NumOfProducts': submit_df['NumOfProducts'],
                               'HasCrCard': submit_df['HasCrCard'],
                               'IsActiveMember': submit_df['IsActiveMember'],
                               'EstimatedSalary': submit_df['EstimatedSalary'],
                               'Geography': submit_df['Geography'],
                               'Gender': submit_df['Gender'],
                               'Exited_prediction': submit_df['Exited_x']
                               })
        # out_df = out_df.sort_values(by='Stock date', ascending=True)
        #return out_df

        # st.dataframe(data=out_df.style.highlight_max(subset=['Exited_prediction'], color='green'), width=4000,
                     # height=700)
        #st.markdown("Date represents customers will churn or not ")
    else:
        st.warning('You need to upload a csv file before')




