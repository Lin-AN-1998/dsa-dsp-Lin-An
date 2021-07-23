import io
import ModelTraining
from starlette.responses import Response
import ui
from fastapi import FastAPI, File

app = FastAPI(
    title="DeepLabV3 image segmentation",
    description="""Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
                           Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)


@app.post("/prediction")
def get_prediction():
    """Get segmentation maps from image file"""
    file = ModelTraining.read_data('Test.csv')
    df_test = ModelTraining.process(file)
    encode_df_test = ModelTraining.encode_data(ModelTraining.enc, df_test)
    df_total_test = ModelTraining.pre_process(encode_df_test)
    prediction = ModelTraining.predict(ModelTraining.model, df_total_test)
    return Response(prediction.getvalue())
