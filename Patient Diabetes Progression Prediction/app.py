# backend
import pickle
import uvicorn
from fastapi import File, Form
from fastapi import FastAPI
from fastapi import UploadFile
import pandas as pd
# import numpy as np
# from PIL import Image
from pydantic import BaseModel

app = FastAPI()

'''
class Item(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
'''

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


# @app.post("/items/")
# async def create_item(item: Item):
#   return item


@app.post("/predict/single")
def single_predict(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6):
    data = {'age': [age], 'sex': [sex], 'bmi': [bmi], 'bp': [bp], 's1': [s1], 's2': [s2],
            's3': [s3], 's4': [s4], 's5': [s5], 's6': [s6]}
    df_test = pd.DataFrame(data)
    loaded_model = pickle.load(open('Patient_Prediction.pkl', 'rb'))
    prediction = loaded_model.predict(df_test)
    return prediction


@app.post("/predict/multiple")
def multi_predict(file: UploadFile = File(...)):
    df_test = pd.read_csv(file)
    loaded_model = pickle.load(open('Patient_Prediction.pkl', 'rb'))
    prediction = loaded_model.predict(df_test)
    df_test['target'] = prediction
    return df_test


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8080)
