import pickle

import pandas as pd
import streamlit as st
# EDA
import matplotlib.pyplot as plt
import matplotlib
import base64

matplotlib.use('Agg')
from pandas import DataFrame
import app

from PIL import Image


# functions
################### Display stuff
# from inference import predict
def main():
    st.title('Patient diabetes progression Prediction')
    Menu = ["Home", "Single Prediction", "Multiple Prediction"]
    choice = st.sidebar.selectbox("Menu", Menu)
    if choice == "Home":
        st.subheader("Welcome to Prediction World!")
        st.markdown("This application will help you to predict patient diabetes progression")
        image = Image.open('photo.jpg')
        st.image(image)

    if choice == "Single Prediction":
        st.subheader("Single Prediction")
        # Input your information
        # Numeric Variables

        age = st.number_input('Please input age')
        sex = st.number_input('Please input sex')
        bmi = st.number_input('Please input bmi')
        bp = st.number_input('Please input bp')
        s1 = st.number_input('Please input s1')
        s2 = st.number_input('Please input s2')
        s3 = st.number_input('Please input s3')
        s4 = st.number_input('Please input s4')
        s5 = st.number_input('Please input s5')
        s6 = st.number_input('Please input s6')

        # Multi line text
        Note = st.text_area('Enter your notes here')
        # execute model
        if st.button('Predict'):
            # single_df
            prediction = app.single_predict(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6)
            prediction

    if choice == "Multiple Prediction":
        st.subheader("Multiple Prediction")
        csv_file = st.file_uploader('Choose a CSV file')
        if csv_file:
            df_test = app.multi_predict(csv_file)

        # execute model
        if st.button('Predict'):
            if csv_file is not None:
                st.subheader("Output")
                df_test
            else:
                st.warning('You need to upload a csv file before')


if __name__ == '__main__':
    main()
