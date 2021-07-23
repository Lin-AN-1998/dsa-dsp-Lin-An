# here on web page totally we have three functions:
# 1, single prediction
# (predict button: prediction; add button: add this customer's information into prediction history)
# 2, multiple prediction
# 3, Prediction view
# (for this function, the data added in single prediction part will be displayed here.)
import pandas as pd
import streamlit as st
# EDA
import matplotlib.pyplot as plt
import matplotlib
import base64
matplotlib.use('Agg')
# DB
import sqlite3
from pandas import DataFrame

import ModelTraining

conn = sqlite3.connect('data.db')
c = conn.cursor()

from PIL import Image



# functions

def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS predict_table(CreditScore INT, Age INT, Tenure INT, Balance FLOAT, '
              'NumOfProducts INT, EstimatedSalary FLOAT, Surname , Geography OBJECT, Gender OBJECT, '
              'HasCrCard INT, IsActiveMember INT)')


def add_data(CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary, Surname, Geography, Gender, HasCrCard,
             IsActiveMember):
    c.execute('INSERT INTO predict_table(CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary,Surname, '
              'Geography, Gender, HasCrCard, IsActiveMember) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
              (CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary, Surname, Geography, Gender, HasCrCard,
               IsActiveMember))
    conn.commit()


def view_all_notes():
    c.execute('SELECT * FROM predict_table')
    data = c.fetchall()
    return data


################### Display stuff
# from inference import predict
def main():
    st.title('Bank Customer Churn Prediction')
    create_table()
    Menu = ["Home", "Single Prediction", "Multiple Prediction", "Prediction View"]
    choice = st.sidebar.selectbox("Menu", Menu)
    if choice == "Home":
        st.subheader("Welcome to Prediction World!")
        st.markdown("This application will help you to predict customer churn")
        image = Image.open('photo.jpg')
        st.image(image)

    if choice == "Single Prediction":
        st.subheader("Single Prediction")

        # Input your information
        # Numeric Variables
        CreditScore = st.number_input('Please input your CreditScore', min_value=0, max_value=850, step=1)
        Age = st.number_input('Please input your Age', min_value=18, max_value=92, step=1)
        Tenure = st.number_input('Please input your Tenure', min_value=0, max_value=10, step=1)
        Balance = st.number_input('Please input your Balance', min_value=0, max_value=251000)
        EstimatedSalary = st.number_input('Please input your EstimatedSalary', min_value=11, max_value=200000)
        NumOfProductslist = [1, 2, 3, 4]
        NumOfProducts = st.selectbox('Please input your NumOfProducts', NumOfProductslist)
        Surname = st.text_input('Please input your Surname')
        Geographylist = ["France", "Germany", "Spain"]
        Geography = st.selectbox("Select your geography", Geographylist)
        Genderlist = ["Male", "Female"]
        Gender = st.selectbox("Select your gender", Genderlist)
        HasCrCardlist = [1, 0]
        HasCrCard = st.selectbox("Do you have credit card?", HasCrCardlist)
        IsActiveMemberlist = [1, 0]
        IsActiveMember = st.selectbox("Do you have an active account?", IsActiveMemberlist)

        # Multi line text
        Note = st.text_area('Enter your notes here')
        # execute model
        if st.button('Predict customer churn'):
            # single_df

            data = {'Age': [Age], 'CreditScore': [CreditScore], 'Tenure': [Tenure], 'Balance': [Balance],
                    'NumOfProducts': [NumOfProducts], 'HasCrCard': [HasCrCard], 'IsActiveMember': [IsActiveMember],
                    'EstimatedSalary': [EstimatedSalary], 'Geography': [Geography], 'Gender': [Gender]}
            single_df = pd.DataFrame(data)
            # single_df.dtypes
            encode_single = ModelTraining.encode_data(ModelTraining.enc, single_df)
            # encode_single
            prediction = ModelTraining.predict(ModelTraining.model, encode_single)
            # prediction
            if prediction == 1:
                st.markdown('This customer will not be lost in the bank!')
            else:
                st.markdown('We might lose this customer!')

        # add data
        if st.button('Add'):
            add_data(CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary, Surname, Geography, Gender,
                     HasCrCard, IsActiveMember)
            st.success("Customer:{} information saved".format(Surname))

    if choice == "Multiple Prediction":
        st.subheader("Multiple Prediction")
        csv_file = st.file_uploader('Choose a CSV file')
        if csv_file:
            df_test = ModelTraining.read_data('Test.csv')
            df_test = ModelTraining.process(df_test)
            encode_df_test = ModelTraining.encode_data(ModelTraining.enc, df_test)
            df_total_test = ModelTraining.pre_process(encode_df_test)

        # execute model
        if st.button('Predict customer churn'):
            if csv_file is not None:
                st.subheader("Output")
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

                st.dataframe(data=out_df.style.highlight_max(subset=['Exited_prediction'], color='green'), width=4000,
                             height=700)
                st.markdown("Date represents customers will churn or not ")

            else:
                st.warning('You need to upload a csv file before')

    if choice == "Prediction View":
        st.subheader("View predictions history")
        result = view_all_notes()
        clean_db = pd.DataFrame(result, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                                                 'EstimatedSalary', 'Surname', 'Geography', 'Gender', 'HasCrCard',
                                                 'IsActiveMember'])
        # st.dataframe(clean_db)
        # clean_db.dtypes
        # int_list = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        clean_db['Age'] = clean_db['Age'].astype(str).astype(int)
        clean_db['Tenure'] = clean_db['Tenure'].astype(str).astype(int)
        clean_db['Balance'] = clean_db['Balance'].astype(float)
        clean_db['NumOfProducts'] = clean_db['NumOfProducts'].astype(str).astype(int)
        clean_db['EstimatedSalary'] = clean_db['EstimatedSalary'].astype(float)

        clean_df = clean_db.drop(['Surname'], axis=1)
        #clean_df

        clean_df.fillna(clean_df.select_dtypes(include='object').mode().iloc[0], inplace=True)
        clean_df.fillna(clean_df.select_dtypes(include='number').mean().iloc[0], inplace=True)
        encode_clean_df = ModelTraining.encode_data(ModelTraining.enc, clean_df)
        # encode_clean_df
        prediction = ModelTraining.predict(ModelTraining.model, encode_clean_df)
        # prediction
        clean_db['Prediction'] = prediction
        # clean_db
        # st.dataframe(clean_db)
        st.dataframe(data=clean_db.style.highlight_max(subset=['Prediction'], color='green'), width=4000,
                     height=700)


if __name__ == '__main__':
    main()
