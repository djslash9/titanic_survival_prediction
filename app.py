#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
#from PIL import Image

#load the model from disk
import joblib
model = joblib.load(r"./notebook/model.sav")

#Import python scripts
from preprocessing import preprocess

def main():
    #Setting Application title
    st.title('Titanic Ship Wreck Prediction App')

      #Setting Application description
    st.markdown("""
     :dart:  This online app has been made to predict about the people who survived the Titanic disaster.
    The application is functional for both online prediction and batch data prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    #image = Image.open('App.jpg')
    add_selectbox = st.sidebar.selectbox(
	"How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Titanic Survivals')
    #st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")
        #Based on our optimal features selection
        st.subheader("Demographic data")
        name = st.text_input('Name of the person')
        pclass = st.slider('Pclass', min_value=1, max_value=3, value=2)
        sex = st.radio('Sex:', ('male', 'female'))
        age = st.slider('Age', min_value=14, max_value=72, value=22)
        sibSp = st.slider('SibSp', min_value=0, max_value=8, value=4)
        parch = st.slider('Parch', min_value=0, max_value=6, value=3)
        embarked = st.radio('Embarked:', ('C', 'Q', 'S'))
                
        data = {
            'Pclass' : pclass,
            'Sex' : sex,
            'Age' : age,
            'SibSp' : sibSp,
            'Parch' : parch,
            'Embarked' : embarked
        }
        
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)

        #Preprocess inputs
        preprocess_df = preprocess(features_df, 'Online')

        prediction = model.predict(preprocess_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the persong might be died.')
                st.write(name, 'is dead')
            else:
                st.success('No, the person might be alive.')
                st.write(name, 'is alive')
        

    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            
            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            data.dropna(inplace=True)
            if st.button('Predict'):
                #Get batch prediction
                preprocess_df = preprocess(data, "Batch")
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Yes, the person is not alive.', 
                                                    0:'No, the the person is alive.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)
            
if __name__ == '__main__':
        main()
