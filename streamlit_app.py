import streamlit as st
import pandas as pd
from helper import regression
import pickle



def app_interface(df):
    st.write('Machine Learning Configuration')
    config_dict={}
    model_type=st.selectbox('Select your Model',['Linear Regression'])
    features=st.multiselect('Select the Features',list(df.columns))
    predictor=st.selectbox('Choose the column to predict',list(df.columns))
    
    
    if st.button('Start Model'):
        model=''
        
        df=df[features].copy()
        if model_type=='Linear Regression':
            model,acc=regression(df,predictor)
            st.write(acc)    
            st.download_button(
                        "Download Model",
                        data=pickle.dumps(model),
                        file_name="model.pkl",
                    )
        
        
        
        
        


def app():
    st.header('Machine Learning Model Trainer')
    uploaded_file = st.file_uploader("Choose a file",type='.csv')
    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file)
        app_interface(df)
        
    
    

if __name__== '__main__':
    app()
