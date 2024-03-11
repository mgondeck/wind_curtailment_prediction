import streamlit as st
import pandas as pd
import pickle as pk
from imblearn.over_sampling import SMOTE
import joblib
import requests
from io import BytesIO

@st.cache_data
def load_data(data: str, sep, index = 0) -> pd.DataFrame:
    """Loads a dataset from a CSV file and returns it as a Pandas DataFrame."""
    df = pd.read_csv(data, sep = sep, index_col = index)
    return df

@st.cache_resource
def load_model(url):
    """loads a model from a file path."""
    response = requests.get(url)
    model_file = BytesIO(response.content)
    model = joblib.load(model_file)
    return model

@st.cache_data
def smote_data(X, y):
    """Generates more samples of the minority class using SMOTE."""
    smote = SMOTE(random_state=13, k_neighbors=3)
    X_smoted, y_smoted = smote.fit_resample(X, y)
    return X_smoted, y_smoted

@st.cache_data
def date_interval(df, start, end):
    """returns part of the given dataframe based on specified time interval"""
    df_interval = df.loc[start:end]
    return df_interval

@st.cache_data
def baseline(df, start_pred, end_pred):
    """tbd."""
    df_model_pred = df.loc[start_pred:end_pred]

    X_test = df_model_pred['wind_gust_max_m/s']
    y_test = df_model_pred['redispatch']

    y_pred = [1 if wind_gust_max > 9 else 0 for wind_gust_max in X_test]
    df_model_pred['y_pred_baseline'] = y_pred 

    return df_model_pred

