import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from utils import load_data

# sidebar 
st.sidebar.page_link(page="app.py", label="Home", icon="🏠")
st.sidebar.page_link(page="/pages/data_app.py", label="Data")
st.sidebar.page_link(page="/pages/baseline_app.py", label="Baseline")
st.sidebar.page_link(page="/pages/advanced_model_app.py", label="Advanced Model")
st.sidebar.markdown("---")

# title
st.title("Advanced Model")

########################################################
########## LOAD JOINED DATAFRAME (curtailment_target_features.csv)
########################################################

# get CSV file from Google Drive sharing link
url = 'https://drive.google.com/file/d/1YKpvt4VQCmPfUWfb528Qzju0vhcoWmqP/view?usp=sharing'
path = 'https://drive.google.com/uc?id=' + url.split('/')[-2]

# Load CSV file with custom function for caching
df = load_data(path, ';')

########################################################
########## LAGGED DATA
########################################################

df_lag = df.copy()
df_lag.drop(['level'], axis = 1, inplace=True)

name_mapping = {
    'redispatch': 'Curtailment',
    'wind_speed_m/s': 'Wind Speed [m/s]', 
    'wind_direction_degrees': 'Wind Direction Degrees',
    'radiation_global_J/m2': 'Radiation Global [J/m2]', 
    'air_temperature_K': 'Air Temperature [K]', 
    'humidity_percent': 'Humidity [%]',
    'wind_gust_max_m/s': 'Wind Gust Max [m/s]', 
    'wind_direction_gust_max_degrees': 'Wind Direction Gust Max Degrees',
    'forecast_solar_MW': 'Forecast Solar [MW]', 
    'total_grid_load_MWh': 'Total Grid Load [MWh]',
    'residual_load_MWh': 'Residual Load [MWh]', 
    'pumped_storage_MWh': 'Pumped Storage [MWh]' 
}
df_lag = df_lag.rename(columns=name_mapping)

df_lagged = pd.DataFrame(index=df_lag.index)
for feature in df_lag.columns: 
    df_lagged[feature] = df_lag[feature]
    df_lagged[feature + '_lag_1'] = df_lag[feature].shift(1)
    df_lagged[feature + '_lag_2'] = df_lag[feature].shift(2)

df_lagged.dropna(inplace = True) # maybe better ways
df_lagged.drop(['Curtailment_lag'], axis=1, inplace = True)
df_lagged.index.name = None
df_lagged = df_lagged.round(2)

# Display
st.subheader("Lagged Curtailment Data 💨")
st.dataframe(df_lagged.sample(n=10, random_state=42))

st.markdown("---")