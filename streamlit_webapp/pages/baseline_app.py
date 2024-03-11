import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from utils import load_data, baseline, load_model

# sidebar 
st.sidebar.page_link(page="app.py", label="Home", icon="🏠")
st.sidebar.page_link(page="/pages/data_app.py", label="Data")#, icon="🤖")
st.sidebar.page_link(page="/pages/baseline_app.py", label="Baseline")
st.sidebar.page_link(page="/pages/advanced_model_app.py", label="Advanced Model")
st.sidebar.markdown("---")
st.sidebar.image('cow.jpeg', width=200) 
st.sidebar.write("Datensportverein 🕺🏽💃🏼")

# title
st.title("Baseline Model")

########################################################
########## LOAD JOINED DATAFRAME (curtailment_target_features.csv)
########################################################

# get CSV file from Google Drive sharing link
url = 'https://drive.google.com/file/d/1YKpvt4VQCmPfUWfb528Qzju0vhcoWmqP/view?usp=sharing'
path = 'https://drive.google.com/uc?id=' + url.split('/')[-2]

# Load CSV file with custom function for caching
df = load_data(path, ';')

########################################################
########## LOAD JOINED DATAFRAME (lagged_curtailment_target_features.csv)
########################################################

# get CSV file from Google Drive sharing link
url = 'https://drive.google.com/file/d/1P8rzcsPwg8Ci3ZJQrHopffPTOpFcO-NI/view?usp=sharing'
path = 'https://drive.google.com/uc?id=' + url.split('/')[-2]

# Load CSV file with custom function for caching
df_lagged = load_data(path, ';')

########################################################
########## BASELINE - BOXPLOT WIND GUST MAX
########################################################

df_baseline = df.copy()
df_baseline = df_baseline[['redispatch', 'wind_gust_max_m/s']]
mean_value = df_baseline['wind_gust_max_m/s'].mean()
df_baseline['wind_gust_max_m/s'].fillna(value=mean_value, inplace=True)

df_boxplot = df_baseline.copy()

st.subheader('Boxplot Visualization of Wind Gust Max Values')

boxplot_data = []
for category, group in df_boxplot.groupby('redispatch'):
    boxplot_data.append(go.Box(y=group['wind_gust_max_m/s'], name=str(category)))

boxplot_fig = go.Figure(data=boxplot_data)

boxplot_fig.update_layout(
    xaxis=dict(title="Curtailment"),
    yaxis=dict(title="Count")
)

st.plotly_chart(boxplot_fig)

st.markdown("---")

########################################################
########## ADVANCED - LOAD MODEL
########################################################

# load model
#url = 'https://drive.google.com/file/d/1n04qpt4QHVirKR2guw3XZilwHcNX9Y1V/view?usp=sharing' # extra treees
url = 'https://drive.google.com/file/d/1wmR_E3S29Ux9vb8Ak6aQHvy2M-8NZVyu/view?usp=sharing' #xgboost
model_path = 'https://drive.google.com/uc?id=' + url.split('/')[-2]

xgboost_class = load_model(model_path)

########################################################
########## BASELINE & ADVANCED - PLOT RESULTS
########################################################

st.subheader('True & Predicted Curtailment Values')

# select test datset
start_pred = '2023-07-01'
end_pred = '2023-12-31'

# baseline
df_baseline = df[['redispatch', 'wind_gust_max_m/s']].copy()
df_baseline.index = pd.to_datetime(df_baseline.index)
df_basemodel_pred = baseline(df_baseline, start_pred, end_pred)

# advanced model
df_advanced = df_lagged.copy()
X_advanced = df_advanced.loc[start_pred:end_pred].drop(['redispatch', 'level'], axis=1)
y_pred_advanced = xgboost_class.predict(X_advanced)

# slider
min_value = pd.Timestamp(df_basemodel_pred.index.min())
max_value = pd.Timestamp(df_basemodel_pred.index.max())
start = pd.Timestamp('2023-08-01').to_pydatetime()
end = pd.Timestamp('2023-08-15').to_pydatetime()

start_vis = st.slider('Start', min_value= min_value, max_value = max_value, value=start)
end_vis = st.slider('End', min_value=min_value, max_value= max_value, value=end)

# visualisation
st.markdown("---")

# combine model outputs for vis
df_basemodel_pred['y_pred_advanced'] = y_pred_advanced
df_model_pred = df_basemodel_pred
df_model_pred_vis = df_model_pred.loc[start_vis:end_vis] 
df_model_pred_vis['y_pred_advanced'] = df_model_pred_vis['y_pred_advanced'] - 2 # shift advanced model
df_model_pred_vis['shifted_redispatch'] = df_model_pred_vis['redispatch'] + 2 # shift true values

# traces
trace_true = go.Scatter(x=df_model_pred_vis.index, y=df_model_pred_vis['shifted_redispatch'], mode='markers', name='True Values', marker=dict(color='blue', symbol='circle'))
trace_base = go.Scatter(x=df_model_pred_vis.index, y=df_model_pred_vis['y_pred_baseline'], mode='markers', name='Prediction Baseline', marker=dict(color='orange', symbol='circle'))
trace_adv = go.Scatter(x=df_model_pred_vis.index, y=df_model_pred_vis['y_pred_advanced'], mode='markers', name='Prediction Advanced', marker=dict(color='red', symbol='circle'))


# layout
layout = go.Layout(
    #title="Comparison of True and Predicted Curtailment Status",
    xaxis=dict(title="Date", range=[df_model_pred_vis.index[0], df_model_pred_vis.index[-1]]),
    yaxis=dict(title="Curtailment", showticklabels=False),  
    showlegend=True,
    legend=dict(x=1, y=1),
    margin=dict(l=50, r=50, t=50, b=50),
)

fig = go.Figure(data=[trace_true, trace_base, trace_adv], layout=layout)
st.plotly_chart(fig)

st.markdown("---")




