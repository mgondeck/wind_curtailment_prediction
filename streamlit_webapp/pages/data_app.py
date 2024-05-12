import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from utils import load_data, smote_data, date_interval

# sidebar 
st.sidebar.page_link(page="app.py", label="Home", icon="🏠")
st.sidebar.page_link(page="/pages/data_app.py", label="Data")
st.sidebar.page_link(page="/pages/baseline_app.py", label="Baseline")
st.sidebar.page_link(page="/pages/advanced_model_app.py", label="Advanced Model")
st.sidebar.markdown("---")

# title
st.title("Data")

########################################################
########## LOAD JOINED DATAFRAME (curtailment_target_features.csv)
########################################################

# get CSV file from Google Drive sharing link
url = 'https://drive.google.com/file/d/1YKpvt4VQCmPfUWfb528Qzju0vhcoWmqP/view?usp=sharing'
path = 'https://drive.google.com/uc?id=' + url.split('/')[-2]

# Load CSV file with custom function for caching
df = load_data(path, ';')

########################################################
########## DISPLAY TABLE
########################################################

df_display = df.copy()
df_display.drop(['level'], axis = 1, inplace=True)

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
df_display = df_display.rename(columns=name_mapping)
df_display.index.name = None
df_display = df_display.round(2)

# Display
st.subheader("Overview of Wind Curtailment Data 💨")

def display_random_rows():
    start_index = np.random.randint(100, 130000)
    st.dataframe(df_display.iloc[start_index:start_index + 10])

if st.button("Refresh"):
    display_random_rows()
else:
    display_random_rows()


st.markdown("---")

########################################################
########## CLASS IMBALANCE
########################################################

st.subheader("Imbalanced Distribution of Curtailment Values")
df_balance = df.copy()
df_balance = df_balance.fillna(0)

X = df_balance.drop('redispatch', axis=1) 
y = df_balance['redispatch'] 

# using customised function for caching
X_smoted, y_smoted = smote_data(X, y)

# count values of original and smoted target variable
value_counts = y.value_counts()
value_counts_smote = y_smoted.value_counts()

# checkbox
show_smoted = st.checkbox("Show SMOTE-Resampled")
value_counts_to_display = value_counts_smote if show_smoted else value_counts

# bar chart
imbalance_fig = go.Figure(go.Bar(
    x=value_counts_to_display.index,
    y=value_counts_to_display.values,
    marker_color=['blue', 'red']
))

# layout
imbalance_fig.update_layout(
    xaxis_title='Curtailment Value',
    yaxis_title='Count',
    #title='Plot of Class Imbalance'
)

# display
st.plotly_chart(imbalance_fig)
st.markdown("---")

########################################################
########## CURTAILMENT OVER TIME
########################################################

st.subheader("Curtailment Values Over Time")
df_seasonality = df.copy()

#choosing specified time intervals and returning dataframe with customized cached function
df_year = date_interval(df_seasonality, '2022-01-01','2022-12-31')
df_month = date_interval(df_seasonality, '2022-06-01','2022-06-31')
df_week = date_interval(df_seasonality, '2022-06-01','2022-06-07')
df_day = date_interval(df_seasonality, '2022-06-02 00:00:00','2022-06-02 23:45:00')

# Create a selectbox for choosing the interval
interval = st.selectbox('Select interval:', ['Year [2022]', 'Month [2022/06]', 'Week [2022/06/01-07]', 'Day [2022/06/02]'])

if interval == 'Year [2022]':
    df_selected = df_year
elif interval == 'Month [2022/06]':
    df_selected = df_month
elif interval == 'Week [2022/06/01-07]':
    df_selected = df_week
elif interval == 'Day [2022/06/02]':
    df_selected = df_day

# trace and layout
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_selected.index, y=df_selected['redispatch'], mode='lines', name='Redispatch'))
fig.update_layout(
    title=f'Plot of Redispatch for {interval.lower()}',
    xaxis_title='Date',
    yaxis_title='Curtailment',
    xaxis=dict(showgrid=True, tickformat='%Y-%m-%d'),  # Customize x-axis format
    yaxis=dict(showgrid=True),
)

# display
st.plotly_chart(fig)
st.markdown("---")

########################################################
########## AUTOCORRELATION
########################################################

## Calculate autocorrelation function (ACF)
#acf = pd.Series(data=df['redispatch']).autocorr()

## Calculate ACF values
#lags = 50
#acf_values = [1] + [pd.Series(data=df['redispatch']).autocorr(lag) for lag in range(1, lags+1)]

## Create colors for bars based on ACF values
#colors = ['rgb(0,0,255)'] * (lags + 1)
#for i in range(len(acf_values)):
#    if acf_values[i] > 0:
#        colors[i] = f'rgb({int(255 * acf_values[i])},0,0)'
#    else:
#        colors[i] = f'rgb(0, {int(255 * abs(acf_values[i]))}, 0)'

## Create bar chart figure
#fig = go.Figure(data=[go.Bar(x=np.arange(lags+1), y=acf_values, marker_color=colors)])

## Update layout
#fig.update_layout(
#    title='Autocorrelation Function (ACF)',
#    xaxis_title='Lag',
#    yaxis_title='Autocorrelation',
#    bargap=0.2,
#    xaxis=dict(tickmode='array', tickvals=list(range(lags+1))),
#    showlegend=False
#)

## Display plot
#st.plotly_chart(fig)

#st.markdown("---")
