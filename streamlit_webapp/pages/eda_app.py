import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from statsmodels.graphics.tsaplots import plot_acf

st.title("Exploratory data analysis of wind curtailment data")

# Provide the Google Drive sharing link to the CSV file
url = 'https://drive.google.com/file/d/1YKpvt4VQCmPfUWfb528Qzju0vhcoWmqP/view?usp=sharing'

# Construct the direct download link for the file
path = 'https://drive.google.com/uc?id=' + url.split('/')[-2]

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(path, sep = ';', index_col=0)

# Calculate value counts
value_counts = df['redispatch'].value_counts()

# Create a bar chart
imbalance_fig = go.Figure(go.Bar(
    x=value_counts.index,
    y=value_counts.values,
    marker_color=['blue', 'red']
))

# Update layout
imbalance_fig.update_layout(
    xaxis_title='Redispatch Value',
    yaxis_title='Count of Instances',
    title='Plot of Class Imbalance'
)

# Display the plotly figure using Streamlit
st.subheader("Distribution of redispatch values")
st.plotly_chart(imbalance_fig)


#plotting redispatch over different time intervals
df_year = df.loc['2022-01-01':'2022-12-31']
df_month = df.loc['2022-06-01':'2022-06-31']
df_week = df.loc['2022-06-01':'2022-06-07']
df_day = df.loc['2022-06-02 00:00:00':'2022-06-02 23:45:00']

# Create a selectbox for choosing the interval
interval = st.selectbox('Select interval:', ['Year', 'Month', 'Week', 'Day'])

# Define the DataFrame based on the selected interval
if interval == 'Year':
    df_selected = df_year
elif interval == 'Month':
    df_selected = df_month
elif interval == 'Week':
    df_selected = df_week
elif interval == 'Day':
    df_selected = df_day

# Create a Plotly figure
fig = go.Figure()

# Add a trace for the redispatch data
fig.add_trace(go.Scatter(x=df_selected.index, y=df_selected['redispatch'], mode='lines', name='Redispatch'))

# Customize the layout
fig.update_layout(
    title=f'Plot of Redispatch for {interval.lower()}',
    xaxis_title='Date',
    yaxis_title='Redispatch',
    xaxis=dict(showgrid=True, tickformat='%Y-%m-%d'),  # Customize x-axis format
    yaxis=dict(showgrid=True),
)

# Display the Plotly figure using Streamlit
st.plotly_chart(fig)


# Calculate autocorrelation function (ACF)
acf = pd.Series(data=df['redispatch']).autocorr()

# Calculate ACF values
lags = 50
acf_values = [1] + [pd.Series(data=df['redispatch']).autocorr(lag) for lag in range(1, lags+1)]

# Create colors for bars based on ACF values
colors = ['rgb(0,0,255)'] * (lags + 1)
for i in range(len(acf_values)):
    if acf_values[i] > 0:
        colors[i] = f'rgb({int(255 * acf_values[i])},0,0)'
    else:
        colors[i] = f'rgb(0, {int(255 * abs(acf_values[i]))}, 0)'

# Create bar chart figure
fig = go.Figure(data=[go.Bar(x=np.arange(lags+1), y=acf_values, marker_color=colors)])

# Update layout
fig.update_layout(
    title='Autocorrelation Function (ACF)',
    xaxis_title='Lag',
    yaxis_title='Autocorrelation',
    bargap=0.2,
    xaxis=dict(tickmode='array', tickvals=list(range(lags+1))),
    showlegend=False
)

# Display plot
st.plotly_chart(fig)

