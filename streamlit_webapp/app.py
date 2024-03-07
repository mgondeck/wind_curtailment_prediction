import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

#from utils import load_data
from statsmodels.graphics.tsaplots import plot_acf

#creating pages
st.sidebar.page_link(page="app.py", label="Home", icon="🏠")
st.sidebar.page_link(page="/pages/eda_app.py", label="EDA", icon="🤖")

#setting title
st.title("Datensportverein 🕺🏽💃🏼")

# Adding logo image
st.sidebar.markdown("---")
st.sidebar.image('cow.jpeg', width=200)  # Adjust the width as needed

# Provide the Google Drive sharing link to the CSV file
url = 'https://drive.google.com/file/d/1YKpvt4VQCmPfUWfb528Qzju0vhcoWmqP/view?usp=sharing'

# Construct the direct download link for the file
path = 'https://drive.google.com/uc?id=' + url.split('/')[-2]

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(path, sep = ';', index_col=0)

# Display the DataFrame
df_sample = df.sample(n=10, random_state=42)
st.subheader("Overview of raw wind curtailment data 💨")
st.dataframe(df_sample)

st.markdown("---")
# Provide the Google Drive sharing link to the CSV file
map_url = 'https://drive.google.com/file/d/1P83-639-LZKvgvzzVvG9fk6UUsdYWceR/view?usp=sharing'

# Construct the direct download link for the file
map_path = 'https://drive.google.com/uc?id=' + map_url.split('/')[-2]

# Load the CSV file into a Pandas DataFrame
map_df = pd.read_csv(map_path,sep = ';')
map_df = map_df.rename(columns={"long": "lon"})

st.subheader("Geo plot of the windmills in Langenhorn, Germany")
st.map(map_df, size=50, color='#0044ff', zoom=12)


#home page - intro about the problem
#eda page 
#autocorrelation of redispatch