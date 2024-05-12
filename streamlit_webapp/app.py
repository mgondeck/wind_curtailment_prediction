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
st.sidebar.page_link(page="/pages/data_app.py", label="Data")#, icon="🤖")
st.sidebar.page_link(page="/pages/baseline_app.py", label="Baseline")
st.sidebar.page_link(page="/pages/advanced_model_app.py", label="Advanced Model")
st.sidebar.markdown("---")

# title
st.title("Predict wind energy curtailment 💨")

########################################################
########## POWER PLANT LANGENHORN GEOPLOT
########################################################

# get CSV file from Google Drive sharing link
map_url = 'https://drive.google.com/file/d/1P83-639-LZKvgvzzVvG9fk6UUsdYWceR/view?usp=sharing'
map_path = 'https://drive.google.com/uc?id=' + map_url.split('/')[-2]

# Load CSV file with custom function for caching
map_df = load_data(map_path, ';')
map_df = map_df.rename(columns={"long": "lon"})

st.subheader("Windmill location in Langenhorn, Germany")
st.map(map_df, size=50, color='#0044ff', zoom=12)

st.markdown("---")