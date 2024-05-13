import streamlit as st
import pandas as pd
import numpy as np
from pykalman import KalmanFilter

df1 = df2 = None

st.markdown("<h1 style='text-align: center; color: black; margin-bottom: 50px;'>Fusion your data!</h1>", unsafe_allow_html=True)

accelemeter = st.file_uploader("Upload your accelemeter data", type=["csv"])
gyroscope = st.file_uploader("Upload your gyroscope data", type=["csv"])

if accelemeter and gyroscope:
    df1 = pd.read_csv(accelemeter)
    df2 = pd.read_csv(gyroscope)

    merged_data = pd.merge(df1, df2, on='x', how='inner')
    