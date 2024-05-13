import streamlit as st
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

df1 = df2 = None

st.markdown("<h1 style='text-align: center; color: black; margin-bottom: 50px;'>Fusion your data!</h1>", unsafe_allow_html=True)

accelemeter = st.file_uploader("Upload your accelemeter data", type=["csv"])
gyroscope = st.file_uploader("Upload your gyroscope data", type=["csv"])

if accelemeter and gyroscope:
    df1 = pd.read_csv(accelemeter)
    df2 = pd.read_csv(gyroscope)
    
    vector1 = df1[['x', 'y', 'z']].values[:100]
    vector2 = df2[['x', 'y', 'z']].values[:100]

    cross_product = np.cross(vector1, vector2)
    
    cross_product_df = pd.DataFrame(cross_product, columns=['x', 'y', 'z'])
    csv = cross_product_df.to_csv(index=False)

    st.download_button(
        label="Download cross product data",
        data=csv,
        file_name="cross_product_data.csv",
        mime="text/csv",
    )
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cross_product_df['x'], cross_product_df['y'], cross_product_df['z'])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    st.pyplot(fig)