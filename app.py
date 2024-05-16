import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

df1 = df2 = None

st.markdown("<h1 style='text-align: center; color: black; margin-bottom: 50px;'>Enter your data!</h1>", unsafe_allow_html=True)

accelemeter = st.file_uploader("Upload your accelemeter data", type=["csv"])
gyroscope = st.file_uploader("Upload your gyroscope data", type=["csv"])

if accelemeter and gyroscope:
    # Read the data
    df1 = pd.read_csv(accelemeter)
    df2 = pd.read_csv(gyroscope)

    # Take the first 100 entries of the data 
    vector1 = df1[['x', 'y', 'z']].values[:100]
    vector2 = df2[['x', 'y', 'z']].values[:100]
    print(vector1[0])
    print(vector2[0])
    # Cross product the values to get
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

    class KalmanFilter(object):
        def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

            if(F is None or H is None):
                raise ValueError("Set proper system dynamics.")

            self.n = F.shape[1]
            self.m = H.shape[1]

            self.F = F
            self.H = H # Observation Matrix
            self.B = 0 if B is None else B
            self.Q = np.eye(self.n) if Q is None else Q
            self.R = np.eye(self.n) if R is None else R # Covariance in measurement
            self.P = np.eye(self.n) if P is None else P # Covariance in estimate
            self.x = np.zeros((self.n, 1)) if x0 is None else x0 # Current state estimation

        def predict(self, u = 0):
            self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
            self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
            return self.x

        def update(self, z):
            y = z - np.dot(self.H, self.x)
            S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
            self.x = self.x + np.dot(K, y)
            I = np.eye(self.n)
            self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
                (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
    dt = 1.0/60
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)

    x = np.linspace(-10, 10, 100)
    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)

    tab1, tab2 = st.tabs(["Accelerometer Data", "Gyroscope Data"])

    tab1.subheader("Kalman Filtered Data")

    predictions = []

    data = vector1
    for z in data:
        predictions.append(np.dot(H,  kf.predict())[0])
        kf.update(z)

    for i in range(0, df1.shape[1] - 1):
        measurements = [subarray[i] for subarray in data[1:]]
        preds = [subarray[0] for subarray in predictions[1:]]
        chartData = pd.DataFrame(
            {
                "Predictions": preds,
                "Measured": measurements

            }
        )
        tab1.line_chart(chartData)

    tab2.subheader("Kalman Filtered Data")
    predictions2 = []

    data2 = vector2
    for z in data2:
        predictions2.append(np.dot(H,  kf.predict())[0])
        kf.update(z)

    for i in range(0, df2.shape[1] - 1):
        measurements = [subarray[i] for subarray in data2[1:]]
        preds = [subarray[0] for subarray in predictions2[1:]]
        chartData = pd.DataFrame(
            {
                "Predictions": preds,
                "Measured": measurements

            }
        )
        tab2.line_chart(chartData)

    st.write("Great Job")