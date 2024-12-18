import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Streamlit App
st.title("MNIST Digit Recognizer using KNN")

# File Upload
st.sidebar.header("Upload Dataset")
train_file = st.sidebar.file_uploader("Upload Training Data (CSV Format)", type=["csv"])
test_file = st.sidebar.file_uploader("Upload Test Data (CSV Format)", type=["csv"])

if train_file and test_file:
    # Load Data
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # Reduce size for testing
    train = train.head(1000)

    st.write("### Training Data Preview")
    st.dataframe(train.head())

    st.write("### Test Data Preview")
    st.dataframe(test.head())

    # Data Preprocessing
    X_train = np.array(train.iloc[:, 1:], dtype=np.float32)
    Y_train = np.array(train.iloc[:, 0])
    X_test = np.array(test, dtype=np.float32)

    # Visualize a sample digit
    st.write("### Sample Digit from Training Data")
    sample_index = st.slider("Select a sample index", 0, len(train) - 1, 0)
    sample_image = X_train[sample_index].reshape(28, 28)
    st.write(f"Label: {Y_train[sample_index]}")
    fig, ax = plt.subplots()
    ax.imshow(sample_image, cmap='gray')
    st.pyplot(fig)

    # Train Model
    st.write("### Train KNN Model")
    n_neighbors = st.slider("Select number of neighbors (k)", 1, 10, 5)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, Y_train)
    st.success("Model trained successfully!")

    # Prediction
    st.write("### Make a Prediction")
    test_index = st.slider("Select a test sample index", 0, len(test) - 1, 0)
    test_image = X_test[test_index].reshape(28, 28)
    predicted_label = knn.predict(X_test[test_index].reshape(1, -1))[0]

    st.write(f"Predicted Label: {predicted_label}")
    fig, ax = plt.subplots()
    ax.imshow(test_image, cmap='gray')
    st.pyplot(fig)

else:
    st.warning("Please upload both training and test datasets.")
