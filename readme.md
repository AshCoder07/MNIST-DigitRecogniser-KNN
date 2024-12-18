# MNIST Digit Recognizer using KNN

This is a simple web application built using Streamlit that allows users to upload a dataset of handwritten digits and predict the labels of test samples using the K-Nearest Neighbors (KNN) algorithm.

## Features

- **Upload Dataset**: Upload training and test datasets in CSV format.
- **Data Preview**: View the first few rows of the training and test datasets.
- **Sample Digit Visualization**: Select a sample digit from the training data and view it along with its label.
- **KNN Model**: Train a KNN classifier on the uploaded training data and make predictions on test data.
- **Prediction**: View the predicted label for a selected test sample.

## Requirements

To run this app, you need the following Python packages:

- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install these dependencies using `pip`:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
