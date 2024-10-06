import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained churn model
model = joblib.load('churn_model.pkl')

# Define categorical options
gender_options = ['Male', 'Female']
contract_options = ['Month-to-month', 'One year', 'Two year']
internet_service_options = ['DSL', 'Fiber optic', 'No']

# Streamlit UI
st.title('Churn Prediction App')
st.write("Enter the customer data to predict churn:")

# Input fields for categorical data
gender = st.selectbox('Gender', gender_options)
contract = st.selectbox('Contract', contract_options)
internet_service = st.selectbox('Internet Service', internet_service_options)

# Map categorical values to numerical (this should match your training data preprocessing)
data = {
    'gender': gender,
    'contract': contract,
    'internet_service': internet_service,
}

# Convert categorical inputs into one-hot encoded format (use the same method as in training)
input_data = pd.DataFrame([data])

# One-hot encode the input data (assuming you used pandas get_dummies during training)
input_data_encoded = pd.get_dummies(input_data).reindex(columns=model.feature_names_in_, fill_value=0)

# Predict button
if st.button('Predict'):
    prediction = model.predict(input_data_encoded)
    st.write(f'The predicted churn status is: {"Churn" if prediction[0] == 1 else "No Churn"}')
