import streamlit as st
import pandas as pd
import joblib

# Load the XGBoost model
model = joblib.load('Customer_Churned_.pkl')

# Define options for categorical inputs
internet_service_options = ['DSL', 'Fiber optic', 'No']
contract_options = ['Month-to-month', 'One year', 'Two year']
online_security_options = ['Yes', 'No']
phone_service_options = ['Yes', 'No']
streaming_movies_options = ['Yes', 'No']
tech_support_options = ['Yes', 'No']

# Streamlit UI for input
st.title('Churn Prediction App')

internet_service = st.selectbox('Internet Service', internet_service_options)
contract = st.selectbox('Contract', contract_options)
online_security = st.selectbox('Online Security', online_security_options)
phone_service = st.selectbox('Phone Service', phone_service_options)
streaming_movies = st.selectbox('Streaming Movies', streaming_movies_options)
tech_support = st.selectbox('Tech Support', tech_support_options)

# Prepare the input data for prediction
input_data = {
    'InternetService': internet_service,
    'Contract': contract,
    'OnlineSecurity': online_security,
    'PhoneService': phone_service,
    'StreamingMovies': streaming_movies,
    'TechSupport': tech_support,
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Apply One-Hot Encoding (same as training)
input_df_encoded = pd.get_dummies(input_df).reindex(columns=model.feature_names_in_, fill_value=0)

# Make a prediction
if st.button('Click here'):
    prediction = model.predict(input_df_encoded)
    st.write(f"The customer is {'likely to churn' if prediction[0] == 1 else 'not likely to churn'}")

