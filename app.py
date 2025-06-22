import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# 1) Load model and preprocessors
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 2) Build Streamlit UI
st.title('Customer Churn Prediction')

geography        = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender           = st.selectbox('Gender',    label_encoder_gender.classes_)
age              = st.slider('Age', 18, 92)
balance          = st.number_input('Balance',         min_value=0.0)
credit_score     = st.number_input('Credit Score',    min_value=0.0)
estimated_salary = st.number_input('Estimated Salary',min_value=0.0)
tenure           = st.slider('Tenure', 0, 10)
num_of_products  = st.slider('Number of Products', 1, 4)
has_cr_card      = st.selectbox('Has Credit Card',  [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# 3) Encode inputs
gender_encoded = label_encoder_gender.transform([gender])[0]

# Fixed: Remove .toarray() since the encoder returns a dense array
geo_encoded = onehot_encoder_geo.transform([[geography]])

# 4) Assemble DataFrame
base_df = pd.DataFrame({
    'CreditScore':     [credit_score],
    'Gender':          [gender_encoded],
    'Age':             [age],
    'Tenure':          [tenure],
    'Balance':         [balance],
    'NumOfProducts':   [num_of_products],
    'HasCrCard':       [has_cr_card],
    'IsActiveMember':  [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

X = pd.concat([base_df, geo_df], axis=1)

# 5) Scale and predict
X_scaled   = scaler.transform(X)
pred_proba = model.predict(X_scaled)[0][0]

# 6) Display result
st.write(f'**Churn Probability:** {pred_proba:.2f}')
if pred_proba > 0.5:
    st.error("🚨 The customer is likely to churn.")
else:
    st.success("✅ The customer is not likely to churn.")