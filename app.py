import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("model.pkl")

st.title("House Price Prediction App")
st.write("Predict house price by using a Linear Regression model.")

# Input box
area = st.number_input("Enter the area of the house (sqft) :", min_value=500, step=100 )
bedrooms = st.number_input("Enter the number of bedrooms :", min_value=0, max_value=10)
bathrooms = st.number_input("Enter the number of bathrooms :", min_value=0, max_value=10)
stories = st.number_input("Enter the number of stories :", min_value=0, max_value=10)
parking = st.number_input("Enter the number of parking :",min_value=0, max_value=10)

# Prediction button
if st.button("Predict Price"):
    # Reshape and predict
    input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])
    prediction = model.predict(input_data)

    st.success(f"Predicted House Price: ₹{float(prediction[0]):,.2f}")

st.caption("Model trained using Linear Regression on area vs prie data.")