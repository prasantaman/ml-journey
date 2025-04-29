import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load('model.pkl')

# Create a scaler and fit with dummy mean/std (NOTE: use your actual training scaler if saved)
# Ideally, you should also save and load the scaler, but here is a workaround:
scaler = StandardScaler()
# Fit on dummy data just to define structure (replace with actual train data for real use)
dummy_data = np.array([[6.0, 120.0], [7.0, 130.0]])
scaler.fit(dummy_data)

# App title
st.title("Student Placement Predictor")

# Input fields
cgpa = st.number_input("Enter CGPA of the student", min_value=0.0, max_value=10.0, value=6.5)
iq = st.number_input("Enter IQ of the student", min_value=50.0, max_value=200.0, value=120.0)

# Prediction
if st.button("Predict Placement"):
    input_data = np.array([[cgpa, iq]])
    input_scaled = scaler.transform(input_data)  # scale input as done during training
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("🎉 Student is likely to be PLACED!")
    else:
        st.error("❌ Student is likely NOT to be placed.")
