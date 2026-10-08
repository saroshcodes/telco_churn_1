from pathlib import Path

import pickle
import pandas as pd
import streamlit as st

st.title("Customer cancellation prediction")
st.write("Enter a customer's subscription details to estimate their chance of cancelling.")
with Path(__file__).with_name("model.pkl").open("rb") as file:
    model = pickle.load(file)

with st.form("customer_details"):
    tenure = st.number_input("Customer tenure (months)", min_value=0, max_value=72, value=12)
    charge = st.number_input(
        "Monthly charge (USD)", min_value=18.25, max_value=118.75,
        value=70.0, step=0.01, format="%.2f",
    )
    contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No internet service"])
    submitted = st.form_submit_button("Predict cancellation")

if submitted:
    inputs = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": charge,
        "Contract": contract,
        "InternetService": "No" if internet == "No internet service" else internet,
    }])
    probability = model.predict_proba(inputs)[0, 1]
    st.metric("Estimated chance of cancellation", f"{probability:.1%}")
    st.write("Predicted outcome:", "Likely to cancel" if probability >= 0.5 else "Likely to stay")
    st.caption("This is a model estimate based on sample data, not a certainty.")
