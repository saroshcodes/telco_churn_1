from pathlib import Path
import math
import pickle

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
    service = "No" if internet == "No internet service" else internet
    values = [
        (tenure - model["mean"][0]) / model["scale"][0],
        (charge - model["mean"][1]) / model["scale"][1],
    ]
    values += [int(contract == choice) for choice in model["contracts"]]
    values += [int(service == choice) for choice in model["internet_services"]]
    score = model["intercept"] + sum(weight * value for weight, value in zip(model["weights"], values))
    probability = 1 / (1 + math.exp(-score))
    st.metric("Estimated chance of cancellation", f"{probability:.1%}")
    st.write("Predicted outcome:", "Likely to cancel" if probability >= 0.5 else "Likely to stay")
    st.caption("This is a model estimate based on sample data, not a certainty.")
