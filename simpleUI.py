import streamlit as st
import joblib
import numpy as np


# Load the trained model, scaler, and polynomial features transformer
model = joblib.load("l.pkl")  # Replace with the correct path if needed
scaler = joblib.load("scaler.pkl")  # Replace with the correct path if needed
polynom = joblib.load("polynom.pkl")  # Replace with the correct path if needed

# Streamlit app layout
st.set_page_config(page_title="Water Potability Classification", page_icon="💧", layout="wide")
st.title("Water Quality Classification")
st.markdown("""This app classifies water quality as **Potable** or **Non-Potable** based on various water parameters. Fill in the values below to get the classification.""")

# Sidebar content with WHO standards and definitions
with st.sidebar:
    # Create an expander for the WHO Water Quality Standards section
    with st.expander("WHO Water Quality Standards"):
        st.markdown("""
            ### WHO Water Quality Standards:
            - **pH**: Ideal range 6.5 - 8.5
            - **Hardness (mg/L)**: No specific standard, but generally under 500 mg/L for taste and comfort.
            - **Total Solids (mg/L)**: Less than 500 mg/L for potable water.
            - **Chloramines (mg/L)**: Maximum allowable level: 4 mg/L.
            - **Sulfate (mg/L)**: Maximum allowable level: 250 mg/L.
            - **Conductivity (µS/cm)**: No direct standard, but should be under 500 µS/cm for safe drinking water.
            - **Organic Carbon (mg/L)**: No specific limit defined, but should be low to avoid harmful microorganisms.
            - **Trihalomethanes (µg/L)**: Maximum allowable level: 100 µg/L.
            - **Turbidity (NTU)**: Less than 5 NTU for potable water.
        """)

    # Create a separate expander for the Parameter Definitions section
    with st.expander("Parameter Definitions"):
        st.markdown("""
            ### Parameter Definitions:
            - **pH**: A measure of how acidic or alkaline water is. A neutral pH is 7, acidic water has a pH less than 7, and alkaline water has a pH greater than 7.
            - **Hardness**: The concentration of dissolved minerals (mainly calcium and magnesium) in water. Hard water can cause scaling in pipes.
            - **Total Solids**: Refers to the total concentration of dissolved solids in water, typically measured as TDS (Total Dissolved Solids).
            - **Chloramines**: A disinfectant used in water treatment to eliminate bacteria and other harmful microorganisms.
            - **Sulfate**: A natural compound found in water that can affect the taste of water. High levels can be a health concern.
            - **Conductivity**: A measure of water's ability to conduct electricity, which correlates with the amount of dissolved ions in the water.
            - **Organic Carbon**: Organic compounds in water that could be present due to natural or anthropogenic processes. High organic carbon levels can lead to the growth of harmful bacteria.
            - **Trihalomethanes**: Chemicals that form when chlorine disinfectants react with organic materials in the water. High levels of THMs can be harmful to health.
            - **Turbidity**: The measure of the clarity of water. High turbidity indicates a high presence of suspended particles in the water.
        """)

# Input fields for user to enter water quality parameters
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=150.0, step=1.0)
solids = st.number_input("Total Solids (mg/L)", min_value=0.0, value=200.0, step=1.0)
chloramines = st.number_input("Chloramines (mg/L)", min_value=0.0, value=1.0, step=0.1)
sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=30.0, step=1.0)
conductivity = st.number_input("Conductivity (µS/cm)", min_value=0.0, value=200.0, step=1.0)
organic_carbon = st.number_input("Organic Carbon (mg/L)", min_value=0.0, value=10.0, step=0.1)
trihalomethanes = st.number_input("Trihalomethanes (µg/L)", min_value=0.0, value=50.0, step=1.0)
turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=1.0, step=0.1)

# Button to make prediction
if st.button("Classify Water Quality"):
    # Prepare the input data for prediction
    input_data_test = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity,
                                 organic_carbon, trihalomethanes, turbidity]])

    # Apply the same transformations as during training
    input_data_transformed = polynom.transform(input_data_test)  # Polynomial features transformation
    input_data_scaled = scaler.transform(input_data_transformed)  # Apply MinMax scaling

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display the result
    if prediction[0] == 1:
        st.success("The water is **Potable**.", icon="✅")
    else:
        st.error("The water is **Not Potable**.", icon="❌")