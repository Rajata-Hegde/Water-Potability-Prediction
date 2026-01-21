import joblib
import numpy as np

# Load the trained model, scaler, and polynomial features transformer
model = joblib.load("l.pkl")  # Replace with the correct path if needed
scaler = joblib.load("scaler.pkl")  # Replace with the correct path if needed
polynom = joblib.load("polynom.pkl")  # Replace with the correct path if needed

# Input data to test (replace with your specific data)
input_data_test = np.array([[  
    6.80,    # ph (slightly acidic, borderline between potable and non-potable)
    200.00,  # Hardness (medium range, neither too hard nor soft)
    500.00,  # Solids (medium TDS, not too high but not very low either)
    2.50,    # Chloramines (moderate level, within acceptable range but higher than typical potable)
    50.00,   # Sulfate (medium concentration, may be slightly high but not extreme)
    400.00,  # Conductivity (medium, reasonable for potable water)
    10.00,   # Organic_carbon (moderate level)
    50.00,   # Trihalomethanes (moderate concentration)
    5.00     # Turbidity (moderate, indicating slight cloudiness but still usable)
]])

# Apply the same transformations as during training
input_data_transformed = polynom.transform(input_data_test)  # Polynomial features transformation
input_data_scaled = scaler.transform(input_data_transformed)  # Apply MinMax scaling

# Make prediction
prediction = model.predict(input_data_scaled)

# Output the result
if prediction[0] == 1:
    print("The water is Potable.")
else:
    print("The water is Not Potable.")
