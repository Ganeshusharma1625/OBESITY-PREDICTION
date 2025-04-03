print("Starting Prediction Script...")  # Debugging line

import joblib
import pandas as pd
import numpy as np

# Load trained model and encoders
model = joblib.load("obesity_model.pkl")  # Ensure this file exists
encoder = joblib.load("encoder.pkl")  # One-hot encoder
label_encoder = joblib.load("label_encoder.pkl")  # Label encoder

print("Model and encoders loaded successfully!")  # Debugging line

# Function to take user input
def get_user_input():
    print("Enter the following details:")
    Age = float(input("Age: "))
    Height = float(input("Height (m): "))
    Weight = float(input("Weight (kg): "))
    Gender = input("Gender (Male/Female): ")
    FAVC = input("Frequent consumption of high-caloric food? (yes/no): ")

    categorical_input = pd.DataFrame([[Gender, FAVC]], columns=['Gender', 'FAVC'])
    encoded_categorical = encoder.transform(categorical_input)

    numerical_input = np.array([Age, Height, Weight]).reshape(1, -1)
    final_input = np.hstack((numerical_input, encoded_categorical))

    return final_input

# Prediction function
def predict():
    print("Starting prediction...")  # Debugging line
    user_data = get_user_input()
    prediction = model.predict(user_data)
    result = label_encoder.inverse_transform(prediction)
    print(f"Predicted Obesity Category: {result[0]}")

if __name__ == "__main__":
    predict()
