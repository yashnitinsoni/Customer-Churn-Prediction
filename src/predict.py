import pandas as pd
import joblib

def load_model(model_path):
    """Load a trained model and return it."""
    return joblib.load(model_path)

def predict_churn(model, input_data):
    """Predict churn based on input features."""
    
    # Load feature names from the trained model
    trained_features = model.feature_names_in_

    # Convert input dictionary to DataFrame, ensuring column order matches training data
    input_df = pd.DataFrame([input_data])[trained_features]

    # Make prediction
    prediction = model.predict(input_df)
    return "Churn" if prediction[0] == 1 else "No Churn"

if __name__ == "__main__":
    # Load trained model
    model = load_model("/Users/yashsoni/Library/CloudStorage/OneDrive-SharedLibraries-Onedrive/Project 2 - Customer Churn Prediction/models/logistic_regression.pkl")

    # Example input (Ensure all features exist with correct names)
    input_data = {
        'tenure': 12, 
        'MonthlyCharges': 50, 
        'TotalCharges': 600, 
        'gender': 1, 
        'SeniorCitizen': 0, 
        'Partner': 1, 
        'Dependents': 0, 
        'PhoneService': 1,
        'MultipleLines': 0,
        'InternetService': 2,
        'OnlineSecurity': 1,
        'OnlineBackup': 1,
        'DeviceProtection': 0,
        'TechSupport': 0,
        'StreamingTV': 1,
        'StreamingMovies': 1,
        'Contract': 2,
        'PaperlessBilling': 1,
        'PaymentMethod': 3
    }

    # Predict churn
    result = predict_churn(model, input_data)
    print(f"Prediction: {result}")