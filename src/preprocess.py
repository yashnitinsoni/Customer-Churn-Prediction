import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    """Load dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Clean and preprocess the data."""
    # Drop customerID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert 'TotalCharges' to numeric (handle missing values)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[:, 'TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Encode categorical variables
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Standardize numerical features
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

if __name__ == "__main__":
    df = load_data("/Users/yashsoni/Library/CloudStorage/OneDrive-SharedLibraries-Onedrive/Project 2 - Customer Churn Prediction/data/Telco-Customer-Churn.csv")
    df = preprocess_data(df)
    df.to_csv("/Users/yashsoni/Library/CloudStorage/OneDrive-SharedLibraries-Onedrive/Project 2 - Customer Churn Prediction//data/processed_data.csv", index=False)
    print("✅ Data preprocessing completed!")