import os
import preprocess
import train_model

# Define paths
data_path = "data/Telco-Customer-Churn.csv"
processed_data_path = "data/processed_data.csv"

if __name__ == "__main__":
    print("\n🔹 Step 1: Data Preprocessing")
    df = preprocess.load_data(data_path)
    df = preprocess.preprocess_data(df)
    df.to_csv(processed_data_path, index=False)
    print("✅ Data preprocessing completed.")

    print("\n🔹 Step 2: Model Training")
    os.system("python src/train_model.py")
    print("✅ Model training completed.")

    print("\n🔹 Step 3: Make Predictions")
    os.system("python src/predict.py")