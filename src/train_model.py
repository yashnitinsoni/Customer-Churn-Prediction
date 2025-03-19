import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
df = pd.read_csv("/Users/yashsoni/Library/CloudStorage/OneDrive-SharedLibraries-Onedrive/Project 2 - Customer Churn Prediction/data/processed_data.csv")

# Split features and target
X = df.drop(columns=["Churn"])  # Features
y = df["Churn"]  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
joblib.dump(log_model, "/Users/yashsoni/Library/CloudStorage/OneDrive-SharedLibraries-Onedrive/Project 2 - Customer Churn Prediction/models/logistic_regression.pkl")

# Train Decision Tree
tree_model = DecisionTreeClassifier(max_depth=5)
tree_model.fit(X_train, y_train)
joblib.dump(tree_model, "/Users/yashsoni/Library/CloudStorage/OneDrive-SharedLibraries-Onedrive/Project 2 - Customer Churn Prediction/models/decision_tree.pkl")

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "/Users/yashsoni/Library/CloudStorage/OneDrive-SharedLibraries-Onedrive/Project 2 - Customer Churn Prediction/models/random_forest.pkl")

# Train SVM (for small datasets, linear kernel is best)
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, "/Users/yashsoni/Library/CloudStorage/OneDrive-SharedLibraries-Onedrive/Project 2 - Customer Churn Prediction/models/svm.pkl")

# Evaluate models
models = {
    "Logistic Regression": log_model,
    "Decision Tree": tree_model,
    "Random Forest": rf_model,
    "Support Vector Machine": svm_model
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))