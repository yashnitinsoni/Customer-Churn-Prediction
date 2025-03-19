# Customer Churn Prediction for a Telecom Provider

### Machine Learning project to predict customer churn using Logistic Regression, Decision Tree, Random Forest, and SVM.

## Project Overview
Customer churn prediction helps telecom providers identify customers who are likely to leave their service. This project uses **machine learning models** to classify whether a customer will churn or stay based on different factors like **contract type, monthly charges, and tenure**.

## Dataset
- **Source**: IBM Telco Customer Churn Dataset
- **Size**: 7,000+ customer records
- **Target Variable**: `Churn` (1 = Churned, 0 = Not Churned)

## Tech Stack
- **Programming Language**: Python 🐍
- **Libraries Used**: `pandas`, `numpy`, `sklearn`, `joblib`, `matplotlib`
- **Models**: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM)

## Exploratory Data Analysis (EDA)
✔ Data visualization of customer behavior 📊  
✔ Feature correlation analysis 🔗  
✔ Data preprocessing: missing values handling, encoding categorical features  

## Model Performance
| Model                  | Accuracy |
|------------------------|----------|
| **Logistic Regression**    | 81.76%   |
| **Decision Tree**         | 79.56%   |
| **Random Forest**        | 80.06%   |
| **Support Vector Machine** | 81.76%   |

**Best Models**: Logistic Regression & SVM  
**Next Steps**: Implement XGBoost and Hyperparameter Tuning.

---