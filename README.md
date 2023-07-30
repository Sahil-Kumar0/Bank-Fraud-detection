# Bank-Fraud-detection
Bank Fraud detection


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load the dataset
data = pd.read_csv("creditcard.csv")

# Explore the dataset
print(data.head())
print(data.info())

# Check the distribution of the target variable
print(data['Class'].value_counts())

# Data preprocessing
X = data.drop(['Time', 'Class'], axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Building and Evaluation
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Neural Network': MLPClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc
    }

# Model Comparison and Analysis
results_df = pd.DataFrame(results).T
print(results_df)

# Best Model Selection
best_model = results_df['F1-Score'].idxmax()
print(f"The best model for fraud detection is: {best_model}")

# Business Recommendation
# Provide recommendations based on the best model's performance and potential benefits for the business.

# Web App Development with Streamlit (Optional)
# Create a Streamlit web app to demonstrate real-time fraud prediction using the best model.

# Conclusion
# Summarize the key findings and results of the project.

