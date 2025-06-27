import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

#Load the dataset
data = pd.read_csv('C:/Users/jay30/OneDrive/Documents/myprojects/python/loan_data.csv')

#Dropping rows with missing values
data.dropna(inplace=True)

#Encode categorical variables
categorical_cols = [
    'person_gender', 'person_education', 'person_home_ownership',
    'loan_intent', 'previous_loan_defaults_on_file'
]
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

#Define features (X) and target (y)
X = data.drop('loan_status', axis=1)
y = data['loan_status']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

#Make predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

#Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
auc = roc_auc_score(y_test, y_proba)
print("ROC AUC Score:", auc)

#Plot ROC Curve with AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Plot Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_vals = np.linspace(-10, 10, 200)
sigmoid_vals = sigmoid(z_vals)

plt.figure(figsize=(6, 4))
plt.plot(z_vals, sigmoid_vals, color='teal', label='Sigmoid')
plt.axhline(0.5, color='red', linestyle='--', label='Default Threshold (0.5)')
plt.title('Sigmoid Function')
plt.xlabel('Linear input (z)')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Threshold tuning
threshold = 0.7
y_thresh_pred = (y_proba >= threshold).astype(int)

print(f"\nMetrics with threshold = {threshold}:")
print(confusion_matrix(y_test, y_thresh_pred))
print(classification_report(y_test, y_thresh_pred))
