# Loan Approval Prediction with Logistic Regression

This project builds a **binary classification model** using **Logistic Regression** to predict whether a loan application will be approved or not based on user details such as income, employment, credit score, etc.

---

## 🧠 Objective

- **Task**: Build a binary classifier using logistic regression.
- **Goal**: Predict the likelihood of a loan getting approved (1 = Approved, 0 = Rejected).

---

## 🛠️ Tools & Libraries

- `Python`
- `pandas` – for data loading and manipulation
- `scikit-learn` – for machine learning algorithms and preprocessing
- `matplotlib` – for visualizations
- `numpy` – for numerical operations

---

## 📂 Dataset

The dataset used is a CSV file named:  
`loan_data.csv`  
Located at: "loan_data.csv"

---

Features include:
- Personal details (age, gender, education)
- Loan characteristics (amount, interest rate, intent)
- Credit history
- Target column: `loan_status` (0 = Rejected, 1 = Approved)

---

## 🔄 Workflow

1. **Load and clean dataset**  
2. **Encode categorical variables**
3. **Split data into training and test sets**
4. **Scale features using StandardScaler**
5. **Train Logistic Regression model**
6. **Evaluate using accuracy, precision, recall, ROC-AUC**
7. **Visualize ROC Curve and Sigmoid Function**
8. **Tune classification threshold**

---

## 📊 Outputs

- **Accuracy**: ~89.9%
- **Confusion Matrix & Classification Report**
- **ROC Curve** with AUC value
- **Sigmoid Curve** to explain probability-to-class conversion

---

## 📈 Sample Visualizations

### ROC Curve:
Shows model performance at different thresholds.
- AUC (Area Under Curve) indicates how well the model separates classes.

### ⚙️ Sigmoid Function:

The sigmoid function is the core of logistic regression. It maps any real-valued number to a value between 0 and 1, making it ideal for binary classification.
The function is defined as:

##  σ(z) = 1 / (1 + e^(-z)) ##

z is the linear combination of input features and model weights.
Output of σ(z) represents the predicted probability that a sample belongs to class 1.

📌 Example:
If σ(z) = 0.8, then there's an 80% probability that the instance belongs to class 1.
The default threshold is 0.5, meaning:
If σ(z) ≥ 0.5 → Predict class 1 (e.g., loan approved)
If σ(z) < 0.5 → Predict class 0 (e.g., loan rejected)

🧩 Why It's Important:
The sigmoid curve helps visualize how confident the model is in its predictions.

You can tune the threshold (e.g., use 0.7 instead of 0.5) to adjust sensitivity based on the problem.
Visual representation of how logistic regression maps inputs to probability between 0 and 1.

---
