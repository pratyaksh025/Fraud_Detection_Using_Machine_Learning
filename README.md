# Fraud Detection Using Machine Learning

## Overview
This project builds a **fraud detection system** for financial transactions. The goal is to identify potentially fraudulent transactions using machine learning, while handling data imbalance and ensuring features are meaningful. The system uses **Logistic Regression** as a baseline model and **Random Forest** for improved performance.

---

## Dataset
The dataset contains transaction records with the following columns:
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud


- `step`: Time step of the transaction.  
- `type`: Transaction type (payment, transfer, etc.).  
- `amount`: Transaction amount.  
- `oldbalanceOrg` & `newbalanceOrig`: Sender’s account balances before and after transaction.  
- `oldbalanceDest` & `newbalanceDest`: Receiver’s account balances before and after transaction.  
- `isFraud`: Label indicating fraudulent transaction (1) or not (0).  
- `isFlaggedFraud`: Flagged for further monitoring.  

---

## Steps Performed

### **1. Data Cleaning & Exploration**
- Checked for missing values and duplicates (none found).  
- **Analyzed zero balances for senders and receivers:**
```python
df['isZeroOldBalance'] = (df['oldbalanceOrg'] == 0).astype(int)
df['isZeroNewBalanceDest'] = (df['newbalanceDest'] == 0).astype(int)
```
- **Explored amount distribution and applied log transformation:**
```python
df['log_amount'] = np.log1p(df['amount'])
```

### 2. Feature Engineering

- **Created features to capture transaction anomalies:**
``` python
df['diffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
df['diffDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
```
- **Dropped irrelevant columns and encoded categorical features:**
``` python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["type"] = le.fit_transform(df["type"])
```

### **3. Preparing Data for Modeling**
- **Split data into features and target:**
``` python
X = df.drop(["isFraud","isFlaggedFraud"], axis=1)
y = df["isFraud"]
```
- **Train-test split with stratification:**
``` python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
```
- **Standardized features**:
``` python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
### **4. Handling Class Imbalance**
- **Applied SMOTE to balance rare fraud cases:**
``` python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
```

### 5. **Model Training**
- **Logistic Regression (Baseline):**
``` python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)
```
- **Random Forest (Advanced Model):**
``` python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42
)
rf.fit(X_train_res, y_train_res)
```

### 6. Model Evaluation
- **Predictions and probabilities:**
``` python
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]
```

- **Metrics:**
``` python
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(confusion_matrix(y_test, y_pred))
```
- Outcome: Random Forest achieved near-perfect detection (ROC-AUC: 0.999) and significant improvement over Logistic Regression:
```
precision    recall  f1-score   support

           0       1.00      1.00      1.00   1906322
           1       0.96      1.00      0.98      2464

    accuracy                           1.00   1908786
   macro avg       0.98      1.00      0.99   1908786
weighted avg       1.00      1.00      1.00   1908786

ROC-AUC: 0.9991785674321287
[[1906225      97]
 [      9    2455]]
```

### **7. Key Insights**

- Most predictive features: `errorBalanceOrig`, `diffDest`, `log_amount`.

- Zero balances and large transaction differences strongly indicate fraudulent activity.

- SMOTE was critical for detecting rare fraud cases.

### **8. Next Steps**

- Deploy Random Forest in **real-time monitoring**.

- Retrain with new transactions to maintain accuracy.

- Integrate alert systems for unusual activity.

- Use feature importance to guide fraud prevention rules.

### **9. Tools & Libraries**

- Python, Pandas, NumPy, Scikit-learn, imbalanced-learn, Matplotlib, Seaborn

- Jupyter Notebook, Google Colab


  


df['isZeroOldBalance'] = (df['oldbalanceOrg'] == 0).astype(int)
df['isZeroNewBalanceDest'] = (df['newbalanceDest'] == 0).astype(int)
