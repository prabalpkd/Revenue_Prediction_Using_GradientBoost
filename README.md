# 📈 Revenue Prediction using Machine Learning

This project aims to predict **Revenue ($)** based on **Ad Spend ($)** and **Season** using a simple dataset. It walks through the complete machine learning workflow: from data visualization and preprocessing to model training and performance evaluation.

---

## 📂 Dataset Overview

The dataset contains **200 rows** and **3 columns**:
- `Ad Spend ($)`
- `Season` (categorical: Summer, Monsoon, Winter)
- `Revenue ($)`

---

## 🔍 Exploratory Data Analysis (EDA)

### 📊 1. Bar Graph: Ad Spend & Revenue per Season
Used bar graphs to compare average **Ad Spend** and **Revenue** across different **Seasons**.

### ⚫ 2. Scatter Plot: Ad Spend vs Revenue (Grouped by Season)
Plotted scatter plots to explore the relationship between **Ad Spend ($)** and **Revenue ($)** for each Season.

---

## 🧹 Data Preprocessing

- Applied **One-Hot Encoding** to convert the categorical `Season` column into numeric format using `pd.get_dummies()`.
- Used `train_test_split()` to divide the data into training and test sets.
- Final features used for model training:
  - `Ad Spend ($)`
  - `Season_Summer`
  - `Season_Winter`
- Target variable: `Revenue ($)`

---

## 🤖 Model Building

### 🔷 Linear Regression
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)  # ≈ 0.88
