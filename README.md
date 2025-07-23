# 📈 Revenue Prediction using Gradient Boosting : Regression

This project aims to predict **Revenue ($)** based on **Ad Spend ($)** and **Season** using a simple dataset. The project walks through the complete machine learning workflow: from visualization and preprocessing to model training and evaluation.

---

## 📂 Dataset Overview

The dataset contains **200 rows** and **3 columns**:
- `Ad Spend ($)`
- `Season` (categorical: Summer, Monsoon, Winter)
- `Revenue ($)`

---

## 🔍 Exploratory Data Analysis (EDA)

### 📊 1. Bar Graph: Ad Spend & Revenue per Season
Visualized average Ad Spend and Revenue across different seasons to observe seasonal trends.

![Bar Plot](visuals/barplot.png)

### ⚫ 2. Scatter Plot: Ad Spend vs Revenue (per Season)
Plotted a scatter plot to examine the relationship between Ad Spend and Revenue, grouped by Season.

![Scatter Plot](visuals/scatterplot.png)

---

## 🧹 Data Preprocessing

- Applied **One-Hot Encoding** to convert the `Season` categorical variable into numeric format using `pd.get_dummies()`.
- Used `train_test_split()` to split data into training and testing sets (features: `Ad Spend`, `Season_Summer`, `Season_Winter`).

---

## 🤖 Model Building

### 🔷 Linear Regression

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)
# R2 Score ≈ 0.88

### 🌳 Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
model.score(X_test, y_test)
# R2 Score ≈ 0.91

🚀 Gradient Boosting Regression

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()
model.fit(X_train, y_train)
model.score(X_test, y_test)
# R2 Score ≈ 0.93

## 📈 Prediction Evaluation
Sample prediction comparison (Actual vs Predicted Revenue):
| Actual | Predicted |
| ------ | --------- |
| 605    | 668.07    |
| 1125   | 1052.14   |
| 699    | 637.09    |
| 504    | 434.48    |

### 🧰 Tools & Libraries Used

Python, Pandas, Matplotlib / Seaborn, Scikit-learn

### 📌 Key Learnings
1. Visualizing seasonal patterns helps understand marketing impact.

2. One-hot encoding is essential for handling categorical variables.

3. Ensemble methods like Gradient Boosting can significantly improve prediction accuracy.

### 🚀 Future Enhancements
1. Experiment with advanced models like XGBoost or CatBoost

2. Add cross-validation and hyperparameter tuning

3. Deploy the model using Streamlit or Flask for real-time prediction

### 🙋‍♂️ Author
Prabal Kumar Deka
