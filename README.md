# 📉 Customer Churn Prediction

> A Machine Learning project to predict customer churn in a Telecom company using the Telco Customer Churn dataset.

---

## 📖 Overview

This project builds a Machine Learning model to predict whether a telecom customer will **churn** (i.e., stop using the service) based on their account information, demographics, and service usage patterns.

By identifying at-risk customers early, businesses can take **proactive retention actions** such as offering discounts, better plans, or personalized support — ultimately reducing revenue loss.

---

## ❓ Problem Statement

Telecom companies spend significantly more on **acquiring new customers** than retaining existing ones. Customer churn directly impacts revenue and growth. The goal of this project is:

> **To build a binary classification model that predicts whether a customer will churn (Yes/No) based on their features, helping the business to take timely action.**

---

## 🔍 What is Customer Churn?

**Customer Churn** (also called customer attrition) refers to the phenomenon where customers **stop doing business** with a company within a certain time period.

### Types of Churn:
| Type | Description |
|------|-------------|
| **Voluntary Churn** | Customer actively decides to leave (e.g., switches to competitor) |
| **Involuntary Churn** | Customer leaves due to payment failure or contract expiry |

### Why Does Churn Happen?
- Poor customer service
- Better offers from competitors
- High pricing or billing issues
- Lack of features or value
- Contract end without renewal

### Business Impact:
- Acquiring a new customer costs **5–7x more** than retaining an existing one.
- Even a **5% reduction in churn** can increase profit by **25–95%**.
- Churn prediction enables companies to focus retention efforts on **high-risk customers**.

---

## 📊 Dataset Description

**Dataset:** IBM Telco Customer Churn Dataset  
**File:** `Telco-Customer-Churn.csv`  
**Rows:** ~7,043 customers  
**Columns:** 21 features + 1 target variable

### Features:

| Feature | Type | Description |
|--------|------|-------------|
| `customerID` | Object | Unique customer identifier |
| `gender` | Categorical | Male / Female |
| `SeniorCitizen` | Binary | Whether the customer is a senior citizen (1 = Yes, 0 = No) |
| `Partner` | Categorical | Whether the customer has a partner |
| `Dependents` | Categorical | Whether the customer has dependents |
| `tenure` | Numerical | Number of months the customer has stayed |
| `PhoneService` | Categorical | Whether the customer has phone service |
| `MultipleLines` | Categorical | Whether the customer has multiple lines |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | Categorical | Whether the customer has online security |
| `OnlineBackup` | Categorical | Whether the customer has online backup |
| `DeviceProtection` | Categorical | Whether the customer has device protection |
| `TechSupport` | Categorical | Whether the customer has tech support |
| `StreamingTV` | Categorical | Whether the customer streams TV |
| `StreamingMovies` | Categorical | Whether the customer streams movies |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | Categorical | Whether the customer uses paperless billing |
| `PaymentMethod` | Categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | Numerical | Monthly amount charged to the customer |
| `TotalCharges` | Numerical | Total amount charged over tenure |
| **`Churn`** | **Target** | **Whether the customer churned (Yes/No)** |

---

## 🔄 Project Workflow

The project follows the standard **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology:

```
1. Data Collection
       ↓
2. Exploratory Data Analysis (EDA)
       ↓
3. Data Preprocessing
       ↓
4. Feature Engineering
       ↓
5. Model Building
       ↓
6. Model Evaluation
       ↓
7. Conclusion & Insights
```

### Step-by-Step Explanation:

#### 1️⃣ Data Collection
- The Telco Customer Churn CSV dataset is loaded using **Pandas**.
- Initial shape, data types, and sample rows are inspected.

#### 2️⃣ Exploratory Data Analysis (EDA)
EDA helps understand the data distribution, relationships, and patterns before modeling.

- **Univariate Analysis:** Distribution of individual features (histograms, count plots)
- **Bivariate Analysis:** Relationship between features and the target variable `Churn`
- **Correlation Heatmap:** Understanding feature correlations
- **Churn Rate Analysis:** Examining which groups have higher churn rates
  - Customers with **month-to-month contracts** churn more
  - Customers with **higher monthly charges** are more likely to churn
  - Customers with **shorter tenure** have higher churn rates

#### 3️⃣ Data Preprocessing
Raw data needs to be cleaned and prepared before feeding into a model.

- **Handling Missing Values:** `TotalCharges` contains blank strings that are converted to NaN and filled/dropped.
- **Data Type Conversion:** `TotalCharges` converted from object to float.
- **Encoding Categorical Variables:**
  - Binary columns (Yes/No) → Label Encoding (0/1)
  - Multi-category columns → One-Hot Encoding
- **Feature Scaling:** Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) are normalized using **StandardScaler** or **MinMaxScaler** to bring them to the same range.

#### 4️⃣ Feature Engineering
- Dropping irrelevant columns like `customerID`.
- Creating new features if applicable (e.g., tenure groups).
- Addressing **class imbalance** since the dataset has more non-churned customers (~73%) than churned (~27%).

#### 5️⃣ Model Building
Multiple classification algorithms are trained and compared.

#### 6️⃣ Model Evaluation
Models are evaluated using appropriate metrics for classification.

---

## 🤖 Algorithms Used

### 1. Logistic Regression
A statistical model used for **binary classification**. It estimates the probability that a given observation belongs to a category (Churn = Yes or No) using the **sigmoid function**.

- **Pros:** Simple, interpretable, fast
- **Best when:** Features have a roughly linear relationship with the log-odds of the outcome

### 2. Decision Tree Classifier
A tree-structured model that splits data at each node based on the **best feature and threshold** to minimize impurity (Gini Index or Entropy).

- **Pros:** Easy to visualize and interpret, handles non-linear data
- **Cons:** Prone to overfitting on training data

### 3. Random Forest Classifier
An **ensemble method** that builds multiple Decision Trees and combines their outputs through **majority voting**, reducing overfitting.

- **Pros:** High accuracy, robust to noise, handles missing values well
- **Cons:** Less interpretable, slower for prediction

### 4. Support Vector Machine (SVM)
Finds the **optimal hyperplane** that best separates churn vs. non-churn customers, maximizing the margin between classes. Uses kernels (linear, RBF) to handle non-linear boundaries.

- **Pros:** Effective in high-dimensional spaces
- **Cons:** Slow to train on larger datasets

### 5. K-Nearest Neighbors (KNN)
Classifies a data point based on the **majority class of its K nearest neighbors** in the feature space.

- **Pros:** Simple, no training phase
- **Cons:** Slow for large datasets, sensitive to feature scaling

### 6. Ensemble & Boosting Methods
This project goes further than single models and compares a wide range of ensemble techniques:

| Model | Core Idea |
|---|---|
| **Extra Trees** | Like Random Forest, but splits are chosen randomly rather than optimally, adding more variance reduction |
| **Gradient Boosting** | Sequentially builds trees, each correcting the residual errors of the previous one |
| **AdaBoost** | Boosts weak learners by giving more weight to misclassified samples in each round |
| **XGBoost** | Regularized, optimized gradient boosting — faster and less prone to overfitting |
| **LightGBM** | Histogram-based gradient boosting, very fast on large datasets |
| **CatBoost** | Gradient boosting optimized for categorical features with built-in handling — no need for one-hot encoding |
| **Naive Bayes** | Applies Bayes' theorem assuming feature independence — fast probabilistic baseline |
| **LDA / QDA** | Linear/Quadratic Discriminant Analysis — models class distributions using Gaussian assumptions |

### 7. Hyperparameter Tuning — GridSearchCV
Every model above is tuned using **`GridSearchCV`** with **5-fold cross-validation**, exhaustively testing combinations of hyperparameters (e.g., `max_depth`, `learning_rate`, `n_estimators`) to find the configuration with the best accuracy on validation folds.

### 8. Deep Dive — CatBoost (Final Tuned Model)
After comparing all 14 models, the notebook performs a deeper, dedicated tuning pass on **CatBoost**, including:
- Native handling of categorical features (`cat_features`) without manual encoding
- **Early stopping** (50 rounds) to prevent overfitting, monitored via F1-score on a validation set
- **SHAP (SHapley Additive exPlanations)** values to interpret which features drive individual predictions
- **Threshold tuning** — instead of the default 0.5 probability cutoff, the notebook searches for the threshold that maximizes F1-score, since churn datasets are imbalanced

---

## 📏 Evaluation Metrics

Since this is a **binary classification** problem with **class imbalance**, accuracy alone is not sufficient.

### Confusion Matrix
|  | Predicted: No Churn | Predicted: Churn |
|--|--|--|
| **Actual: No Churn** | True Negative (TN) | False Positive (FP) |
| **Actual: Churn** | False Negative (FN) | True Positive (TP) |

### Key Metrics:

| Metric | Formula | Description |
|--------|---------|-------------|
| **Accuracy** | (TP + TN) / Total | Overall correct predictions |
| **Precision** | TP / (TP + FP) | Of predicted churns, how many were correct |
| **Recall (Sensitivity)** | TP / (TP + FN) | Of actual churns, how many were detected |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of Precision and Recall |
| **ROC-AUC Score** | Area under ROC Curve | Overall model discrimination ability |

> **In churn prediction, Recall is more important** — missing a churner (False Negative) is more costly than falsely flagging a loyal customer.

---

## 🛠️ Technologies Used

| Tool/Library | Purpose |
|---|---|
| **Python 3.x** | Programming language |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical operations |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical data visualization |
| **Scikit-learn** | ML models, preprocessing, `GridSearchCV`, metrics |
| **XGBoost** | Gradient boosting classifier |
| **LightGBM** | Fast histogram-based gradient boosting |
| **CatBoost** | Gradient boosting with native categorical feature support |
| **SHAP** | Model interpretability / feature importance |
| **Google Colab / Jupyter Notebook** | Interactive development environment |

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── CUSTOMER_CHURN_PREDICTION.ipynb   # Main Jupyter/Colab Notebook
├── Telco-Customer-Churn.csv          # Dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv)
└── README.md                         # Project documentation
```

---

## ▶️ How to Run

### Prerequisites

Make sure you have **Python 3.x** and the following libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter xgboost lightgbm catboost shap
```

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ravi-attada/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open the notebook:**
   ```
   Open: CUSTOMER_CHURN_PREDICTION.ipynb
   ```

4. **Run all cells** from top to bottom using `Kernel → Restart & Run All`.

---

## 📈 Results

All 14 models were trained with `GridSearchCV` (5-fold CV) and evaluated on the held-out test set:

| Rank | Model | Test Accuracy |
|------|-------|----------------|
| 🥇 1 | **CatBoost** | **81.80%** |
| 🥈 2 | LightGBM | 81.28% |
| 🥉 3 | GradientBoosting | 81.09% |
| 4 | AdaBoost | 81.04% |
| 4 | XGBoost | 81.04% |
| 6 | ExtraTrees | 81.00% |
| 7 | LogisticRegression | 80.76% |
| 8 | RandomForest | 80.57% |
| 9 | LDA | 80.05% |
| 10 | SVM | 79.91% |
| 11 | DecisionTree | 78.44% |
| 12 | QDA | 76.21% |
| 13 | NaiveBayes | 76.40% |
| 14 | KNeighbors | 76.11% |

**🏆 Best Model: CatBoost** — Accuracy: **81.80%**, F1-Score: **0.6145**

### Final Tuned CatBoost Model (deep-dive with early stopping)

| Metric | Class 0 (No Churn) | Class 1 (Churn) |
|--------|---------------------|-------------------|
| Precision | 0.84 | 0.68 |
| Recall | 0.91 | 0.53 |
| F1-Score | 0.87 | 0.59 |

- **Overall Accuracy:** 81%
- **ROC-AUC Score:** 0.8476

**After threshold tuning** (optimizing for F1 instead of the default 0.5 cutoff, best threshold ≈ 0.313):

| Metric | Class 0 (No Churn) | Class 1 (Churn) |
|--------|---------------------|-------------------|
| Precision | 0.90 | 0.56 |
| Recall | 0.78 | 0.77 |
| F1-Score | 0.84 | 0.64 |

> Lowering the decision threshold trades some precision for a big jump in **Recall on churners (53% → 77%)** — meaning the model catches far more actual churn cases, which matters more for retention campaigns even if it flags a few more false positives.

---

## ✅ Conclusion

- Out of **14 machine learning algorithms** compared using `GridSearchCV`, **CatBoost** achieved the best test accuracy (**81.80%**), narrowly ahead of LightGBM and GradientBoosting — confirming that gradient-boosted trees handle this tabular, mixed-type dataset best.
- A dedicated tuning pass on CatBoost with **early stopping** and **SHAP-based interpretability** was used to both improve robustness and explain individual predictions.
- Because only **~53% of actual churners** were caught at the default 0.5 threshold, **threshold tuning** was applied — shifting the cutoff to ~0.31 raised churn-recall to **77%**, a critical improvement for a business that wants to catch as many at-risk customers as possible, even at some cost to precision.
- Key factors influencing churn (from EDA and SHAP analysis) typically include:
  - **Contract type** — Month-to-month customers churn far more than those on 1–2 year contracts
  - **Tenure** — Newer customers are at higher risk of leaving
  - **Monthly Charges** — Higher charges correlate with increased churn
  - **Internet Service type** — Fiber optic users tend to churn more than DSL users
- Businesses can use these insights, along with the tuned CatBoost model, to design **targeted retention campaigns** for at-risk segments — e.g., offering contract upgrades or discounts to month-to-month, high-charge, low-tenure customers.


---

## 🙋‍♂️ Author

**Ravi Attada**  
GitHub: [@Ravi-attada](https://github.com/Ravi-attada)

