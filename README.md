# 🧠 Student Mental Health Score Prediction

A machine learning regression project that predicts a student's **Mental Health Score** using demographic, academic, social-media usage, lifestyle, sleep, physical activity, and stress-related features.

The project demonstrates an end-to-end machine learning workflow including **EDA, data validation, data cleaning, feature engineering, preprocessing pipelines, categorical encoding, feature scaling, regression modeling, RandomizedSearchCV hyperparameter tuning, model comparison, evaluation, and model serialization**.

> **Project Type:** Supervised Machine Learning — Regression
> **Target:** `Mental_Health_Score`
> **Dataset Size:** 5,000 records × 13 columns
> **Best evaluated model:** Random Forest Regressor
> **Evaluation:** R², MAE, MSE, RMSE

---

## 📌 Project Overview

Student mental health can be influenced by several academic, lifestyle, and digital-behavior factors.

This project uses machine learning to estimate a student's **Mental Health Score** from variables such as:

* Age
* Gender
* Country
* Academic level
* Most-used social media platform
* Purpose of social-media use
* Average daily usage hours
* Daily device unlocks
* Study hours
* Physical activity hours
* Sleep hours
* Stress level

The project treats `Mental_Health_Score` as a **continuous numerical target**, making this a **regression problem**.

---

## 🎯 Objectives

The main objectives of this project are:

1. Understand the student mental-health dataset.
2. Perform exploratory data analysis.
3. Identify invalid and unusual values.
4. Clean the dataset.
5. Engineer useful categorical features.
6. Build a robust preprocessing pipeline.
7. Train multiple regression models.
8. Tune the Random Forest model using `RandomizedSearchCV`.
9. Compare model performance.
10. Save the trained machine learning pipeline for future predictions.

---

# 📊 Dataset

The dataset contains **5,000 student records** and **13 columns**.

### Dataset Features

| Feature                   | Description                         | Type        |
| ------------------------- | ----------------------------------- | ----------- |
| `Age`                     | Student age                         | Numerical   |
| `Gender`                  | Student gender                      | Categorical |
| `Country`                 | Student's country                   | Categorical |
| `Academic_Level`          | Academic level                      | Categorical |
| `Most_Used_Platform`      | Most-used social media platform     | Categorical |
| `Purpose_Of_Use`          | Main purpose of social-media use    | Categorical |
| `Avg_Daily_Usage_Hours`   | Average daily usage of social media | Numerical   |
| `Daily_Unlocks`           | Number of daily device/app unlocks  | Numerical   |
| `Study_Hours`             | Daily study hours                   | Numerical   |
| `Physical_Activity_Hours` | Physical activity hours             | Numerical   |
| `Sleep_Hours_Per_Night`   | Average sleep per night             | Numerical   |
| `Stress_Level`            | Reported stress level               | Ordinal     |
| `Mental_Health_Score`     | Target mental-health score          | Numerical   |

### Target

```text
Mental_Health_Score
```

The target is continuous, so the project uses regression algorithms rather than classification algorithms.

---

# 🔍 Step 1 — Data Understanding

The initial dataset contains:

```text
Rows:     5,000
Columns:  13
```

The notebook checks:

* First and last records
* Random samples
* Dataset shape
* Column names
* Data types
* Missing values
* Duplicate records
* Descriptive statistics

### Missing Values

The initial dataset contains **no missing values**.

### Duplicate Records

The dataset contains:

```text
2 duplicate rows
```

The notebook performs further cleaning before model training.

---

# 📈 Step 2 — Exploratory Data Analysis

Several visual and statistical analyses were performed to understand the relationships between the features and the target.

## Target Distribution

The distribution of:

```text
Mental_Health_Score
```

was visualized using a histogram with KDE.

The target has:

```text
Mean ≈ 6.23
Minimum = 3.6
Maximum = 9.4
```

---

## Correlation Analysis

A correlation heatmap was used to examine relationships between numerical variables.

This provides an initial understanding of how numerical behavioral and lifestyle variables relate to the target.

---

## Stress Level vs Mental Health Score

A boxplot was used to compare:

```text
Stress_Level
```

with:

```text
Mental_Health_Score
```

The stress feature contains ordered categories:

```text
Low
Medium
High
Very High
```

Because these categories have a natural order, they are treated as an **ordinal feature** during preprocessing.

---

## Study Hours vs Mental Health Score

A scatter plot was used to examine the relationship between:

```text
Study_Hours
```

and:

```text
Mental_Health_Score
```

---

## Sleep Hours vs Mental Health Score

The relationship between:

```text
Sleep_Hours_Per_Night
```

and:

```text
Mental_Health_Score
```

was visualized using a scatter plot.

---

## Physical Activity vs Mental Health Score

A scatter plot was used to investigate:

```text
Physical_Activity_Hours
```

against:

```text
Mental_Health_Score
```

---

## Most Used Platform

The distribution of:

```text
Most_Used_Platform
```

was analyzed using a count plot.

---

# 🧹 Step 3 — Data Validation & Cleaning

The notebook performs validation before model training.

## Invalid Physical Activity Values

The dataset contained negative values in:

```text
Physical_Activity_Hours
```

Since physical activity hours cannot logically be negative, these records were removed.

The initial dataset contained:

```text
22 negative values
```

These invalid records were removed before modeling.

---

## Outlier Analysis

The project uses the **IQR method** to identify potential outliers in numerical variables.

The analysis showed potential outliers in some variables, including:

* `Study_Hours`
* `Physical_Activity_Hours`
* Other numerical features

However, the project does **not blindly remove every statistical outlier**.

This is important because an observation being statistically unusual does not automatically mean that it is invalid.

---

# 📐 Step 4 — Skewness Analysis

Numerical feature skewness was calculated.

The most noticeably skewed feature used for special preprocessing was:

```text
Study_Hours
```

Its skewness was approximately:

```text
0.436
```

Therefore, a logarithmic transformation was applied to this feature during preprocessing.

---

# ⚙️ Step 5 — Feature Engineering

## Country Grouping

The dataset contains multiple countries with different frequencies.

Instead of one-hot encoding every country individually, the notebook identifies the **top 10 countries** by frequency.

All remaining countries are grouped into:

```text
Other
```

A new feature is created:

```text
Grouped_Country
```

Example:

```text
USA       → USA
India     → India
Canada    → Canada
France    → France
Unknown/less frequent country → Other
```

This reduces the number of categorical levels and helps make categorical encoding more manageable.

---

# ✂️ Step 6 — Train-Test Split

The cleaned data is divided into training and testing sets using:

```text
Training data: 80%
Testing data: 20%
random_state = 42
```

The target variable is:

```text
Mental_Health_Score
```

---

# 🔄 Step 7 — Data Preprocessing

A major part of the project is the use of:

```text
Pipeline
+
ColumnTransformer
```

This keeps preprocessing and model training together and helps prevent inconsistent transformations between training and prediction.

---

## Feature Groups

### Skewed Numerical Features

```python
["Study_Hours"]
```

Processing:

```text
Log Transformation
       ↓
StandardScaler
```

---

### Normal Numerical Features

```python
[
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night"
]
```

Processing:

```text
StandardScaler
```

---

### Ordinal Feature

```python
["Stress_Level"]
```

The categories are explicitly ordered:

```text
Low
Medium
High
Very High
```

They are encoded using:

```python
OrdinalEncoder
```

with the specified category order.

---

### Nominal Categorical Features

```python
[
    "Gender",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Grouped_Country"
]
```

Processing:

```text
OneHotEncoder
```

with:

```python
handle_unknown="ignore"
drop="first"
```

---

# 🏗️ Preprocessing Architecture

The preprocessing pipeline can be summarized as:

```text
                    Input Features
                          │
             ┌────────────┴────────────┐
             │                         │
      Numerical Features        Categorical Features
             │                         │
     ┌───────┴────────┐        ┌───────┴────────┐
     │                │        │                │
 Study_Hours      Other       Stress        Nominal
     │           Numeric        │          Categories
 Log1p              │        Ordinal           │
     │           Scaling      Encoding      One-Hot
 Scaling
     │                │          │               │
     └────────────────┴──────────┴───────────────┘
                          │
                   Model Training
```

---

# 🤖 Step 8 — Machine Learning Models

The project evaluates three regression configurations.

## 1. Linear Regression

Linear Regression is used as the **baseline model**.

```python
LinearRegression()
```

The baseline establishes a reference point for evaluating more complex models.

### Performance

| Metric |      Score |
| ------ | ---------: |
| R²     | **0.7452** |
| MAE    | **0.5182** |
| MSE    | **0.4328** |
| RMSE   | **0.6579** |

---

# 🌲 2. Random Forest Regressor

A Random Forest Regressor is trained using the same preprocessing pipeline.

```python
RandomForestRegressor()
```

Random Forest can capture nonlinear relationships and interactions between features that a simple linear model may not capture.

### Performance

| Metric |      Score |
| ------ | ---------: |
| R²     | **0.8960** |
| MAE    | **0.3146** |
| MSE    | **0.1767** |
| RMSE   | **0.4203** |

---

# 🔧 3. Tuned Random Forest

Hyperparameter optimization is performed using:

```python
RandomizedSearchCV
```

### Search Configuration

```text
n_iter = 15
cv = 5
scoring = "r2"
random_state = 42
n_jobs = -1
```

### Parameters Tuned

The search covers:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
```

### Best Parameters Found

```python
{
    "Model__n_estimators": 100,
    "Model__min_samples_split": 2,
    "Model__min_samples_leaf": 2,
    "Model__max_features": 1.0,
    "Model__max_depth": None
}
```

### Tuned Model Performance

| Metric |      Score |
| ------ | ---------: |
| R²     | **0.8900** |
| MAE    | **0.3243** |
| MSE    | **0.1869** |

The tuned Random Forest configuration was evaluated on the held-out test set.

---

# 📊 Model Comparison

The notebook compares the models using R² and MAE.

| Model                 |         R² |        MAE |
| --------------------- | ---------: | ---------: |
| Linear Regression     |     0.7452 |     0.5182 |
| Random Forest         | **0.8960** | **0.3146** |
| Random Forest (Tuned) |     0.8900 |     0.3243 |

### Interpretation

The Random Forest model achieved an R² of approximately:

```text
0.896
```

on the test set, while its MAE was approximately:

```text
0.315
```

The tuned configuration did not improve the held-out test metrics compared with the original Random Forest configuration in this particular experiment.

This demonstrates an important machine-learning principle:

> Hyperparameter tuning does not guarantee better performance on every test set.

---

# 📏 Evaluation Metrics

## R² — Coefficient of Determination

R² measures how much of the variance in the target is explained by the model.

```text
Higher R² → better explanatory performance
```

For example:

```text
R² = 0.896
```

means the model explains approximately 89.6% of the variance in the test target under this evaluation setup.

---

## MAE — Mean Absolute Error

MAE measures the average absolute difference between:

```text
Actual Score
```

and:

```text
Predicted Score
```

The Random Forest achieved:

```text
MAE ≈ 0.315
```

So its predictions differed from the actual mental-health score by about 0.315 score units on average on the test set.

---

## MSE — Mean Squared Error

MSE squares prediction errors before averaging them.

Lower values indicate smaller prediction errors.

Random Forest:

```text
MSE ≈ 0.177
```

---

## RMSE — Root Mean Squared Error

RMSE is the square root of MSE and is expressed in the same units as the target.

Random Forest:

```text
RMSE ≈ 0.420
```

---

# 💾 Model Saving

The notebook uses `joblib` to serialize the trained model pipeline.

The saved file is:

```text
Predicting Student Mental Health Score.pkl
```

The notebook currently saves:

```python
pipeline_RF
```

which is the **baseline Random Forest pipeline**, not the `rf_best_pipeline` tuned estimator.

This distinction is important when reproducing or deploying the notebook.

---

# 🔮 Making Predictions

After loading the saved pipeline:

```python
import joblib

model = joblib.load(
    "Predicting Student Mental Health Score.pkl"
)
```

New student data can then be passed to the pipeline for prediction.

Because preprocessing is included inside the pipeline, the same transformations used during training are automatically applied during prediction.

---

# 🧠 Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

* Regression
* Exploratory Data Analysis
* Data validation
* Data cleaning
* Outlier detection
* Skewness analysis
* Log transformation
* Feature engineering
* Train-test split
* Feature scaling
* Ordinal encoding
* One-hot encoding
* `Pipeline`
* `ColumnTransformer`
* Linear Regression
* Random Forest Regression
* Hyperparameter tuning
* `RandomizedSearchCV`
* Cross-validation
* R²
* MAE
* MSE
* RMSE
* Model serialization with Joblib

---

# 🛠️ Technology Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Model Persistence

* Joblib

### Development Environment

* Google Colab / Jupyter Notebook

---

# 📁 Suggested Repository Structure

```text
Student-Mental-Health-Score-Prediction/
│
├── ML Project_ Predicting Student Mental Health Score.ipynb
├── Predicting Student Mental Health Score.pkl
├── Student Social Media And Mental Health Impact.csv
├── requirements.txt
└── README.md
```

A cleaner production-oriented naming convention could be:

```text
student-mental-health-score-prediction/
│
├── notebook/
│   └── student_mental_health_prediction.ipynb
│
├── models/
│   └── mental_health_score_model.pkl
│
├── data/
│   └── Student_Social_Media_And_Mental_Health_Impact.csv
│
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/student-mental-health-score-prediction.git
```

Navigate into the project:

```bash
cd student-mental-health-score-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
```

---

# ▶️ Running the Project

Open the notebook:

```text
ML Project_ Predicting Student Mental Health Score.ipynb
```

Then execute the workflow in order:

```text
Load Dataset
      ↓
Data Understanding
      ↓
EDA
      ↓
Data Cleaning
      ↓
Skewness Analysis
      ↓
Feature Engineering
      ↓
Train-Test Split
      ↓
Preprocessing Pipeline
      ↓
Linear Regression
      ↓
Random Forest
      ↓
RandomizedSearchCV
      ↓
Model Comparison
      ↓
Save Model
```

---

# 📌 Key Results

### Dataset

```text
5,000 records
13 original features
```

### Data Quality

```text
0 missing values
2 duplicate rows detected
22 negative Physical_Activity_Hours values removed
```

### Best Test Performance in the Notebook

```text
Model: Random Forest Regressor

R²   = 0.8960
MAE  = 0.3146
MSE  = 0.1767
RMSE = 0.4203
```

---

# ⚠️ Important Project Considerations

This project predicts a **mental-health score from a dataset**, but the resulting predictions should not be interpreted as a clinical diagnosis.

A machine-learning prediction is not equivalent to:

* Psychological assessment
* Clinical screening
* Medical diagnosis
* Professional mental-health evaluation

The model should therefore be treated as an educational machine-learning project rather than a medical diagnostic system.

Additionally, relationships discovered in the dataset should not automatically be interpreted as causal relationships.

---

# 🔮 Future Improvements

Possible improvements include:

### 1. Cross-Validated Model Comparison

Evaluate all models using repeated cross-validation rather than relying primarily on a single train-test split.

### 2. Additional Regression Models

Experiment with:

* Gradient Boosting Regressor
* HistGradientBoostingRegressor
* XGBoost Regressor
* Extra Trees Regressor

### 3. Feature Importance

Analyze which features contribute most strongly to predictions using:

* Random Forest feature importance
* Permutation importance
* SHAP

### 4. Better Hyperparameter Search

Increase the search space and number of iterations for RandomizedSearchCV or compare it with GridSearchCV where appropriate.

### 5. Prediction Interface

Deploy the trained pipeline through:

* Streamlit
* FastAPI
* Flask

### 6. Model Monitoring

For a production implementation, monitor:

* Prediction distributions
* Input data drift
* Model performance
* Data quality
* Out-of-range inputs

---

# 🏆 Project Outcome

This project demonstrates an end-to-end **machine learning regression workflow** for predicting student mental-health scores.

The workflow combines:

```text
EDA
+
Data Validation
+
Data Cleaning
+
Feature Engineering
+
ColumnTransformer
+
Pipeline
+
Regression
+
Random Forest
+
Hyperparameter Tuning
+
Model Evaluation
+
Model Serialization
```

The baseline Random Forest achieved an R² of approximately **0.896** and an MAE of approximately **0.315** on the held-out test set used in the notebook.

---

# 👨‍💻 Author

**Muhammad Ibrahim**

BS Software Engineering
National Textile University, Faisalabad

Interested in:

* Machine Learning
* Deep Learning
* Generative AI
* AI Engineering
* MLOps

---

# 📄 License

This project is intended for educational and portfolio purposes.

If you publish it publicly, add an appropriate license such as the MIT License if you want others to reuse and modify the code.
