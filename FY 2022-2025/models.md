Of course. Here is a comprehensive markdown document detailing both predictive models.

-----

# Predictive Modeling for Hospital Penalties

This document outlines two predictive modeling approaches to analyze hospital penalty indicators based on the provided dataset. The goal is to first predict penalties for specific medical conditions and then to forecast which hospitals are likely to receive a penalty in the upcoming year.

-----

## Model 1: Predicting Penalties for Specific Conditions

This model focuses on predicting whether a hospital will receive a penalty for individual conditions such as Heart Failure (HF), Pneumonia, and others. The objective is to achieve a high level of accuracy (over 95%) for each condition-specific prediction.

### Feature Engineering: The "Why"

The selection of features is critical for building an accurate model. The logic for choosing the features was to provide the model with a comprehensive view of a hospital's performance, peer context, and financial standing.

  * **Core Performance Metrics (`ERR for condition`, `Peer group median ERR for condition`):** The Excess Readmission Ratio (ERR) is a direct measure of a hospital's performance. Comparing it to the peer group median tells the model whether the hospital is performing better or worse than similar institutions. The difference between these two is a powerful predictor.
  * **Engineered Feature (`ERR_vs_median`):** We created this feature by subtracting the peer median ERR from the hospital's ERR. This explicitly tells the model how far above or below the average a hospital is performing, which is often a more powerful signal than the raw numbers alone.
  * **Financial and Contextual Factors (`Payment adjustment factor`, `Payment reduction percentage`, `Dual proportion`):** These features provide context. The `Dual proportion` (percentage of patients eligible for both Medicare and Medicaid) can indicate the socioeconomic status of the patient population, which often correlates with readmission rates. The payment-related features give the model insight into the hospital's overall financial health under the program.
  * **Categorical Information (`Peer group assignment`):** This tells the model which group of hospitals a given institution is being compared against. Different peer groups have different performance expectations.

### The Code: Predicting by Condition

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def predict_penalty_for_condition(df, condition_name):
    """
    Trains a model to predict the penalty indicator for a specific condition.
    """
    print(f"--- Predicting Penalty Indicator for: {condition_name} ---")

    # Define features and target based on the specific condition
    features = [
        'Payment adjustment factor', 'Payment reduction percentage', 'Dual proportion',
        'Peer group assignment', 'Neutrality modifier',
        f'ERR for {condition_name}', f'Peer group median ERR for {condition_name}',
        f'DRG payment ratio for {condition_name}', 'Year'
    ]
    target = f'Penalty indicator for {condition_name}'

    # Prepare data: drop missing values and encode the target
    condition_df = df[features + [target]].dropna()
    condition_df[target] = LabelEncoder().fit_transform(condition_df[target])

    # Feature Engineering: Create the 'ERR_vs_median' feature
    condition_df['ERR_vs_median'] = condition_df[f'ERR for {condition_name}'] - condition_df[f'Peer group median ERR for {condition_name}']
    
    final_features = features + ['ERR_vs_median']

    # Split, train, and evaluate the model
    X = condition_df[final_features]
    y = condition_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy for {condition_name}: {accuracy:.4f}\n")
    return accuracy

# --- Execution ---
# df = pd.read_csv('FY_2022-2025.csv') # Load your data
# df.columns = df.columns.str.strip()
# conditions = ['HF', 'pneumonia', 'CABG', 'THA/TKA']
# for condition in conditions:
#     predict_penalty_for_condition(df, condition)
```

-----

## Model 2: Forecasting Next Year's Penalties

This model addresses a more forward-looking question: "Which hospitals are likely to be penalized next year?" It does this by training on historical data to predict future outcomes.

### Feature Engineering: The "Why"

To predict a future event, the most powerful features are often those that describe past behavior and recent performance.

  * **Aggregated Target (`Overall_Penalty`):** The first step was to simplify the target. Instead of predicting penalties for each condition separately, we created a single binary variable: `Overall_Penalty`. A hospital gets a `1` if it was penalized for *any* condition in a year, and `0` otherwise. This transforms the problem into a clear, single-target prediction task.
  * **The Lag Feature (`Overall_Penalty_Previous_Year`):** This is the cornerstone of the model. We created a feature that, for each hospital and each year, shows whether that hospital was penalized in the *previous* year. This feature alone is a very strong predictor because penalty status often has high inertia (penalized hospitals tend to stay penalized).
  * **Performance Metrics as Features (`ERR for...`, `Dual proportion`):** While the previous year's penalty is a strong feature, it isn't enough. By adding the **Excess Readmission Ratios (ERRs)** for all conditions, we give the model the *reasoning* behind a potential penalty. A hospital with consistently high ERRs is a prime candidate for a penalty, even if it wasn't penalized last year. The `Dual proportion` adds crucial context about the patient population.

This combination allows the model to learn not just from past outcomes, but also from the underlying performance metrics that *drive* those outcomes.

### The Code: Forecasting At-Risk Hospitals

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Data Preparation and Feature Engineering ---
# df = pd.read_csv('FY_2022-2025.csv') # Load data
# df.columns = df.columns.str.strip()

# 1. Create the 'Overall_Penalty' target
penalty_cols = [col for col in df.columns if 'Penalty indicator for' in col]
df['Overall_Penalty'] = df[penalty_cols].apply(lambda row: 1 if 'Y' in row.values else 0, axis=1)

# 2. Create the 'Overall_Penalty_Previous_Year' lag feature
df.sort_values(by=['Hospital CCN', 'Year'], inplace=True)
df['Overall_Penalty_Previous_Year'] = df.groupby('Hospital CCN')['Overall_Penalty'].shift(1)

# 3. Define the full feature set
feature_cols = [
    'Dual proportion', 'Overall_Penalty_Previous_Year',
    'ERR for AMI', 'ERR for COPD', 'ERR for HF',
    'ERR for pneumonia', 'ERR for CABG', 'ERR for THA/TKA'
]

# Clean and prepare data for modeling
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)

df_model = df.dropna(subset=['Overall_Penalty_Previous_Year'])

# --- Chronological Train-Test Split ---
train_df = df_model[df_model['Year'] < 2025]
test_df = df_model[df_model['Year'] == 2025]

X_train = train_df[feature_cols]
y_train = train_df['Overall_Penalty']
X_test = test_df[feature_cols]
y_test = test_df['Overall_Penalty']

# --- Train and Evaluate the Model ---
# Use class_weight='balanced' to handle the imbalance between penalized/not penalized hospitals
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Predict and evaluate on the 2025 data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Enhanced Model Accuracy on Predicting 2025 Penalties: {accuracy:.4f}")
if accuracy >= 0.95:
    print("This model meets the 95% accuracy requirement. ✅")

# --- Generate List of At-Risk Hospitals ---
results_df = test_df.copy()
results_df['Predicted_Penalty'] = y_pred
at_risk_hospitals = results_df[results_df['Predicted_Penalty'] == 1]

print(f"\nIdentified {len(at_risk_hospitals)} hospitals likely to be penalized in 2025.")
print(at_risk_hospitals[['Hospital CCN', 'Year']].head())

```