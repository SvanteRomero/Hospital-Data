# Work Report: FY 2022-2025 Data Analysis and Predictive Modeling

**Author:** [Your Name]
**Date:** July 25, 2025

---

## 1. Executive Summary

This report details the data analysis and predictive modeling projects undertaken from FY 2022-2025. The work evolved from initial exploratory analysis to the development of a sophisticated **regression model that directly predicts the financial penalty percentage** for hospitals. By leveraging advanced feature engineering and powerful machine learning models like XGBoost, this project provides a clear, quantifiable forecast of revenue risk, enabling a highly targeted and proactive approach to performance improvement.

---

## 2. Key Findings & Actionable Insights

* **Direct Financial Risk Prediction:** The final regression model accurately predicts the `Payment reduction percentage` for FY 2025, moving beyond simple classification to provide a specific financial risk forecast for each hospital.
* **Top-Priority Hospitals Identified:** The model has generated a ranked list of the **top 20 hospitals with the highest predicted payment reductions**. These institutions represent the most critical targets for immediate intervention to mitigate financial losses.
* **Key Penalty Drivers:** Advanced feature engineering confirmed that a hospital's historical penalty status (`Overall_Penalty_Last_Year`) and the interaction between its patient population (`Dual proportion`) and readmission rates for key conditions are strong predictors of future penalties.
* **Model Performance:** The final XGBoost Regressor model demonstrated strong predictive power on the FY 2025 data, achieving an **R-squared (R2) score of 0.78**, indicating that it explains a significant portion of the variance in payment reductions.

---

## 3. Projects and Workstreams

### 3.1. Data Consolidation & Exploratory Analysis

* **Objective:** To create a unified dataset and perform initial analysis to understand penalty drivers.
* **Methodology:**
    * Data from multiple fiscal years was consolidated into a single CSV file.
    * Exploratory data analysis was conducted to identify high-penalty conditions (HF, COPD), underperforming peer groups, and hospitals with high Excess Readmission Ratios (ERR).
* **Outcome:** A foundational understanding of the key factors associated with penalties, which guided the feature engineering for the predictive models.

### 3.2. Predictive Modeling V1: Penalty Classification

* **Objective:** To build and tune advanced classification models to predict the likelihood of a penalty.
* **Methodology:**
    * **Models:** Employed `RandomForestClassifier`, `GradientBoostingClassifier`, `XGBoost`, and `LightGBM`.
    * **Techniques:** Utilized SMOTE to handle class imbalance and created key features like `ERR_vs_median`.
* **Outcome:** The models achieved high accuracy in classification (e.g., **98.66% for HF**), validating the strength of the engineered features. This success laid the groundwork for a more advanced regression approach.

### 3.3. Advanced Modeling V2: Direct Penalty Prediction (Regression)

* **Objective:** To build a regression model to directly predict the `Payment reduction percentage` and quantify the financial risk.
* **Methodology:**
    * **Advanced Feature Engineering:** Created more sophisticated features, including a hospital's penalty status from the previous year (`Overall_Penalty_Last_Year`) and interaction terms (e.g., `Dual_x_ERR_HF`).
    * **Model:** An **XGBoost Regressor** was chosen for its high performance. The model was trained on data from FY 2022-2024 to predict penalties for FY 2025.
    * **Robust Validation:** Used **Time-Series Cross-Validation** to ensure the model's validity, respecting the chronological nature of the data.
* **Outcome:**
    * A highly accurate regression model that can forecast the financial penalty percentage for each hospital.
    * A prioritized list of the top 20 hospitals with the highest predicted financial risk for FY 2025.

    *_(Here, you can embed the final, corrected bar chart from your notebook showing the "Top 20 At-Risk Hospitals by Predicted Payment Reduction")_*

---

## 4. Tools and Technologies

* **Language & Libraries:** Python, Pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn
* **Development Environment:** Jupyter Notebook

---

## 5. Recommendations and Next Steps

Based on the predictive model's output, the following actions are recommended:

* **Immediate Engagement with Top 20 Hospitals:** Prioritize outreach and support for the **top 20 hospitals with the highest predicted payment reductions for FY 2025**. These institutions represent the most significant and immediate financial risk.
* **Deploy the Regression Model:** Operationalize the XGBoost regression model to create a proactive risk-monitoring dashboard. This will allow for ongoing, data-driven prioritization of performance improvement efforts.
* **Focus on Historical Performance:** The `Overall_Penalty_Last_Year` feature was a strong predictor. This indicates that interventions should be focused on hospitals with a history of penalties to prevent recurring issues.