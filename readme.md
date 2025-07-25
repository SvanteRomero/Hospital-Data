# Work Report: FY 2022-2025 Data Analysis and Predictive Modeling

**Author:** Svante Romero
**Date:** July 25, 2025

---

## 1. Executive Summary

This report summarizes the data analysis and predictive modeling projects undertaken from FY 2022-2025. The work progressed from foundational data consolidation and exploratory analysis to the development of a sophisticated **regression model that directly predicts the financial penalty percentage for at-risk hospitals**. This final model moves beyond simple classification to provide a precise, actionable metric for quantifying and prioritizing financial risk, culminating in a targeted list of hospitals for immediate intervention in FY 2025.

---

## 2. Key Findings and Actionable Insights

* **Direct Penalty Prediction:** The final model (`04_improved_revenue_risk_model.ipynb`) successfully predicts the actual `Payment reduction percentage` for the upcoming fiscal year, allowing for precise financial risk assessment.
* **High-Priority Hospitals for FY 2025:** A targeted list of the **top 20 hospitals with the highest predicted payment reductions** for FY 2025 has been generated. This provides a clear, data-driven action plan for the business.
* **Key Penalty Drivers:** Exploratory analysis consistently identified **Heart Failure (HF) and COPD** as the conditions most frequently leading to penalties, underscoring their importance in quality improvement initiatives.
* **Historical Performance as a Predictor:** The models confirmed that a hospital's penalty status in the previous year (`Overall_Penalty_Last_Year`) is a powerful predictor of future performance.

---

## 3. Projects and Workstreams

### 3.1. Data Preprocessing and Consolidation (`01_setup_test.ipynb`)

* **Objective:** To create a unified and analysis-ready dataset from multiple fiscal year Excel files.
* **Outcome:** A consolidated CSV file (`FY_2022-2025.csv`) that served as the foundation for all subsequent work.

### 3.2. Exploratory Data Analysis and Insights (`02_analysis_data.ipynb`)

* **Objective:** To analyze the consolidated data to identify trends, correlations, and actionable insights.
* **Key Findings:** Identified the top penalized hospitals, underperforming peer groups, and the conditions driving the highest number of penalties.

### 3.3. Penalty Prediction Models (V2 & V3)

* **Objective:** To build and tune a series of advanced **classification models** to predict whether a hospital would be penalized.
* **Methodology:**
    * **Models:** Used `RandomForestClassifier`, `GradientBoostingClassifier`, **XGBoost**, and **LightGBM**.
    * **Techniques:** Employed **SMOTE** to handle class imbalance and `GridSearchCV` for hyperparameter tuning.
* **Outcome:** These models achieved high accuracy (up to **98.66%** for the HF condition) in classifying hospitals as "at-risk" or "not at-risk."

### 3.4. Revenue at Risk (RaR) Simulation (`03_revenue_risk.ipynb`)

* **Objective:** To quantify the *potential* financial opportunity by simulating the impact of performance improvements.
* **Methodology:**
    * Calculated the potential revenue change if a hospital's ERR improved to its peer group's median.
    * A **Monte Carlo simulation** was used to model uncertainty and produce a 95% confidence interval for the RaR.
* **Outcome:** This analysis provided a simulated financial risk, which was a valuable step toward quantifying the problem.

### 3.5. Predictive Revenue Risk Modeling (`04_improved_revenue_risk_model.ipynb`)

* **Objective:** To improve upon the RaR simulation by building a **regression model** to directly predict the `Payment reduction percentage`.
* **Methodology:**
    * **Advanced Feature Engineering:** Created more powerful features, including `Overall_Penalty_Last_Year` and interaction terms (e.g., `Dual_x_ERR_HF`).
    * **Regression Models:** An **XGBoost Regressor** was trained and tuned using `TimeSeriesSplit` cross-validation to respect the chronological nature of the data.
    * **Chronological Split:** The model was trained on data from FY 2022-2024 to make predictions for FY 2025.
* **Outcome:**
    * A highly accurate regression model that provides a specific, predicted penalty percentage for each hospital.
    * A definitive, prioritized list of the **top 20 at-risk hospitals for FY 2025** based on their predicted financial penalty.

    *_(Here, you can embed the bar chart from your notebook showing the "Top 20 At-Risk Hospitals by Predicted Payment Reduction")_*

---

## 4. Tools and Technologies Used

* **Programming Language:** Python
* **Core Libraries:** Pandas, NumPy, scikit-learn
* **Advanced Modeling:** XGBoost, LightGBM, imblearn (for SMOTE)
* **Visualization:** Matplotlib, Seaborn
* **Development Environment:** Jupyter Notebook

---

## 5. Conclusion and Recommendations

This work successfully evolved from data exploration to a powerful predictive tool. The final regression model provides a precise, forward-looking measure of financial risk, enabling a more strategic and data-driven approach to hospital performance management.

**Recommendations:**

1.  **Immediate Action on Top 20 List:** Prioritize engagement with the **top 20 hospitals identified by the XGBoost regression model**. These hospitals have the highest predicted financial penalties for FY 2025 and represent the most critical targets for intervention.
2.  **Deploy the Regression Model:** Operationalize the predictive model into a dashboard to provide ongoing, real-time risk scores for all hospitals.
3.  **Focus QI Efforts:** Continue to allocate quality improvement resources to address **Heart Failure and COPD**, as they remain the primary drivers of penalties.