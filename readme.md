Of course! Based on the content of your Jupyter Notebooks, I've created a comprehensive markdown report that documents your work from FY 2022-2025.

Here is the markdown report, which you can copy and paste into a `.md` file.

---

# Work Report: FY 2022-2025 Data Analysis and Predictive Modeling

**Author:** [Your Name]
**Date:** July 25, 2025

---

## 1. Executive Summary

This report summarizes the data analysis and predictive modeling projects undertaken during the fiscal years 2022-2025. The work involved consolidating and cleaning raw data, performing in-depth exploratory data analysis to uncover key insights, and building a series of increasingly sophisticated models to predict hospital penalties. A key outcome of this work is the identification of at-risk hospitals and the quantification of "Revenue at Risk," providing actionable intelligence to guide strategic interventions.

---

## 2. Project Goals and Objectives

* **Data Consolidation:** To create a unified and clean dataset from multiple fiscal year files for comprehensive analysis.
* **Insight Generation:** To explore the consolidated data to identify key drivers of penalties, underperforming segments, and high-risk conditions.
* **Predictive Modeling:** To develop and refine a series of machine learning models to accurately predict penalty status for various conditions.
* **Risk Quantification:** To estimate the financial impact of performance by calculating the "Revenue at Risk" for hospitals.

---

## 3. Projects and Workstreams

This section details the individual projects, drawing from the analyses and models developed in your notebooks.

### 3.1. Data Preprocessing and Consolidation (`01_setup_test.ipynb`)

* **Objective:** The initial phase of the project focused on aggregating data from separate fiscal year Excel files into a single, analysis-ready CSV file.
* **Methodology:**
    * Loaded Excel files for FY 2023, 2024, and 2025.
    * Appended a `Year` column to each file to maintain temporal context.
    * Concatenated the individual dataframes into a unified dataset.
    * Cleaned the final dataframe by removing any unnamed or empty columns.
* **Outcome:** A consolidated CSV file (`FY_2022-2025.csv`) that served as the foundation for all subsequent analysis and modeling.

### 3.2. Exploratory Data Analysis and Insights (`02_analysis_data.ipynb`)

* **Objective:** To analyze the consolidated dataset to identify trends, correlations, and actionable insights related to hospital penalties.
* **Key Analyses and Findings:**
    * **Penalty Drivers:** Identified that Heart Failure (HF) and COPD are the conditions most frequently leading to penalties.
    * **Top Penalized Hospitals:** Isolated and listed the top 10 hospitals with the highest payment reduction percentages, identifying them as prime candidates for intervention.
    * **Peer Group Performance:** Analyzed performance across different peer groups, revealing that groups 2 and 4 have the highest penalty rates.
    * **High ERR Hospitals:** Pinpointed the top 10 hospitals with the worst Excess Readmission Ratios (ERR) for specific conditions, with THA/TKA (Total Hip/Knee Arthroplasty) appearing most frequently.
    * **Actionable Insights:** The analysis concluded with a summary of high-priority hospitals (those with payment reductions >= 1%), underperforming peer groups, and the top conditions driving penalties.

### 3.3. Penalty Prediction Model V2 (`01_penalty_v2.ipynb`)

* **Objective:** To build and tune advanced classification models to predict penalty status for a given condition.
* **Methodology:**
    * **Models Used:** `RandomForestClassifier` and `GradientBoostingClassifier`.
    * **Feature Engineering:** Created a key feature, `ERR_vs_median`, which measures the difference between a hospital's ERR and its peer group median.
    * **Handling Class Imbalance:** Utilized the **SMOTE (Synthetic Minority Over-sampling Technique)** to create a balanced training dataset, improving model performance on the minority class.
    * **Hyperparameter Tuning:** Employed `GridSearchCV` to find the optimal parameters for each model, tuning for the best ROC AUC score.
* **Outcome:** A robust pipeline for predicting penalties for any given condition, complete with model evaluation and confusion matrices for performance assessment. The final tuned Random Forest model achieved an accuracy of **90%** for the AMI condition.

### 3.4. At-Risk Hospital Prediction for FY 2025 (`02_penalized_v1.ipynb`)

* **Objective:** To build a predictive model to identify hospitals that are likely to be penalized in the upcoming fiscal year (2025).
* **Methodology:**
    * **Chronological Split:** The data was split chronologically, with the model trained on data from FY 2022-2024 and tested on FY 2025 data.
    * **Feature Engineering:** A critical feature, `Overall_Penalty_Previous_Year`, was created to capture a hospital's historical performance.
    * **Model:** A `RandomForestClassifier` was used, with `class_weight='balanced'` to handle the imbalanced nature of the target variable.
* **Outcome:** The model achieved an accuracy of **92.38%** in predicting which hospitals would be penalized in FY 2025, successfully identifying 2,451 at-risk hospitals.

### 3.5. Enhanced Penalty Prediction Model V3 (`01_penalty_v3.ipynb`)

* **Objective:** To further enhance the penalty prediction models by incorporating more powerful algorithms and robust validation techniques.
* **Methodology:**
    * **Advanced Models:** Introduced **XGBoost** and **LightGBM** classifiers to the pipeline.
    * **Feature Engineering:** Added a new feature, `ERR_ratio`, which is the ratio of a hospital's ERR to its peer group median.
    * **Robust Validation:** Implemented a `Pipeline` that includes `StandardScaler` and used `StratifiedKFold` cross-validation for a more reliable assessment of model performance.
* **Outcome:** This enhanced pipeline provided even higher accuracy. For the HF condition, the RandomForest model achieved an accuracy of **98.66%**, and the Gradient Boosting model achieved **98.58%**.

### 3.6. Revenue at Risk Analysis (`03_revenue_risk.ipynb`)

* **Objective:** To quantify the financial opportunity or risk associated with a hospital's readmission performance.
* **Methodology:**
    * **Revenue at Risk (RaR) Calculation:** A function was created to calculate the potential revenue change if a hospital's ERR for a given condition improved to its peer group's median.
    * **Monte Carlo Simulation:** A simulation with 1,000 iterations was run to model the uncertainty in peer group medians (assuming a ±5% normal noise). This produced a distribution of potential RaR, from which a mean and 95% confidence interval were derived.
* **Outcome:** A list of the top 20 hospitals with the highest "Revenue at Risk," providing a clear, data-driven priority list for performance improvement initiatives.

---

## 4. Tools and Technologies Used

* **Programming Language:** Python
* **Core Libraries:** Pandas, NumPy, scikit-learn
* **Advanced Modeling:** XGBoost, LightGBM, imblearn (for SMOTE)
* **Visualization:** Matplotlib, Seaborn
* **Development Environment:** Jupyter Notebook

---

## 5. Conclusion

This body of work successfully transitioned from raw, multi-source data to a clean, consolidated dataset, and ultimately to a suite of high-accuracy predictive models. The exploratory analysis provided critical insights into penalty drivers, while the "Revenue at Risk" analysis offers a clear financial imperative for targeted interventions. The final models are well-validated and ready to be deployed to proactively identify and support at-risk hospitals.