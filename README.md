## 🏦 Loan Status Prediction – EDA & Machine Learning Modeling

This project focuses on analyzing a large-scale loan dataset and building a robust machine learning model to predict loan approval status. With **45,000 records and 14 variables**, this project highlights the full data science pipeline: data cleaning, visualization, modeling, and optimization.

---

### 🎯 Main Objectives

* Understand the relationship between applicant attributes and loan status
* Perform **data cleaning**, **outlier removal**, and **encoding**
* Handle **imbalanced data** via under-sampling
* Build and fine-tune an **XGBoost model** for loan status prediction
* Evaluate the model using appropriate metrics

---

### 🗂️ Dataset Description

The dataset includes the following 14 variables:

* `person_age`: Age of the applicant
* `person_gender`: Gender
* `person_education`: Education level
* `person_income`: Annual income
* `person_emp_exp`: Employment experience (in years)
* `person_home_ownership`: Home ownership status (Rent/Own/Mortgage)
* `loan_amnt`: Loan amount requested
* `loan_intent`: Loan purpose (e.g., education, medical, personal)
* `loan_int_rate`: Interest rate of the loan
* `loan_percent_income`: Loan amount as a % of income
* `cb_person_cred_hist_length`: Length of credit history (years)
* `credit_score`: Applicant’s credit score
* `previous_loan_defaults_on_file`: Indicator for any past loan defaults
* `loan_status`: **Target variable** (1 = Approved, 0 = Rejected)

---

### 🧹 Data Preprocessing & Cleaning

* **Outlier Removal**:

  * `person_age` values above **80** and `person_emp_exp` above **60** were removed as unrealistic.
  * These outliers were few in number and had negligible impact on overall dataset size.

* **Categorical Encoding**:

  * All categorical variables were encoded for machine learning models.
  * Variables like `person_gender`, `person_home_ownership`, and `loan_intent` were one-hot or label encoded as appropriate.

* **Class Imbalance Handling**:

  * `loan_status` was **imbalanced** (more approvals than rejections).
  * Applied **under-sampling** to balance the dataset and improve model generalization for the minority class.

---

### 📊 Exploratory Data Analysis (EDA)

* **Visualizations**:

  * Count plots and pie charts for all categorical variables (with respect to loan status)
  * Box plots and histograms for numerical variables to observe spread and detect outliers
  * Correlation heatmap for numerical features

* **Findings**:

  * High loan amounts relative to income and low credit scores often result in loan rejection
  * Applicants with longer credit history and higher education levels generally get approved
  * `loan_intent` (e.g., for medical or personal reasons) is a strong predictor of default risk

---

### 🧠 Machine Learning Modeling

Model used:

* **XGBoost**

**Model Evaluation** based on:

* Accuracy
* Precision, Recall, F1-Score
* Confusion Matrix
* **ROC-AUC Curve**

---

### 🔧 Fine-Tuning with XGBoost

* Used **GridSearchCV** for hyperparameter tuning
* Tuned parameters:

  * `n_estimators`, `max_depth`, `learning_rate`, `subsample`
* Achieved better **recall** for rejected cases and improved overall F1-score
* Final model generalized well on the validation set

---

### 📌 Conclusion

* Successfully implemented a complete machine learning pipeline to predict loan approval status
* Outliers and data imbalance were effectively handled
* XGBoost was selected and fine-tuned as the best-performing model based on ROC-AUC and other metrics
* The final model demonstrated strong generalization and is suitable for deployment or further analysis
