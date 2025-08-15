# Predicting Hospital Readmissions for Diabetic Patients Using Logistic Regression

A compact, reproducible ML project that predicts readmission risk for diabetic patients using an interpretable logistic regression pipeline. The work is based on the public diabetes readmission dataset analyzed by Strack et al. (2014).

> **TL;DR:** Final model reaches **61.46% accuracy** and **0.65 ROC–AUC** on a 70/30 train/test split. It’s moderately predictive and easy to explain; recall is limited, so if you care about catching more at-risk patients, you’ll want threshold tuning or a different model class.

---

## Abstract

Hospital readmissions pose a significant challenge in healthcare, leading to adverse outcomes for diabetic patients and increased resource demands on hospitals. This project uses 101,763 patient encounters from Strack et al. (2014) to develop logistic regression models that predict readmissions among diabetic patients. The workflow covers data preprocessing, exploratory data analysis (EDA), iterative model development, and diagnostics. The final model achieves **61.46% accuracy** and **0.65 AUC**. Key predictors—especially prior inpatient visits and diabetes medication use—help identify at-risk patients so clinicians can target follow-up interventions and reduce readmissions.

**Index terms:** hospital readmission, logistic regression, diabetes, predictive modeling, healthcare analytics

---

## Dataset

* **Source:** Strack et al., “Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records,” *BioMed Research International*, 2014.
* **Rows/Columns:** \~101,763 encounters, 47 features (demographics, utilization, labs/meds, etc.).
* **Target:** `readmitted` with values `No`, `Within30Days`, `After30Days`. We binarize to `readmitted_binary` where `1` = readmitted (within or after 30 days), `0` = not readmitted.


---

## Methodology (Short)

1. **Preprocessing**
   Replace `"?"` with `NaN`; impute categorical with mode/“Unknown”, numeric with median. Drop `weight` (≈97% missing). Create `log_number_inpatient = log1p(number_inpatient)` to reduce skew. Encode categoricals (current notebook uses label encoding; see “Limitations”).

2. **EDA**
   Distribution checks for stay length, meds, and utilization. Readmission rates by age, sex, and `diabetesMed`. Correlation heatmap and boxplots linking utilization to readmissions.

3. **Modeling**
   **Initial predictors:** `time_in_hospital`, `num_lab_procedures`, `num_medications`, `number_inpatient`.
   **Refined predictors:** initial + `age`, `gender`, `number_outpatient`, `diabetesMed`, `log_number_inpatient`.
   70/30 split (`random_state=42`). Report accuracy, ROC–AUC, precision, recall, F1. Use VIF to inspect multicollinearity.

4. **Diagnostics**
   Confusion matrices, ROC curves, and VIF summaries. Discuss class imbalance and the cost of false negatives.

---

## Results

**Initial model (4 features):**

* Accuracy **59.94%**, ROC–AUC **0.63**
* Precision **59.94%**, Recall **40.98%**, F1 **48.68%**
* Confusion counts: TN **12,500**, TP **5,800**, FP **3,876**, FN **8,353**

**Refined model (9 features):**

* Accuracy **61.46%**, ROC–AUC **0.65**
* Precision **62.48%**, Recall **42.22%**, F1 **50.39%**
* Confusion counts: TN **12,788**, TP **5,975**, FP **3,588**, FN **8,178**

**Most influential predictors (logistic coefficients):**

* `log_number_inpatient` (≈ **0.7773**) → odds ratio ≈ **2.18**
* `diabetesMed` (≈ **0.2498**) → odds ratio ≈ **1.28**
* `number_outpatient` (≈ **0.1006**)

**Reality check:** Numbers are decent for a simple, interpretable baseline but below literature that often reports AUC 0.70–0.85 with tree ensembles or richer features. High false negatives limit clinical utility unless you adjust thresholds or costs.

---

## How to Reproduce

**Requirements:** Python 3.10+, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `statsmodels`, `jupyter`.

```bash
# 1) Create env (example with venv) and install deps
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pandas numpy scikit-learn matplotlib seaborn statsmodels jupyter

# 2) Put data in the expected path
mkdir -p data

# 3) Run the notebook
jupyter lab  # or: jupyter notebook
```

---

## Limitations & Ways to Improve

* **Encoding & leakage:** Current notebook uses `LabelEncoder` on full data, which can leak information and impose fake ordinality. Switch to a `Pipeline` + `ColumnTransformer` with `OneHotEncoder(drop='first', handle_unknown='ignore')` and fit only on the training split.
* **Class imbalance:** Recall is weak. Either tune the decision threshold for higher recall, use `class_weight='balanced'`, or adopt cost-sensitive learning.
* **Multicollinearity:** `number_inpatient` and `log_number_inpatient` are related (VIF ≈ 10–11). Consider keeping only the log term or use ridge/logistic with regularization.
* **Model class:** If you need raw performance, try gradient boosting (XGBoost/LightGBM) or calibrated random forests. Expect AUC improvements at the cost of interpretability.
* **Features:** Add social determinants (if available), discharge disposition, prior diagnoses detail, and meds trajectories. Temporal/sequence models can also help.

---

## References

1. B. Strack et al., “Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records,” *BioMed Research International*, vol. 2014, 2014, Art. no. 781670.
2. T. Hastie, R. Tibshirani, and J. Friedman. *The Elements of Statistical Learning*, 2nd ed., Springer, 2009.
