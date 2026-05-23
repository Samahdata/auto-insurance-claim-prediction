# 🚗 Auto-Insurance Claim Prediction

A machine learning project that predicts whether an auto-insurance policyholder will file a claim, using behavioral, demographic, and policy-level features.

---

## 📌 Problem Statement

Insurance companies need to identify high-risk policyholders **before** claims are filed — enabling proactive risk management, better pricing, and targeted outreach. This project builds and evaluates a binary classification model to predict claim likelihood.

---

## 📂 Dataset

- **Size:** ~180,000 policies
- **Target:** `filed_claim` (binary: 1 = claim filed, 0 = no claim)
- **Class imbalance:** ~28% positive (claim filed)
- **Features include:** driver demographics, vehicle info, driving behavior (telematics), policy details, credit band, NPS score

---

## 🔧 Project Pipeline

### 1. Exploratory Data Analysis
- Identified columns with suspicious negative values (`premium_monthly`, `nps_score`)
- Detected a column with 92% missing rate → dropped
- Found and removed duplicate customer records
- Identified a **data leakage column** (`claim_propensity_internal`) — an internal score computed from the target, removed before modeling

### 2. Data Cleaning
- Fixed negative monetary values using `.abs()`
- Clamped `nps_score` to valid range [0, 10]
- Parsed mixed-format dates (`yyyy-mm-dd` and `dd/mm/yyyy`) using `format='mixed'`
- Stripped whitespace and normalized casing across all categorical columns
- Imputed missing values with median (numeric) and mode (categorical)

### 3. Feature Engineering
| Feature | Description |
|---------|-------------|
| `claims_per_year_licensed` | Prior claims normalized by years of driving experience |
| `log_vehicle_value` | Log-transform to reduce skewness in vehicle value |
| `miles_per_violation` | Mileage per violation — lower = riskier driver |
| `risk_score` | Composite behavioral score (hard braking + late night trips + avg speed) |
| `policy_duration_days` | Days between policy start and renewal date |

### 4. Modeling

Three models were trained and compared on the same stratified 80/20 split:

| Model | Accuracy | PR-AUC | ROC-AUC |
|-------|----------|--------|---------|
| **XGBoost** | 0.6178 | **0.4666** | **0.6808** |
| Logistic Regression | 0.7335 | 0.4616 | 0.6778 |
| Random Forest | 0.7258 | 0.4262 | 0.6588 |

> **PR-AUC** was chosen as the primary metric due to class imbalance — accuracy alone is misleading when only 28% of records are positive.

### 5. Hyperparameter Tuning
- Tuned XGBoost `learning_rate` via 5-fold cross-validation on training set only
- Best value: `learning_rate = 0.1` → CV PR-AUC = 0.48

### 6. Explainability
Top 3 features driving claim risk:
1. **`years_licensed`** — less experienced drivers file more claims
2. **`claims_per_year_licensed`** — high claim rate relative to experience signals risk
3. **`n_accidents_5yr`** — past accidents strongly predict future claims

### 7. Temporal Evaluation
Re-split data chronologically (oldest 80% → train, newest 20% → test) to simulate real deployment:
- Random split PR-AUC: **0.4795**
- Temporal split PR-AUC: **0.4661**
- Small gap (0.013) indicates the model generalizes reasonably well across time, with minor drift expected as driving patterns evolve

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)
![Pandas](https://img.shields.io/badge/Pandas-latest-lightgrey)

- **Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, XGBoost, matplotlib
- **Modeling:** sklearn Pipeline with ColumnTransformer

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/auto-insurance-claim-prediction.git
cd auto-insurance-claim-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook notebook.ipynb
```

---

## 📁 Repository Structure

```
auto-insurance-claim-prediction/
│
├── notebook.ipynb          # Main analysis and modeling notebook
├── insurance_claims.csv    # Dataset
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## 📊 Key Insights

- Urban drivers have **~32% claim rate** vs **~23%** for rural drivers
- Drivers with **Fair/Poor credit** file claims ~6% more than Excellent credit holders
- **Business vehicles** show the highest claim rate (~33%) due to higher usage frequency
- Model shows slight temporal drift → periodic retraining recommended in production
