# 🏦 Geldium Delinquency Risk Prediction AI Consulting Project

> **Tata iQ x Geldium** | AI Transformation Consultant: Hadeel
> Forage Job Simulation — Tata iQ GenAI Powered Data Analytics

---

## Project Overview

Geldium, a digital lending and consumer credit provider, experienced an increase in credit card delinquency rates. This project applies **AI-powered data analysis** to:

- Identify customers at risk of missing payments
- Build a predictive modeling framework (Random Forest)
- Recommend targeted, fair intervention strategies
- Ensure Responsible AI principles throughout

---

## Project Structure

```
geldium-delinquency-prediction/
│
├── geldium_eda.py                  # Exploratory Data Analysis code
├── geldium_eda_dashboard.png       # EDA visualizations
├── reports/
│   ├── EDA_Summary_Report.docx     # Task 1 — EDA Report
│   ├── Model_Plan.docx             # Task 2 — Predictive Model Plan
│   ├── Business_Summary.docx       # Task 3 — Stakeholder Report
│   └── AI_Collections_Presentation.pptx  # Task 4 — Executive Deck
└── README.md
```

---

##Key Findings

| Finding | Detail |
|---------|--------|
| **Dataset** | 500 customers, 19 features, 16% delinquency rate |
| **Strongest Predictor** | Missed_Payments + 6-month payment history |
| **Counter-intuitive** | Delinquent customers had *higher* avg income ($113K vs $107K) |
| **Data Issues** | Employment_Status inconsistencies + 4 Credit_Utilization outliers |
| **Class Imbalance** | 84% non-delinquent vs 16% delinquent |

---

## Model Approach

- **Algorithm:** Random Forest Classifier
- **Why:** Balances accuracy + interpretability — critical for financial compliance
- **Top 5 Features:**
  1. Missed_Payments
  2. Month_1 to Month_6 (payment history)
  3. Debt_to_Income_Ratio
  4. Credit_Utilization
  5. Account_Tenure
- **Evaluation:** AUC-ROC > 0.75 | F1 Score > 0.70

---

## Responsible AI

- Disparate Impact Analysis across Age, Location, Employment segments
- Human review required for all high-risk decisions
- Plain-language explanations for customers
- GDPR & ECOA compliance considered

---

## 🛠️ Tools & Skills

`Python` `Pandas` `Matplotlib` `Seaborn` `GenAI Prompting`
`EDA` `Predictive Modeling` `Business Communication` `Responsible AI`

---


##  How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/geldium-delinquency-prediction.git

# 2. Install requirements
pip install pandas numpy matplotlib seaborn openpyxl

# 3. Add the dataset file to the folder
# (Delinquency_prediction_dataset.xlsx)

# 4. Run the analysis
python geldium_eda.py
```

---

*This project was completed as part of the Tata iQ GenAI Powered Data Analytics Job Simulation on Forage.*
