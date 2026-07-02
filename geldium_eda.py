# ============================================================
# Geldium Delinquency Risk - Exploratory Data Analysis (EDA)
# Author: Hadeel
# Role: AI Transformation Consultant, Tata iQ
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ─────────────────────────────────────────────
df = pd.read_excel('Delinquency_prediction_dataset.xlsx')

print("=" * 55)
print("        GELDIUM - EDA REPORT")
print("=" * 55)

# ── 2. DATASET OVERVIEW ──────────────────────────────────────
print("\n📊 DATASET OVERVIEW")
print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")
print(f"\n  Column Names:\n  {list(df.columns)}")
print(f"\n  Data Types:\n{df.dtypes}")

# ── 3. MISSING VALUES ─────────────────────────────────────────
print("\n⚠️  MISSING VALUES")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).query('`Missing Count` > 0')
print(missing_df)

# Imputation
df['Income']       = df['Income'].fillna(df['Income'].median())
df['Loan_Balance'] = df['Loan_Balance'].fillna(df['Loan_Balance'].median())
df['Credit_Score'] = df['Credit_Score'].fillna(df['Credit_Score'].mean())
print("\n  ✅ Missing values imputed successfully.")

# ── 4. DATA QUALITY ISSUES ────────────────────────────────────
print("\n🔍 DATA QUALITY ISSUES")

# Standardize Employment_Status
df['Employment_Status'] = df['Employment_Status'].str.strip().str.title()
print(f"  Employment_Status values after cleaning:\n  {df['Employment_Status'].value_counts().to_dict()}")

# Credit Utilization outliers
outliers = (df['Credit_Utilization'] > 1).sum()
print(f"\n  Credit_Utilization > 100%: {outliers} records (flagged for review)")

# ── 5. TARGET VARIABLE ────────────────────────────────────────
print("\n🎯 TARGET VARIABLE: Delinquent_Account")
counts = df['Delinquent_Account'].value_counts()
print(f"  Non-Delinquent (0): {counts[0]} ({counts[0]/len(df)*100:.1f}%)")
print(f"  Delinquent     (1): {counts[1]} ({counts[1]/len(df)*100:.1f}%)")
print(f"  ⚠️  Class Imbalance Detected: 84% vs 16%")

# ── 6. KEY RISK INDICATORS ────────────────────────────────────
print("\n🔴 KEY RISK INDICATORS")

features = ['Income', 'Credit_Score', 'Credit_Utilization',
            'Missed_Payments', 'Debt_to_Income_Ratio']

comparison = df.groupby('Delinquent_Account')[features].mean().round(2)
comparison.index = ['Non-Delinquent', 'Delinquent']
print(comparison.T.to_string())

print("\n  💡 Counter-intuitive Finding:")
print("     Delinquent customers have HIGHER avg income ($113,902)")
print("     than non-delinquent ($107,306).")
print("     → Income alone is NOT a reliable predictor.")
print("     → Behavioral features (Missed_Payments, payment history)")
print("       carry stronger predictive weight.")

# ── 7. MISSED PAYMENTS ANALYSIS ───────────────────────────────
print("\n📅 MISSED PAYMENTS DISTRIBUTION")
print(df['Missed_Payments'].describe().round(2).to_string())

# ── 8. PAYMENT HISTORY (6 MONTHS) ────────────────────────────
print("\n📆 6-MONTH PAYMENT HISTORY SUMMARY (Month_1)")
print(df['Month_1'].value_counts().to_string())

# ── 9. VISUALIZATIONS ─────────────────────────────────────────
print("\n📈 Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Geldium – EDA Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Delinquency distribution
colors = ['#1E2761', '#028090']
axes[0, 0].bar(['Non-Delinquent', 'Delinquent'],
               [counts[0], counts[1]], color=colors)
axes[0, 0].set_title('Delinquency Distribution')
axes[0, 0].set_ylabel('Count')
for i, v in enumerate([counts[0], counts[1]]):
    axes[0, 0].text(i, v + 5, f'{v}\n({v/len(df)*100:.0f}%)',
                    ha='center', fontweight='bold')

# Plot 2: Missed Payments by Delinquency
df.groupby('Delinquent_Account')['Missed_Payments'].mean().plot(
    kind='bar', ax=axes[0, 1], color=colors, rot=0)
axes[0, 1].set_title('Avg Missed Payments by Delinquency Status')
axes[0, 1].set_xlabel('')
axes[0, 1].set_xticklabels(['Non-Delinquent', 'Delinquent'])
axes[0, 1].set_ylabel('Avg Missed Payments')

# Plot 3: Credit Utilization Distribution
df[df['Delinquent_Account'] == 0]['Credit_Utilization'].hist(
    ax=axes[1, 0], alpha=0.6, color='#1E2761', label='Non-Delinquent', bins=20)
df[df['Delinquent_Account'] == 1]['Credit_Utilization'].hist(
    ax=axes[1, 0], alpha=0.6, color='#028090', label='Delinquent', bins=20)
axes[1, 0].set_title('Credit Utilization Distribution')
axes[1, 0].set_xlabel('Credit Utilization')
axes[1, 0].legend()

# Plot 4: Income vs Delinquency (counter-intuitive finding)
df.groupby('Delinquent_Account')['Income'].mean().plot(
    kind='bar', ax=axes[1, 1], color=colors, rot=0)
axes[1, 1].set_title('Avg Income by Delinquency Status\n(Counter-intuitive Finding!)')
axes[1, 1].set_xticklabels(['Non-Delinquent', 'Delinquent'])
axes[1, 1].set_ylabel('Average Income ($)')

plt.tight_layout()
plt.savefig('geldium_eda_dashboard.png', dpi=150, bbox_inches='tight')
print("  ✅ Saved: geldium_eda_dashboard.png")

# ── 10. SUMMARY ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("  EDA COMPLETE — KEY TAKEAWAYS")
print("=" * 55)
print("""
  1. Dataset: 500 records, 19 features, 16% delinquency rate
  2. Missing: Income (7.8%), Loan_Balance (5.8%) → Median imputed
              Credit_Score (0.4%) → Mean imputed
  3. Data Quality: Employment_Status had 3 variants → standardized
                   4 records with Credit_Utilization > 100% → flagged
  4. Top Predictors:
     🔴 Missed_Payments (strongest behavioral signal)
     🔴 Month_1 to Month_6 (sequential payment pattern)
     🟠 Debt_to_Income_Ratio (avg 30%)
     🟡 Credit_Utilization (avg 49%)
  5. Key Insight: Income ≠ safety. Behavioral features
     outweigh financial metrics for delinquency prediction.
""")
