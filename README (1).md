# Interpretable Student Outcome Prediction

A machine learning project that predicts student academic outcomes (pass/fail) using demographic and social features from the UCI Student Performance Dataset. The project emphasises model interpretability through SHAP analysis and evaluates algorithmic fairness across protected attributes.

## Author

**Advaith Katkojwala**  
MSc Artificial Intelligence — Pattern Recognition and Machine Learning (Summer 2026 Retake)

## Project Overview

This project addresses a binary classification task: predicting whether a student will pass (G3 ≥ 10) or fail (G3 < 10) their final examination. Crucially, the prediction relies solely on **pre-outcome features** (30 demographic, social, and school-related attributes), deliberately excluding interim grades (G1, G2) to simulate an early-warning intervention scenario.

Three classifiers are compared:

| Model | Test Macro F1 | CV Mean Macro F1 |
|-------|:---:|:---:|
| Dummy Baseline (Most Frequent) | 0.399 | 0.399 |
| Logistic Regression | 0.759 | 0.742 |
| Decision Tree (Balanced) | 0.684 | 0.689 |

## Dataset

- **Source:** UCI Student Performance Dataset (Cortez and Silva, 2008)
- **Subjects:** Mathematics (395 samples) + Portuguese (649 samples) = **1,044 combined**
- **Features:** 30 pre-outcome attributes (demographic, family, school, lifestyle)
- **Target:** Binary — Pass (G3 ≥ 10) vs Fail (G3 < 10)
- **Class distribution:** ~67% Pass, ~33% Fail

## Repository Structure

```
├── README.md
├── student_outcome_prediction.py    # Main ML pipeline (local execution)
├── kaggle_notebook_final.py         # Kaggle-ready notebook version
├── generate_data.py                 # Synthetic data generator
├── build_report.py                  # Word report generator (python-docx)
├── student-mat.csv                  # Mathematics dataset
├── student-por.csv                  # Portuguese dataset
└── outputs/                         # Generated figures and results
    ├── fig1_target_distribution.png
    ├── fig2_grade_distribution.png
    ├── fig3_correlation_heatmap.png
    ├── fig4_model_comparison.png
    ├── fig5_confusion_matrices.png
    ├── fig6_lr_coefficients.png
    ├── fig7_dt_feature_importance.png
    ├── fig8_decision_tree.png
    ├── fig9_shap_summary_dt.png
    ├── fig10_shap_bar_dt.png
    ├── fig11_shap_summary_lr.png
    ├── fig12_fairness_gender.png
    ├── fig13_fairness_address.png
    ├── fig14_proxy_correlation.png
    ├── fig15_feature_distributions.png
    └── results_summary.csv
```

## Methodology

### Preprocessing
- Combined mathematics and Portuguese datasets with a `subject` identifier
- Label-encoded 17 categorical features using scikit-learn `LabelEncoder`
- Applied `StandardScaler` to features for Logistic Regression
- Excluded G1, G2, G3 from the feature set (pre-outcome prediction)

### Models
1. **DummyClassifier** (most_frequent strategy) — naive baseline
2. **Logistic Regression** — linear model with L2 regularisation, StandardScaler preprocessing
3. **Decision Tree** — max_depth=5, min_samples_split=10, min_samples_leaf=5, class_weight='balanced'

### Evaluation
- **Primary metric:** Macro F1 Score (accounts for class imbalance)
- **Validation:** Stratified 5-Fold Cross-Validation
- **Test split:** 80/20 stratified hold-out

### Interpretability
- **Logistic Regression coefficients** — feature direction and magnitude
- **Decision Tree Gini importance** — feature ranking by split contribution
- **SHAP TreeExplainer** — local and global explanations for Decision Tree
- **SHAP KernelExplainer** — model-agnostic explanations for Logistic Regression (100-sample subset)

### Fairness Analysis
- **Protected attributes:** sex (male/female), address (urban/rural)
- **Metrics:** Statistical Parity Ratio, Equal Opportunity Ratio (threshold ≥ 0.8)
- **Proxy risk assessment:** Pearson correlation of non-protected features with protected attributes

## Requirements

```
Python >= 3.8
numpy
pandas
matplotlib
seaborn
scikit-learn
shap
python-docx (for report generation only)
```

Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn shap python-docx
```

## Usage

### Run locally
```bash
python student_outcome_prediction.py
```

### Run on Kaggle
Upload `student-mat.csv` and `student-por.csv` as a Kaggle dataset, then run `kaggle_notebook_final.py` in a Kaggle notebook.

**Kaggle Notebook:** [https://www.kaggle.com/code/coolboyw3r21](https://www.kaggle.com/code/coolboyw3r21)

## Key Findings

1. **Logistic Regression outperforms Decision Tree** on Macro F1 (0.759 vs 0.684), demonstrating that a well-regularised linear model can capture the predictive signal in demographic features effectively.

2. **`failures` is the strongest predictor** across all interpretability methods — prior academic failures are the single most informative feature for predicting future outcomes.

3. **Family education (Medu, Fedu) and study time** contribute positively to pass probability, aligning with established educational research.

4. **Fairness analysis reveals acceptable parity** across gender and urban/rural groups, with both Statistical Parity Ratio and Equal Opportunity Ratio generally above the 0.8 threshold.

5. **Proxy risk is low** — correlations between non-protected features and protected attributes remain below concerning thresholds, suggesting limited indirect discrimination.

## Outputs

The pipeline generates 15 publication-quality figures and a CSV summary table. All figures are saved to the `outputs/` directory at 150 DPI.

## References

Cortez, P. and Silva, A. (2008) 'Using data mining to predict secondary school student performance', *Proceedings of 5th FUture BUsiness TEChnology Conference*, pp. 5–12.

Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', *Advances in Neural Information Processing Systems*, 30, pp. 4765–4774.

## Licence

This project is submitted as coursework for academic assessment. The UCI Student Performance Dataset is publicly available under its original terms.
