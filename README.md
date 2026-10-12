# LendingClub Loan Default Prediction

UE24CS352A Machine Learning, Mini-Project

**Team:** Sohan MR, Sinchana Kulkarni

## Overview

Credit risk is the main concern for peer-to-peer lending investors. This project builds machine-learning classifiers that predict, from information available at application time, whether a LendingClub loan will be **Fully Paid** (label 1) or end in **Default / Charged Off** (label 0).

The project is inspired by the CS229 report *"LendingClub Loan Default and Profitability Prediction"* (Li & Han, 2018). We keep its dataset period (2012 to 2015) and model families, but deliberately avoid features that reveal the outcome (see [Data leakage](#data-leakage)).

## Dataset

- **Source:** Kaggle, "Lending Club Loan Data" (file `accepted_2007_to_2018Q4.csv`, uploaded by *wordsforthewise*). LendingClub's original download page is no longer available. The CSV is about 1.6 GB, so it is **not** included in this repository.
- **Raw size:** 2,260,701 rows, 151 columns.
- **Filtering:** loans issued 2012 to 2015 with a final outcome (`Fully Paid`, `Charged Off`, `Default`). `Current` and `Late` loans are dropped.
- **After filtering:** 786,820 loans, 81.4% positive (Fully Paid) and 18.6% negative.

### Data leakage

Only features known when the application is made are used (27 features: 20 numeric, 7 categorical). Columns such as `total_pymnt`, `recoveries` and `last_pymnt_d` are excluded, because they reveal whether the loan was repaid and would inflate every score.

## Repository contents

| File | Purpose |
|---|---|
| `lending_club_classifier.py` | v1 baseline: Logistic Regression, MLP, Random Forest, no class balancing for MLP/RF |
| `lending_club_classifier_v2.py` | Final pipeline: train/dev/test split, class balancing, ROC-AUC, threshold tuning, gradient boosting |
| `results_summary.csv` | Test-set metrics written by v2 |
| `roc_curves.png` | ROC curves written by v2 |
| `LendingClub_Default_Prediction.pptx` | Presentation slides |

## Setup

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle and place the CSV next to the scripts (or note its path).

## How to run

### Locally

```bash
# quick test on a random sample
python lending_club_classifier_v2.py --path accepted_2007_to_2018Q4.csv --sample 200000

# full run
python lending_club_classifier_v2.py --path accepted_2007_to_2018Q4.csv
```

### Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/accepted_2007_to_2018Q4.csv /content/accepted.csv
!python lending_club_classifier_v2.py --path /content/accepted.csv
```

The `--sample N` flag uses a random sample of N loans, which is handy for a quick demo (about 1 to 2 minutes). The full run takes roughly 5 minutes on free Colab. The script writes `results_summary.csv` and `roc_curves.png` to the working directory.

## Method

1. **Split:** stratified 60 / 20 / 20 train / dev / test. The dev set is used only to choose the decision threshold.
2. **Preprocessing** (fitted on the training set only): median imputation and standardization for numeric features; most-frequent imputation and one-hot encoding for categorical features. This gives 136 features.
3. **Models:**
   - Logistic Regression (`class_weight="balanced"`)
   - MLP (layers 32-16-8, ReLU), trained on a 1:1 undersampled training set
   - Random Forest (100 trees, depth 12, `class_weight="balanced_subsample"`)
   - HistGradientBoosting with balanced sample weights
4. **Threshold tuning:** the threshold on P(default) that maximises Default-class F1 on the dev set is applied to the test set.
5. **Metrics:** ROC-AUC, plus per-class precision, recall and F1. Accuracy alone is misleading with an 81/19 class split.

## Results (test set, 157,364 loans)

| Model | ROC-AUC | Default recall | Default precision | Default F1 |
|---|---|---|---|---|
| Logistic Regression | 0.7185 | 0.596 | 0.325 | 0.420 |
| Neural Network (MLP) | 0.7189 | 0.583 | 0.328 | 0.420 |
| Random Forest | 0.7147 | 0.580 | 0.324 | 0.416 |
| **HistGradientBoosting** | **0.7245** | 0.609 | 0.327 | 0.426 |

Recall, precision and F1 are for the Default class at the dev-tuned threshold.

### Baseline (v1) lesson

The first version used no class balancing for the MLP and Random Forest. They reached 0.82 accuracy, but caught only 7% (MLP) and 4% (Random Forest) of real defaults, because they predicted "Paid" almost every time. Always predicting "Paid" already gives 0.81 accuracy. Class weighting, undersampling and threshold tuning fixed this.

## Conclusions

- Gradient boosting is best (AUC 0.7245), but only about 0.006 ahead of Logistic Regression.
- With application-time features only, all four models plateau near 0.72 AUC.
- The reference paper reports about 0.89 weighted precision and recall. That figure is not directly comparable to ours, since our metrics are per-class and we exclude outcome-revealing columns.

## Limitations and future work

- Retrain without `grade`, `sub_grade` and `int_rate` (LendingClub's own risk assessment) to measure the models' independent signal.
- Add feature importance for the boosting model.
- Test on finalized 2016 to 2018 loans to check generalization over time.
- Add the regression models and loan-selection strategy from the reference paper.

## References

1. P. Li and G. Han, "LendingClub Loan Default and Profitability Prediction," Stanford CS229, 2018.
2. Kaggle, "Lending Club Loan Data."
3. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," JMLR 12, 2011.
4. Related public project: <https://github.com/yanhan-si/LendingClub-Loan-Default-Prediction> (LightGBM and MLP on 2007 to 2011 loans), consulted for comparison only.
