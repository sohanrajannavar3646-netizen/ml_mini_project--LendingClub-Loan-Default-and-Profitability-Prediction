"""
LendingClub loan default prediction - v2.

Changes from v1:
  * 60/20/20 train/dev/test split (dev used for threshold tuning)
  * Class imbalance handled for every model
  * ROC-AUC reported for every model
  * Decision threshold tuned on dev set (maximises F1 of the Default class)
  * Added HistGradientBoosting (LightGBM-style) model
  * Saves results_summary.csv and roc_curves.png

Usage:
    python lending_club_classifier_v2.py --path /content/accepted.csv --sample 200000   # quick test
    python lending_club_classifier_v2.py --path /content/accepted.csv                   # full run
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42

# Only features known at application time (no leakage from payment columns).
NUMERIC = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "mths_since_last_delinq",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
    "pub_rec_bankruptcies", "term_months", "emp_length_years", "credit_history_years",
]
CATEGORICAL = [
    "grade", "sub_grade", "home_ownership", "verification_status",
    "purpose", "addr_state", "application_type",
]
RAW_COLS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade", "emp_length",
    "home_ownership", "annual_inc", "verification_status", "issue_d", "loan_status",
    "purpose", "addr_state", "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low",
    "fico_range_high", "inq_last_6mths", "mths_since_last_delinq", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "application_type", "mort_acc",
    "pub_rec_bankruptcies",
]


def load_data(path, sample=None):
    df = pd.read_csv(path, usecols=RAW_COLS, low_memory=False)
    print(f"Raw rows: {len(df):,}")
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df[df["issue_d"].dt.year.between(2012, 2015)]
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].copy()
    df["target"] = (df["loan_status"] == "Fully Paid").astype(int)
    print(f"After filtering: {len(df):,} rows, {df['target'].mean():.1%} positive (Fully Paid)")

    df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float)
    emp = df["emp_length"].str.extract(r"(\d+)")[0].astype(float)
    emp[df["emp_length"] == "< 1 year"] = 0
    df["emp_length_years"] = emp
    cr_year = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce").dt.year
    df["credit_history_years"] = df["issue_d"].dt.year - cr_year

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=SEED)
        print(f"Sampled down to {len(df):,} rows")
    return df


def build_preprocessor():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                     sparse_output=False))])
    return ColumnTransformer([("num", numeric, NUMERIC),
                              ("cat", categorical, CATEGORICAL)])


def undersample(X, y, seed=SEED):
    """Balance classes 1:1 by randomly dropping majority-class rows."""
    rng = np.random.default_rng(seed)
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n = min(len(idx0), len(idx1))
    keep = np.concatenate([rng.choice(idx0, n, replace=False),
                           rng.choice(idx1, n, replace=False)])
    rng.shuffle(keep)
    return X[keep], y[keep]


def best_default_threshold(y_dev, p_paid_dev):
    """Threshold on P(default) that maximises F1 of the Default class on dev."""
    p_def = 1 - p_paid_dev
    prec, rec, thr = precision_recall_curve((y_dev == 0).astype(int), p_def)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return thr[np.argmax(f1[:-1])]


def report(title, y_true, y_pred):
    print(f"\n--- {title} ---")
    print("Confusion matrix (rows=true, cols=pred; 0=Default, 1=Paid):")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred,
                                target_names=["Default", "Paid Off"], digits=3))


def run_model(name, model, fit_kwargs, X_tr, y_tr, X_dev, y_dev, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr, **fit_kwargs)
    secs = time.time() - t0
    print(f"\n{'=' * 60}\n{name}  (trained in {secs:.0f}s)\n{'=' * 60}")

    p_dev = model.predict_proba(X_dev)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p_te)
    print(f"Test ROC-AUC: {auc:.4f}")

    # Default 0.5 threshold
    pred_default = (p_te >= 0.5).astype(int)
    report("TEST @ threshold 0.5", y_te, pred_default)

    # Tuned threshold (chosen on dev, applied to test)
    thr_def = best_default_threshold(y_dev, p_dev)
    pred_tuned = (1 - p_te < thr_def).astype(int)  # predict Paid unless P(default) >= thr
    report(f"TEST @ tuned threshold (P(default) >= {thr_def:.3f})", y_te, pred_tuned)

    rep = classification_report(y_te, pred_tuned, output_dict=True)
    rep05 = classification_report(y_te, pred_default, output_dict=True)
    row = {
        "model": name, "test_auc": round(auc, 4), "train_seconds": round(secs),
        "default_recall@0.5": round(rep05["0"]["recall"], 3),
        "default_precision@0.5": round(rep05["0"]["precision"], 3),
        "tuned_threshold": round(float(thr_def), 3),
        "default_recall_tuned": round(rep["0"]["recall"], 3),
        "default_precision_tuned": round(rep["0"]["precision"], 3),
        "default_f1_tuned": round(rep["0"]["f1-score"], 3),
        "accuracy_tuned": round(rep["accuracy"], 3),
    }
    return row, p_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()

    df = load_data(args.path, args.sample)
    X, y = df[NUMERIC + CATEGORICAL], df["target"].values

    # 60 / 20 / 20 split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
    X_dev, X_te, y_dev, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                                random_state=SEED, stratify=y_tmp)
    print(f"Split sizes - train: {len(y_tr):,}  dev: {len(y_dev):,}  test: {len(y_te):,}")

    pre = build_preprocessor()
    X_tr = pre.fit_transform(X_tr).astype(np.float32)   # fit on train only
    X_dev = pre.transform(X_dev).astype(np.float32)
    X_te = pre.transform(X_te).astype(np.float32)
    print(f"Feature matrix: {X_tr.shape[1]} features after encoding")

    # Balanced 1:1 training set for the MLP (MLPClassifier has no class_weight)
    X_tr_bal, y_tr_bal = undersample(X_tr, y_tr)

    runs = [
        ("Logistic Regression",
         LogisticRegression(class_weight="balanced", max_iter=1000, n_jobs=-1),
         {}, (X_tr, y_tr)),
        ("Neural Network (MLP, undersampled)",
         MLPClassifier(hidden_layer_sizes=(32, 16, 8), activation="relu", batch_size=256,
                       learning_rate_init=1e-3, alpha=1e-4, early_stopping=True,
                       max_iter=100, random_state=SEED),
         {}, (X_tr_bal, y_tr_bal)),
        ("Random Forest (balanced)",
         RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=20,
                                max_features="sqrt", class_weight="balanced_subsample",
                                n_jobs=-1, random_state=SEED),
         {}, (X_tr, y_tr)),
        ("HistGradientBoosting (balanced)",
         HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                        early_stopping=True, random_state=SEED),
         {"sample_weight": compute_sample_weight("balanced", y_tr)}, (X_tr, y_tr)),
    ]

    rows, probs = [], {}
    for name, model, fit_kwargs, (Xf, yf) in runs:
        row, p_te = run_model(name, model, fit_kwargs, Xf, yf, X_dev, y_dev, X_te, y_te)
        rows.append(row)
        probs[name] = p_te

    summary = pd.DataFrame(rows)
    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    print(summary.to_string(index=False))
    summary.to_csv("results_summary.csv", index=False)

    # ROC curves
    plt.figure(figsize=(6, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves (test set)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    print("\nSaved results_summary.csv and roc_curves.png")


if __name__ == "__main__":
    main()
