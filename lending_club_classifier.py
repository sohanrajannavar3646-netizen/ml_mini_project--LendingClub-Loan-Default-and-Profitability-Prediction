"""
LendingClub loan default prediction (classification).

Usage:
    python lending_club_classifier.py --path accepted_2007_to_2018Q4.csv
    python lending_club_classifier.py --path accepted_2007_to_2018Q4.csv --sample 100000   # quick demo run
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

    # Keep loans issued 2012-2015 with a final outcome
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df[df["issue_d"].dt.year.between(2012, 2015)]
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].copy()
    df["target"] = (df["loan_status"] == "Fully Paid").astype(int)
    print(f"After filtering: {len(df):,} rows, {df['target'].mean():.1%} positive (Fully Paid)")

    # Feature engineering
    df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float)
    emp = df["emp_length"].str.extract(r"(\d+)")[0].astype(float)
    emp[df["emp_length"] == "< 1 year"] = 0
    df["emp_length_years"] = emp
    cr_year = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce").dt.year
    df["credit_history_years"] = df["issue_d"].dt.year - cr_year

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
        print(f"Sampled down to {len(df):,} rows")
    return df


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def evaluate(name, model, X_train, y_train, X_test, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"\n{'=' * 60}\n{name}  (trained in {time.time() - t0:.0f}s)\n{'=' * 60}")
    for split, X, y in [("TRAIN", X_train, y_train), ("TEST", X_test, y_test)]:
        pred = model.predict(X)
        print(f"\n--- {split} ---")
        print("Confusion matrix (rows=true, cols=predicted; 0=Default, 1=Paid):")
        print(confusion_matrix(y, pred))
        print(classification_report(y, pred, target_names=["Default", "Paid Off"], digits=3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to accepted_2007_to_2018Q4.csv")
    ap.add_argument("--sample", type=int, default=None, help="Use a random sample of N rows")
    args = ap.parse_args()

    df = load_data(args.path, args.sample)
    X, y = df[NUMERIC + CATEGORICAL], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pre = build_preprocessor()
    X_train_p = pre.fit_transform(X_train)   # fit on train only
    X_test_p = pre.transform(X_test)
    print(f"Feature matrix: {X_train_p.shape[1]} features after encoding")

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, n_jobs=-1),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(10, 10, 5, 3), activation="logistic",
            batch_size=200, learning_rate_init=0.001, alpha=1e-4,
            early_stopping=True, max_iter=200, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=5,
            max_features="sqrt", n_jobs=-1, random_state=42),
    }
    for name, model in models.items():
        evaluate(name, model, X_train_p, y_train, X_test_p, y_test)


if __name__ == "__main__":
    main()