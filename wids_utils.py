"""Helper functions for the WiDS Datathon 2020 ICU-mortality project.

The notebook ``project.ipynb`` explains *why* each step is done; this module
holds the *how*, so that the notebook stays short and readable.

Sections
--------
1. Loading and cleaning
2. Feature matrices
3. Model factories (logistic regression, random forest, LightGBM)
4. Cross-validation
5. Metrics (threshold choice, bootstrap, net benefit, subgroups)
6. Plots
"""
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "hospital_death"
ID_COLUMNS = ["encounter_id", "patient_id", "hospital_id", "icu_id"]
APACHE_PREDICTIONS = ["apache_4a_hospital_death_prob", "apache_4a_icu_death_prob"]
DIAGNOSIS_CODES = ["apache_2_diagnosis", "apache_3j_diagnosis"]


# ---------------------------------------------------------------------------
# 1. Loading and cleaning
# ---------------------------------------------------------------------------
def load_raw_data(data_dir):
    """Return the labelled training table and the data dictionary."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "training_v2.csv")
    dictionary = pd.read_csv(data_dir / "WiDS Datathon 2020 Dictionary.csv")
    return train, dictionary


def min_max_pairs(columns):
    """List (min_column, max_column) pairs present in ``columns``."""
    return [(col, col[:-4] + "_max") for col in columns
            if col.endswith("_min") and col[:-4] + "_max" in columns]


def count_inverted_min_max(df):
    """Number of (row, measurement) cells where the minimum exceeds the maximum."""
    return int(sum((df[lo] > df[hi]).sum() for lo, hi in min_max_pairs(df.columns)))


def swap_inverted_min_max(df):
    """Swap min and max values that were recorded in the wrong order."""
    df = df.copy()
    for lo, hi in min_max_pairs(df.columns):
        inverted = df[lo] > df[hi]
        df.loc[inverted, [lo, hi]] = df.loc[inverted, [hi, lo]].to_numpy()
    return df


def clean_data(raw):
    """Apply the cleaning steps justified in the notebook (section 2.3)."""
    df = raw.drop(columns=["readmission_status"])  # constant column (always 0)
    df["apache_2_bodysystem"] = df["apache_2_bodysystem"].replace(
        "Undefined Diagnoses", "Undefined diagnoses")
    # -1 is a sentinel for "APACHE could not be computed", not a probability.
    df[APACHE_PREDICTIONS] = df[APACHE_PREDICTIONS].where(df[APACHE_PREDICTIONS] >= 0)
    # A negative stay before ICU admission is impossible: treat as unknown.
    df["pre_icu_los_days"] = df["pre_icu_los_days"].where(df["pre_icu_los_days"] >= 0)
    df = swap_inverted_min_max(df)
    # Diagnosis codes are labels, not quantities: 502.01 is not "more" than 501.05.
    # The integer part of the APACHE III code is the diagnosis group.
    df["apache_3j_diagnosis"] = df["apache_3j_diagnosis"].floordiv(1)
    for col in DIAGNOSIS_CODES:
        df[col] = df[col].astype("Int64").astype(str).replace("<NA>", np.nan)
    return df


# ---------------------------------------------------------------------------
# 2. Feature matrices
# ---------------------------------------------------------------------------
def categorical_columns(df):
    """Names of the non-numeric columns."""
    return df.select_dtypes(exclude="number").columns.tolist()


def build_features(df, include_apache=False):
    """Return X (features), y (target) and groups (hospital of each stay).

    Identifiers are removed: they describe *where* or *who*, not the patient's
    physiology, and would not generalise to a new hospital.
    """
    drop = ID_COLUMNS + [TARGET] + ([] if include_apache else APACHE_PREDICTIONS)
    X = df.drop(columns=drop)
    numeric = X.select_dtypes("number").columns
    # Number of missing measurements: a proxy for how intensively the patient was monitored.
    X["n_missing_values"] = X[numeric].isna().sum(axis=1)
    for col in categorical_columns(X):
        X[col] = X[col].astype("category")
    return X, df[TARGET].to_numpy(), df["hospital_id"].to_numpy()


def missingness_matrix(X):
    """1 if a numeric measurement is missing, 0 otherwise (no values at all)."""
    numeric = X.drop(columns="n_missing_values").select_dtypes("number")
    return numeric.isna().astype(int)


# ---------------------------------------------------------------------------
# 3. Model factories
# ---------------------------------------------------------------------------
def make_logistic_regression(X, C=0.1):
    """Median imputation + missing indicators + scaling + one-hot, then L2 logistic regression."""
    categorical = categorical_columns(X)
    numeric = [c for c in X.columns if c not in categorical]
    preprocess = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              StandardScaler()), numeric),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                              sparse_output=False), categorical),
    ])
    model = LogisticRegression(C=C, max_iter=2000)
    return make_pipeline(preprocess, model)


def make_random_forest(X, **params):
    """Ordinal-encode categories; numeric NaNs are handled natively by sklearn trees."""
    categorical = categorical_columns(X)
    encode = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-2), categorical)],
        remainder="passthrough")
    defaults = dict(n_estimators=300, min_samples_leaf=10, max_features="sqrt",
                    max_samples=0.5, n_jobs=-1, random_state=RANDOM_STATE)
    return make_pipeline(encode, RandomForestClassifier(**{**defaults, **params}))


def make_lightgbm(**params):
    """LightGBM handles NaNs and pandas categoricals natively."""
    defaults = dict(n_estimators=500, learning_rate=0.03, num_leaves=31,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    return lgb.LGBMClassifier(**{**defaults, **params})


# ---------------------------------------------------------------------------
# 4. Cross-validation
# ---------------------------------------------------------------------------
def cross_validated_predictions(model, X, y, cv, groups=None):
    """Out-of-fold probabilities plus the ROC-AUC and PR-AUC of each fold.

    Every patient is predicted by a model that never saw them (nor, with a
    grouped splitter, anyone from their hospital).
    """
    oof = np.zeros(len(y))
    fold_scores = []
    for train_idx, valid_idx in cv.split(X, y, groups):
        fitted = clone(model).fit(X.iloc[train_idx], y[train_idx])
        oof[valid_idx] = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
        fold_scores.append({"roc_auc": roc_auc_score(y[valid_idx], oof[valid_idx]),
                            "pr_auc": average_precision_score(y[valid_idx], oof[valid_idx])})
    return oof, pd.DataFrame(fold_scores)


def tune_hyperparameters(model, param_distributions, X, y, groups, cv, n_iter, n_jobs=1):
    """Random search scored by grouped-CV ROC-AUC; returns the table of tried settings."""
    search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, scoring="roc_auc",
                                cv=cv, n_jobs=n_jobs, refit=False, random_state=RANDOM_STATE)
    search.fit(X, y, groups=groups)
    results = pd.DataFrame(search.cv_results_)
    columns = ["params", "mean_test_score", "std_test_score", "mean_fit_time"]
    return search.best_params_, results[columns].sort_values("mean_test_score", ascending=False)


def summarise_folds(fold_scores):
    """'mean ± std' strings for each metric column of a fold-score table."""
    return {col: f"{fold_scores[col].mean():.4f} ± {fold_scores[col].std():.4f}"
            for col in fold_scores.columns}


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------
def threshold_for_recall(y, proba, target_recall):
    """Highest threshold whose recall is still >= ``target_recall``.

    A higher threshold means fewer alerts, so we take the highest one that
    keeps the clinical goal (catching ``target_recall`` of the deaths).
    """
    _, recall, thresholds = precision_recall_curve(y, proba)
    feasible = thresholds[recall[:-1] >= target_recall]
    return float(feasible.max())


def threshold_for_best_f1(y, proba):
    """Threshold that maximises F1 (used only as a comparison point)."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate_predictions(y, proba, threshold):
    """Threshold-free and threshold-dependent metrics in one dictionary."""
    pred = proba >= threshold
    tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
    fn, tn = np.sum(~pred & (y == 1)), np.sum(~pred & (y == 0))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "ROC-AUC": roc_auc_score(y, proba),
        "PR-AUC": average_precision_score(y, proba),
        "Brier": brier_score_loss(y, proba),
        "recall": recall,
        "precision": precision,
        "F1": 2 * precision * recall / max(precision + recall, 1e-12),
        "specificity": tn / max(tn + fp, 1),
        "alert rate": pred.mean(),
    }


def bootstrap_metric(y, proba_a, metric, proba_b=None, n_boot=500, seed=RANDOM_STATE):
    """95% bootstrap interval of ``metric`` (or of metric(a) - metric(b) if ``proba_b`` is given)."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():  # resample with one class only
            continue
        value = metric(y[idx], proba_a[idx])
        if proba_b is not None:
            value -= metric(y[idx], proba_b[idx])
        values.append(value)
    return np.percentile(values, [2.5, 97.5])


def net_benefit(y, proba, thresholds):
    """Decision-curve net benefit: TP/n - FP/n * t / (1 - t) for each threshold t.

    The ratio t/(1-t) is the exchange rate between a false alarm and a missed
    death that is implied by choosing threshold t.
    """
    n = len(y)
    benefits = []
    for t in thresholds:
        pred = proba >= t
        tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
        benefits.append(tp / n - fp / n * t / (1 - t))
    return np.array(benefits)


def subgroup_report(y, proba, threshold, groups, min_deaths=20):
    """Per-subgroup size, mortality, discrimination, recall and calibration."""
    rows = []
    for name in pd.Series(groups).dropna().unique():
        mask = np.asarray(groups == name)
        if y[mask].sum() < min_deaths:
            continue
        rows.append({
            "group": name,
            "n": int(mask.sum()),
            "deaths": int(y[mask].sum()),
            "observed mortality": y[mask].mean(),
            "mean predicted risk": proba[mask].mean(),
            "ROC-AUC": roc_auc_score(y[mask], proba[mask]),
            "recall": np.mean(proba[mask][y[mask] == 1] >= threshold),
            "alert rate": np.mean(proba[mask] >= threshold),
        })
    return pd.DataFrame(rows).set_index("group").sort_values("n", ascending=False)


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------
def plot_roc_and_pr(y, probas, prevalence):
    """ROC and precision-recall curves for several models side by side."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y, proba)
        precision, recall, _ = precision_recall_curve(y, proba)
        ax_roc.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, proba):.3f})")
        ax_pr.plot(recall, precision, label=f"{name} ({average_precision_score(y, proba):.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance (0.500)")
    ax_pr.axhline(prevalence, color="k", ls="--", lw=0.8, label=f"chance ({prevalence:.3f})")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    for ax in (ax_roc, ax_pr):
        ax.legend(fontsize=8, title="model (AUC)", title_fontsize=8)
    fig.tight_layout()
    return fig


def plot_calibration(y, probas, n_bins=10):
    """Reliability diagram: mean predicted risk vs observed death rate per risk decile."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    for name, proba in probas.items():
        observed, predicted = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")
        ax.plot(predicted, observed, "o-", ms=4,
                label=f"{name} (Brier {brier_score_loss(y, proba):.4f})")
    ax.set(xlabel="Mean predicted risk", ylabel="Observed mortality",
           title="Calibration (deciles of predicted risk)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_decision_curve(y, probas, thresholds):
    """Net benefit of each model vs 'alert everyone' and 'alert no one'."""
    prevalence = y.mean()
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, proba in probas.items():
        ax.plot(thresholds, net_benefit(y, proba, thresholds), label=name)
    ax.plot(thresholds, treat_all, "k:", label="alert everyone")
    ax.axhline(0, color="k", lw=0.8, label="alert no one")
    ax.set(xlabel="Risk threshold", ylabel="Net benefit", title="Decision curve",
           ylim=(-0.01, prevalence * 1.1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
