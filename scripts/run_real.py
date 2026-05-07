"""
Run the real Logistic Regression + LightGBM experiments end-to-end.
Mirrors the notebooks but inline so we can run unattended.

Outputs:
- LogisticRegression_Training and LightGBM_Training experiments on DagsHub
- ieee_fraud_best registered model (alias 'production' = best)
- submission.csv ready for kaggle
"""
import os
import sys
import gc
import time
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from src.preprocessing import (
    load_raw, EmailDomainCleaner, TransactionAmtFeats, CategoricalEncoder,
    NaNFiller, HighNullDropper, CorrelatedDropper,
)

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import dagshub

dagshub.init(repo_owner="gurtm23", repo_name="Fraud-Detection", mlflow=True)
print("tracking_uri:", mlflow.get_tracking_uri())

SEED = 42
np.random.seed(SEED)

DATA_PATH = os.environ.get("IEEE_DATA", os.path.join(ROOT, "data"))
print("loading data from:", DATA_PATH)
t0 = time.time()
train, test = load_raw(DATA_PATH)
print(f"loaded in {time.time()-t0:.1f}s   train: {train.shape}   test: {test.shape}")
print("fraud rate:", round(train["isFraud"].mean(), 4))


def build_pipe(model):
    return Pipeline([
        ("drop_high_null", HighNullDropper(threshold=0.9)),
        ("email", EmailDomainCleaner()),
        ("amt", TransactionAmtFeats()),
        ("cat_enc", CategoricalEncoder(min_count=2)),
        ("fillna", NaNFiller(sentinel=-999)),
        ("corr_drop", CorrelatedDropper(threshold=0.95)),
        ("model", model),
    ])


def run_experiment(name, base_model, params, n_folds=3):
    """Run all 5 MLflow runs for one model architecture."""
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    mlflow.set_experiment(f"{name}_Training")

    X_full = train.drop(columns=["isFraud", "TransactionID"])
    y_full = train["isFraud"]

    # 1. Cleaning run
    with mlflow.start_run(run_name=f"{name}_Cleaning"):
        mlflow.log_param("nan_threshold", 0.9)
        d = HighNullDropper(threshold=0.9).fit(X_full)
        mlflow.log_metric("n_cols_before", X_full.shape[1])
        mlflow.log_metric("n_cols_after", len(d.keep_))
        print(f"  [cleaning] kept {len(d.keep_)}/{X_full.shape[1]} cols")

    # 2. Feature Engineering run
    with mlflow.start_run(run_name=f"{name}_FeatureEngineering"):
        mlflow.log_params({"email_buckets": True, "amt_log": True, "amt_decimal": True, "tx_hour_day": True})
        fe = Pipeline([
            ("drop_high_null", HighNullDropper(threshold=0.9)),
            ("email", EmailDomainCleaner()),
            ("amt", TransactionAmtFeats()),
            ("cat_enc", CategoricalEncoder(min_count=2)),
            ("fillna", NaNFiller(sentinel=-999)),
        ])
        # fit on subsample for speed (just for the metric log)
        sub = X_full.sample(min(50_000, len(X_full)), random_state=0)
        X_fe = fe.fit_transform(sub)
        mlflow.log_metric("n_features_after_fe", X_fe.shape[1])
        print(f"  [FE] {X_fe.shape[1]} features after FE")
        del fe, X_fe, sub
        gc.collect()

    # 3. Feature Selection run
    with mlflow.start_run(run_name=f"{name}_FeatureSelection"):
        mlflow.log_param("corr_threshold", 0.95)
        # done lazily inside the pipeline; just log the parameter
        print("  [FS] correlation pruning |r|>0.95 (applied in pipeline)")

    # 4. Cross-Validation run
    pipe = build_pipe(base_model)
    pipe.set_params(**{f"model__{k}": v for k, v in params.items()})

    aucs = []
    with mlflow.start_run(run_name=f"{name}_CrossValidation"):
        mlflow.log_params(params)
        mlflow.log_param("n_folds", n_folds)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        for fold, (tr, va) in enumerate(skf.split(X_full, y_full)):
            t = time.time()
            pipe.fit(X_full.iloc[tr], y_full.iloc[tr])
            pred = pipe.predict_proba(X_full.iloc[va])[:, 1]
            auc = roc_auc_score(y_full.iloc[va], pred)
            aucs.append(auc)
            mlflow.log_metric("fold_auc", auc, step=fold)
            print(f"  [CV] fold {fold}  AUC = {auc:.5f}   ({time.time()-t:.1f}s)")
        mlflow.log_metric("cv_mean_auc", float(np.mean(aucs)))
        mlflow.log_metric("cv_std_auc", float(np.std(aucs)))
        print(f"  [CV] mean AUC = {np.mean(aucs):.5f}  (std {np.std(aucs):.5f})")

    # 5. FinalFit run + log model
    with mlflow.start_run(run_name=f"{name}_FinalFit") as run:
        mlflow.log_params(params)
        mlflow.log_metric("cv_mean_auc", float(np.mean(aucs)))

        Xtr, Xva, ytr, yva = train_test_split(
            X_full, y_full, test_size=0.15, stratify=y_full, random_state=SEED,
        )
        pipe.fit(Xtr, ytr)
        train_auc = roc_auc_score(ytr, pipe.predict_proba(Xtr)[:, 1])
        val_auc = roc_auc_score(yva, pipe.predict_proba(Xva)[:, 1])
        mlflow.log_metric("train_auc", train_auc)
        mlflow.log_metric("val_auc", val_auc)
        gap = train_auc - val_auc
        mlflow.log_metric("train_val_gap", gap)
        if gap > 0.05:
            mlflow.set_tag("fit_status", "overfitting")
        elif val_auc < 0.85:
            mlflow.set_tag("fit_status", "underfitting")
        else:
            mlflow.set_tag("fit_status", "balanced")
        print(f"  [final] train AUC {train_auc:.5f}   val AUC {val_auc:.5f}   gap {gap:.5f}")

        pipe.fit(X_full, y_full)   # full refit for the saved artifact
        sig = infer_signature(X_full.head(5), pipe.predict_proba(X_full.head(5)))
        info = mlflow.sklearn.log_model(pipe, name="pipeline", signature=sig)
        print(f"  [final] model_uri = {info.model_uri}")

    return float(np.mean(aucs)), info.model_uri, pipe


# ---------------- Run Logistic Regression ----------------
lr_cv, lr_uri, lr_pipe = run_experiment(
    "LogisticRegression",
    LogisticRegression(max_iter=200, solver="liblinear", n_jobs=None),
    {"C": 0.1, "penalty": "l2", "class_weight": "balanced"},
    n_folds=3,
)

# ---------------- Run LightGBM ----------------
lgbm_cv, lgbm_uri, lgbm_pipe = run_experiment(
    "LightGBM",
    LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
    {
        "n_estimators": 800, "learning_rate": 0.05, "num_leaves": 127,
        "min_data_in_leaf": 100, "feature_fraction": 0.7, "bagging_fraction": 0.85,
        "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 0.1,
        "random_state": SEED, "n_jobs": -1, "verbose": -1,
    },
    n_folds=3,
)

# ---------------- Register the winner ----------------
print("\n" + "="*60)
print(f"  LR  CV AUC: {lr_cv:.5f}")
print(f"  LGBM CV AUC: {lgbm_cv:.5f}")
winner_uri = lgbm_uri if lgbm_cv > lr_cv else lr_uri
winner_pipe = lgbm_pipe if lgbm_cv > lr_cv else lr_pipe
winner_name = "LightGBM" if lgbm_cv > lr_cv else "LogisticRegression"
print(f"  -> winner: {winner_name}")
print("="*60)

client = MlflowClient()
result = mlflow.register_model(model_uri=winner_uri, name="ieee_fraud_best")
client.set_registered_model_alias("ieee_fraud_best", "production", result.version)
print(f"registered ieee_fraud_best v{result.version}, alias 'production' -> v{result.version}")

# ---------------- Generate submission ----------------
print("\nrunning inference on raw test set ...")
X_test = test.drop(columns=["TransactionID"])
proba = winner_pipe.predict_proba(X_test)[:, 1]
sub = pd.DataFrame({"TransactionID": test["TransactionID"], "isFraud": proba})
sub_path = os.path.join(ROOT, "submission.csv")
sub.to_csv(sub_path, index=False)
print(f"submission.csv written: {sub.shape}   range {proba.min():.4f}-{proba.max():.4f}")

# write summary for the README update
with open(os.path.join(ROOT, "_run_summary.txt"), "w") as f:
    f.write(f"winner: {winner_name}\n")
    f.write(f"LogisticRegression CV AUC: {lr_cv:.5f}\n")
    f.write(f"LightGBM CV AUC: {lgbm_cv:.5f}\n")
    f.write(f"submission rows: {len(sub)}\n")

print("\nDONE")
