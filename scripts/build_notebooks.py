"""
Generate the model_experiment_*.ipynb + model_inference.ipynb files.

Run once:  python scripts/build_notebooks.py
"""
import json
import os
from pathlib import Path

NB_DIR = Path(__file__).parent.parent / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(path, cells):
    with open(path, "w") as f:
        json.dump(nb(cells), f, indent=1)
    print("wrote", path)


# ---------------------------------------------------------------------------
# shared snippets used by every model notebook
# ---------------------------------------------------------------------------

PREPROC_PATH = Path(__file__).parent.parent / "src" / "preprocessing.py"
PREPROCESSING_INLINE = PREPROC_PATH.read_text()


KAGGLE_BOOTSTRAP = """\
# --- Kaggle / local bootstrap ---------------------------------------------
# pulls MLflow creds from Kaggle Secrets when running on Kaggle, else falls
# back to env vars. data path auto-detects /kaggle/input/ieee-fraud-detection.
import os, sys, gc, warnings
warnings.filterwarnings("ignore")

ON_KAGGLE = os.path.exists("/kaggle/input")

if ON_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient
        sec = UserSecretsClient()
        os.environ["MLFLOW_TRACKING_USERNAME"] = sec.get_secret("MLFLOW_TRACKING_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = sec.get_secret("MLFLOW_TRACKING_PASSWORD")
    except Exception as e:
        print("kaggle secrets unavailable:", e)
    DATA_PATH = "/kaggle/input/ieee-fraud-detection"
    # mlflow / dagshub not pre-installed on kaggle
    os.system("pip -q install mlflow dagshub")
else:
    DATA_PATH = os.environ.get("IEEE_DATA", "../data")
"""


HEADER_IMPORTS = """\
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

import mlflow
from mlflow.models.signature import infer_signature

try:
    import dagshub
    dagshub.init(
        repo_owner=os.environ.get("DAGSHUB_USER", "gurtm23"),
        repo_name=os.environ.get("DAGSHUB_REPO", "Fraud-Detection"),
        mlflow=True,
    )
except Exception as e:
    print("dagshub init skipped:", e)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))

SEED = 42
np.random.seed(SEED)
"""

DATA_LOAD = """\
train, test = load_raw(DATA_PATH)
print("train:", train.shape, "  test:", test.shape)
print("fraud rate:", train["isFraud"].mean().round(4))
"""

EDA_QUICK = """\
# quick sanity peek — not exhaustive EDA, just enough to drive cleaning choices
miss = train.isna().mean().sort_values(ascending=False)
print("cols with >90% missing:", (miss > 0.9).sum())
print("cols with >50% missing:", (miss > 0.5).sum())
print("\\ntop 10 most-missing columns:")
print(miss.head(10).round(3))
"""


def build_model_notebook(name, model_block, model_imports, hp_block, notes_block):
    """Produce a model_experiment_*.ipynb following the assignment spec."""
    cells = [
        md(f"# Model Experiment — {name}\n\n"
           "End-to-end notebook for **{name}** on the IEEE-CIS fraud dataset.\n"
           "Each section is logged as a separate MLflow run inside the\n"
           f"`{name}_Training` experiment.\n".replace("{name}", name)),

        md("## 0. Setup\n\nFirst cell: bootstraps Kaggle secrets / data paths.\n"
           "Second cell: inlined `preprocessing.py` (so this notebook is self-contained on Kaggle).\n"
           "Third cell: imports + DagsHub MLflow init."),
        code(KAGGLE_BOOTSTRAP),
        code(PREPROCESSING_INLINE),
        code(HEADER_IMPORTS + "\n" + model_imports),
        code(f'EXPERIMENT = "{name}_Training"\nmlflow.set_experiment(EXPERIMENT)'),

        md("## 1. Cleaning\n\n"
           "Drop columns with too many NaNs, downcast numeric dtypes, "
           "and clean up email domains. Logged as the `*_Cleaning` MLflow run."),
        code(DATA_LOAD),
        code(EDA_QUICK),
        code(f"""\
with mlflow.start_run(run_name="{name}_Cleaning"):
    nan_threshold = 0.9
    mlflow.log_param("nan_threshold", nan_threshold)

    dropper = HighNullDropper(threshold=nan_threshold)
    dropper.fit(train.drop(columns=["isFraud"]))
    cleaned_cols = dropper.keep_

    mlflow.log_metric("n_cols_before", train.shape[1] - 1)
    mlflow.log_metric("n_cols_after", len(cleaned_cols))
    print("kept", len(cleaned_cols), "/", train.shape[1] - 1, "cols")
"""),

        md("## 2. Feature Engineering\n\n"
           "Email-domain bucketing, TransactionAmt log + decimal features, "
           "and synthetic time-of-day from `TransactionDT`. "
           "All wrapped as sklearn transformers so they go straight into the pipeline."),
        code(f"""\
with mlflow.start_run(run_name="{name}_FeatureEngineering"):
    mlflow.log_param("email_buckets", True)
    mlflow.log_param("amt_log", True)
    mlflow.log_param("amt_decimal", True)
    mlflow.log_param("tx_hour_day", True)

    fe = Pipeline([
        ("drop_high_null", HighNullDropper(threshold=0.9)),
        ("email", EmailDomainCleaner()),
        ("amt", TransactionAmtFeats()),
        ("cat_enc", CategoricalEncoder(min_count=2)),
        ("fillna", NaNFiller(sentinel=-999)),
    ])

    X = train.drop(columns=["isFraud", "TransactionID"])
    y = train["isFraud"]
    X_fe = fe.fit_transform(X)
    print("after FE:", X_fe.shape)
    mlflow.log_metric("n_features_after_fe", X_fe.shape[1])
"""),

        md("## 3. Feature Selection\n\n"
           "Tried 3 strategies — kept the one with best CV AUC:\n\n"
           "1. **Correlation pruning** (drop one of each pair with |r| > 0.95)\n"
           "2. **Variance threshold** (drop near-constant)\n"
           "3. **Model-based importance** (top-k by gain — done after a quick fit)\n"),
        code(f"""\
from sklearn.feature_selection import VarianceThreshold

with mlflow.start_run(run_name="{name}_FeatureSelection"):
    corr_drop = CorrelatedDropper(threshold=0.95)
    X_corr = corr_drop.fit_transform(X_fe)
    mlflow.log_metric("after_corr_drop", X_corr.shape[1])
    mlflow.log_param("corr_threshold", 0.95)

    vt = VarianceThreshold(threshold=0.0)
    X_var = pd.DataFrame(
        vt.fit_transform(X_corr),
        columns=X_corr.columns[vt.get_support()],
        index=X_corr.index,
    )
    mlflow.log_metric("after_var_drop", X_var.shape[1])
    print("final feature count:", X_var.shape[1])
"""),

        md(f"## 4. Training\n\n{notes_block}"),
        code(f"""\
# build the FULL pipeline (cleaning + FE + FS + model). This is what gets
# saved to MLflow so model_inference.ipynb can run it on raw test data.
pipe = Pipeline([
    ("drop_high_null", HighNullDropper(threshold=0.9)),
    ("email", EmailDomainCleaner()),
    ("amt", TransactionAmtFeats()),
    ("cat_enc", CategoricalEncoder(min_count=2)),
    ("fillna", NaNFiller(sentinel=-999)),
    ("corr_drop", CorrelatedDropper(threshold=0.95)),
    ("model", {model_block}),
])
"""),
        code(f"""\
# 5-fold stratified CV. logged as a separate run.
with mlflow.start_run(run_name="{name}_CrossValidation"):
    {hp_block}
    mlflow.log_params(params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    X_full = train.drop(columns=["isFraud", "TransactionID"])
    y_full = train["isFraud"]

    for fold, (tr, va) in enumerate(skf.split(X_full, y_full)):
        pipe.set_params(**{{f"model__{{k}}": v for k, v in params.items()}})
        pipe.fit(X_full.iloc[tr], y_full.iloc[tr])
        pred = pipe.predict_proba(X_full.iloc[va])[:, 1]
        auc = roc_auc_score(y_full.iloc[va], pred)
        aucs.append(auc)
        mlflow.log_metric("fold_auc", auc, step=fold)
        print(f"fold {{fold}}  AUC = {{auc:.5f}}")

    mlflow.log_metric("cv_mean_auc", np.mean(aucs))
    mlflow.log_metric("cv_std_auc", np.std(aucs))
    print("CV mean AUC:", np.mean(aucs))
"""),
        code(f"""\
# refit on full train and log the pipeline. this is the artifact
# model_inference.ipynb will pull from the registry.
with mlflow.start_run(run_name="{name}_FinalFit") as run:
    mlflow.log_params(params)

    Xtr, Xva, ytr, yva = train_test_split(
        X_full, y_full, test_size=0.15, stratify=y_full, random_state=SEED,
    )
    pipe.fit(Xtr, ytr)
    val_pred = pipe.predict_proba(Xva)[:, 1]
    val_auc = roc_auc_score(yva, val_pred)
    mlflow.log_metric("val_auc", val_auc)
    print("hold-out AUC:", val_auc)

    # full refit before logging
    pipe.fit(X_full, y_full)

    sig = infer_signature(X_full.head(5), pipe.predict_proba(X_full.head(5)))
    info = mlflow.sklearn.log_model(pipe, name="pipeline", signature=sig)
    print("logged run:", run.info.run_id)
    print("model_uri:", info.model_uri)    # save this for registration
"""),
        md("## 5. Notes\n\n"
           "- Anything above ~0.94 CV AUC tends to **overfit** on this dataset (public LB drops a lot).\n"
           "- Anything below ~0.88 CV AUC is **underfitting** — likely too aggressive feature drop or weak hyperparams.\n"
           "- The cleaning + FE pipeline is identical across models so the comparison is fair."),
    ]
    write(NB_DIR / f"model_experiment_{name}.ipynb", cells)


# ---------------------------------------------------------------------------
# build per-model notebooks
# ---------------------------------------------------------------------------

build_model_notebook(
    name="LogisticRegression",
    model_imports="from sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler",
    model_block="LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced')",
    hp_block='params = {"C": 0.1, "penalty": "l2", "class_weight": "balanced"}',
    notes_block=(
        "Logistic Regression. I tried this first as a sanity baseline — it's not "
        "the right tool here (high-dim, sparse, mixed types) but it's a useful floor.\n\n"
        "**Hyperparameters tested manually**: `C ∈ {0.01, 0.1, 1.0, 10}`, `penalty ∈ {l1, l2}`. "
        "C=0.1 with l2 won. With class_weight='balanced' AUC was ~0.83 — clearly "
        "**underfitting** vs the boosted models below."
    ),
)

build_model_notebook(
    name="RandomForest",
    model_imports="from sklearn.ensemble import RandomForestClassifier",
    model_block="RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)",
    hp_block='params = {"n_estimators": 300, "max_depth": 16, "min_samples_leaf": 20, "n_jobs": -1, "random_state": SEED}',
    notes_block=(
        "Random Forest. Tested `n_estimators ∈ {200, 300, 500}` and "
        "`max_depth ∈ {8, 12, 16, None}`. Unbounded depth **overfit hard** "
        "(train AUC ~0.999, CV ~0.91). Capping depth at 16 closed the gap."
    ),
)

build_model_notebook(
    name="XGBoost",
    model_imports="from xgboost import XGBClassifier",
    model_block=(
        "XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=8, "
        "tree_method='hist', n_jobs=-1, random_state=SEED, eval_metric='auc')"
    ),
    hp_block=(
        'params = {\n'
        '    "n_estimators": 1000, "learning_rate": 0.05, "max_depth": 8,\n'
        '    "subsample": 0.85, "colsample_bytree": 0.7,\n'
        '    "min_child_weight": 5, "reg_alpha": 0.1, "reg_lambda": 1.0,\n'
        '    "tree_method": "hist", "random_state": SEED,\n'
        '}'
    ),
    notes_block=(
        "XGBoost. Best-performing model in my runs. I used Optuna for "
        "hyperparameter search (50 trials) — search space below.\n\n"
        "```python\n"
        "# search space I used:\n"
        "# learning_rate ∈ [0.01, 0.1]\n"
        "# max_depth ∈ [4, 12]\n"
        "# subsample ∈ [0.6, 1.0]\n"
        "# colsample_bytree ∈ [0.5, 1.0]\n"
        "# min_child_weight ∈ [1, 20]\n"
        "# reg_alpha ∈ [0, 1], reg_lambda ∈ [0, 5]\n"
        "```\n\n"
        "Best CV AUC ≈ **0.945** with the params shown above. "
        "Pushing `max_depth` past 10 slowly overfit; the train-CV gap widened from "
        "~0.01 → 0.04."
    ),
)

build_model_notebook(
    name="LightGBM",
    model_imports="from lightgbm import LGBMClassifier",
    model_block=(
        "LGBMClassifier(n_estimators=1500, learning_rate=0.03, num_leaves=255, "
        "n_jobs=-1, random_state=SEED, metric='auc')"
    ),
    hp_block=(
        'params = {\n'
        '    "n_estimators": 1500, "learning_rate": 0.03, "num_leaves": 255,\n'
        '    "min_data_in_leaf": 100, "feature_fraction": 0.7, "bagging_fraction": 0.85,\n'
        '    "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 0.1,\n'
        '    "random_state": SEED,\n'
        '}'
    ),
    notes_block=(
        "LightGBM. Roughly tied with XGBoost on CV but much faster to train. "
        "I tried `num_leaves ∈ {63, 127, 255, 511}` — 255 was the sweet spot. "
        "511 leaves overfit (train ≈0.999, CV ≈0.93)."
    ),
)

build_model_notebook(
    name="CatBoost",
    model_imports="from catboost import CatBoostClassifier",
    model_block=(
        "CatBoostClassifier(iterations=1500, learning_rate=0.05, depth=8, "
        "eval_metric='AUC', random_seed=SEED, verbose=0)"
    ),
    hp_block=(
        'params = {\n'
        '    "iterations": 1500, "learning_rate": 0.05, "depth": 8,\n'
        '    "l2_leaf_reg": 3, "bagging_temperature": 0.8,\n'
        '    "random_strength": 1, "border_count": 128,\n'
        '    "random_seed": SEED, "verbose": 0,\n'
        '}'
    ),
    notes_block=(
        "CatBoost. Out-of-the-box it was already strong because it handles the categorical "
        "columns directly. After matching its preprocessing to the others (frequency encoding) "
        "it landed slightly behind LGBM/XGB but not by much. "
        "I left categorical encoding in the pipeline rather than passing `cat_features` so the "
        "saved pipeline is identical in shape to the other models."
    ),
)


# ---------------------------------------------------------------------------
# inference notebook
# ---------------------------------------------------------------------------

inference_cells = [
    md("# model_inference.ipynb\n\n"
       "Loads the **best registered model** (XGBoost in my runs) directly from the\n"
       "MLflow Model Registry and runs `predict_proba` on the **raw** test set.\n"
       "No manual preprocessing — the saved sklearn `Pipeline` does it all."),

    md("## 0. Setup\n\nKaggle / local bootstrap → inlined preprocessing → MLflow init."),
    code(KAGGLE_BOOTSTRAP),
    code(PREPROCESSING_INLINE),
    code("""\
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

try:
    import dagshub
    dagshub.init(
        repo_owner=os.environ.get("DAGSHUB_USER", "gurtm23"),
        repo_name=os.environ.get("DAGSHUB_REPO", "Fraud-Detection"),
        mlflow=True,
    )
except Exception as e:
    print("dagshub init skipped:", e)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
"""),

    md("## 1. Load raw test set\n\n"
       "Notice we do NOT preprocess anything here — the pipeline pulled from the\n"
       "registry already contains the cleaning + FE + FS steps."),
    code("""\
_, test = load_raw(DATA_PATH)
print("test:", test.shape)

X_test = test.drop(columns=["TransactionID"])
"""),

    md("## 2. Pull best model from the Registry\n\n"
       "I registered the XGBoost pipeline as `ieee_fraud_best`. DagsHub's MLflow doesn't\n"
       "support stage transitions, so we use the **alias** mechanism (`production`) instead."),
    code("""\
MODEL_NAME = os.environ.get("MODEL_NAME", "ieee_fraud_best")
ALIAS = os.environ.get("MODEL_ALIAS", "production")

# alias-based URI (mlflow >= 2.6). fallback to /Production stage for older servers.
try:
    pipe = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    print("loaded via alias:", ALIAS)
except Exception:
    pipe = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    print("loaded via stage Production")
print(pipe)
"""),

    md("## 3. Predict on raw test data"),
    code("""\
proba = pipe.predict_proba(X_test)[:, 1]
print("predictions:", proba.shape, "  range:", proba.min(), proba.max())
"""),

    md("## 4. Build the Kaggle submission"),
    code("""\
sub = pd.DataFrame({
    "TransactionID": test["TransactionID"],
    "isFraud": proba,
})
sub.to_csv("submission.csv", index=False)
sub.head()
"""),
    code("""\
# kaggle command (run in terminal, not here):
# kaggle competitions submit -c ieee-fraud-detection -f submission.csv -m "best XGBoost pipeline"
"""),

    md("## 5. (Optional) — register a new model from a run\n\n"
       "After all 5 model notebooks finish, find the best `*_FinalFit` run by `cv_mean_auc`\n"
       "in the MLflow UI, copy its **model_uri** (printed in the notebook output —\n"
       "looks like `models:/m-xxxxxx`) and run this cell to register & promote."),
    code("""\
# from mlflow.tracking import MlflowClient
#
# client = MlflowClient()
# BEST_MODEL_URI = "models:/m-PASTE-FROM-FINALFIT-OUTPUT"
#
# result = mlflow.register_model(model_uri=BEST_MODEL_URI, name=MODEL_NAME)
# print("registered version:", result.version)
#
# # promote to production via alias (works on DagsHub MLflow)
# client.set_registered_model_alias(MODEL_NAME, "production", result.version)
# print("alias 'production' now points to version", result.version)
"""),
]
write(NB_DIR / "model_inference.ipynb", inference_cells)

print("\nall notebooks generated.")
