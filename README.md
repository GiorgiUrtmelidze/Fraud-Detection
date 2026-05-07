# IEEE-CIS Fraud Detection


[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) is a binary classification problem where, based on data from UnitedHealth and IEEE, we have to predict whether a given transaction is fraudulent (`isFraud=1`).

The data is split across two tables:
* **`transaction`** (financial information): includes columns like `TransactionAmt`, `card1..card6`, `addr1/addr2`, `dist1/dist2`, `D1..D15`, `V1..V339`, `C1..C14` and others.
* **`identity`** (device and browser information): includes `id_01..id_38`, `DeviceType` and `DeviceInfo`.

The two tables are joined on `TransactionID`. Since `identity` only covers a subset of transactions, the join is a `LEFT JOIN`.

The evaluation metric is **ROC AUC**. The target is heavily imbalanced (fraud rate ~3.5%).

## Approach

1. **Cleaning** — drop columns whose `NaN` rate exceeds 90%. A few high-NaN-but-informative columns are kept (`dist1/dist2`, `D1..D5`). On the real run, 11 of 432 columns were dropped → 421 retained.
2. **Feature Engineering** — bucket email domains (gmail/google → `google` etc.), compute `log(TransactionAmt)` and the decimal portion of `TransactionAmt`, derive hour-of-day / day-of-week from `TransactionDT`. After FE the feature count is **429**.
3. **Categorical encoding** — frequency encoding with rare-value collapsing. It outperformed label encoding on CV for tree-based models.
4. **Feature selection** — correlation pruning (|r|>0.95) applied **inside** the sklearn `Pipeline` so the same filter runs on the test set without leakage.
5. **Training** — 3-fold StratifiedKFold (5-fold on a CPU with 590k rows is too slow for a homework iteration loop), then a refit on the full train set.
6. **Automation** — every step is wrapped in an `sklearn` `Pipeline`, so raw data goes straight to the model with no manual preprocessing. `model_inference.ipynb` pulls the pipeline from the MLflow Model Registry and runs it on the raw test set.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── preprocessing.py          # custom sklearn transformers
├── scripts/
│   ├── build_notebooks.py        # notebook generator (utility)
│   └── run_real.py               # local training driver (LR + LightGBM)
└── notebooks/
    ├── model_experiment_LogisticRegression.ipynb
    ├── model_experiment_RandomForest.ipynb
    ├── model_experiment_XGBoost.ipynb
    ├── model_experiment_LightGBM.ipynb
    ├── model_experiment_CatBoost.ipynb
    └── model_inference.ipynb
```

## Training — actual results

Because of the local CPU constraint, I only ran 2 models for real on the full 590k × 434 dataset. All five model notebooks are ready to run on Kaggle (more compute available there), but only two have backing real numbers.

| Model | CV AUC (3-fold) | std | Train AUC | Val AUC | Train-Val gap | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression (l2, C=0.1, balanced) | **0.82185** | 0.0033 | 0.8223 | 0.8232 | -0.0008 | **underfitting** — linear model can't capture feature interactions; train ≈ val |
| **LightGBM** (n_est=800, num_leaves=127, lr=0.05) | **0.96621** | 0.0011 | 0.9987 | 0.9701 | 0.0286 | **mild overfitting** — train near 1.0, val solid |

### Overfitting / underfitting analysis

- **Logistic Regression** — train AUC ≈ val AUC (~0.82, gap ~0). Classic underfitting: model capacity is insufficient and even train doesn't improve. A linear model can't see the non-linear interactions across `card`, `addr`, `D*`, `V*`.
- **LightGBM** — train AUC ≈ 0.999, val AUC = 0.970, gap = 0.029. Textbook "near-perfect train, strong val" — a bit of overfitting, but holdout doesn't collapse. The Kaggle Public LB came in ~5% below val AUC, which reflects the well-known distribution shift between train and test in this competition (the public/private split is also drift-y).

### Hyperparameter tuning

LightGBM was tuned by hand (Optuna across 590k rows on local CPU would have been >2 hours per study — outside the homework budget). The chosen config:
```python
{
    "n_estimators": 800, "learning_rate": 0.05, "num_leaves": 127,
    "min_data_in_leaf": 100, "feature_fraction": 0.7, "bagging_fraction": 0.85,
    "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42,
}
```
`num_leaves` > 127 made the gap blow up; < 63 underfit.

### Final-model rationale

LightGBM beat LR by ~14.4 AUC points (0.966 vs 0.822) and produced the best Kaggle Public LB (0.932). The 0.029 train-vs-val gap is acceptable. A heavier-regularized XGBoost might claw back a bit more, but within the local CPU budget LightGBM was the best quality-vs-time trade-off. It is registered in the Model Registry as **`ieee_fraud_best`**, alias `production`.

## MLflow Tracking

### MLflow URL

[https://dagshub.com/gurtm23/Fraud-Detection.mlflow](https://dagshub.com/gurtm23/Fraud-Detection.mlflow)

### Experiment / run structure

| Experiment | Runs |
| --- | --- |
| `LogisticRegression_Training` | `LogisticRegression_Cleaning`, `LogisticRegression_FeatureEngineering`, `LogisticRegression_FeatureSelection`, `LogisticRegression_CrossValidation`, `LogisticRegression_FinalFit` |
| `LightGBM_Training` | `LightGBM_Cleaning`, `LightGBM_FeatureEngineering`, `LightGBM_FeatureSelection`, `LightGBM_CrossValidation`, `LightGBM_FinalFit` |

### Logged metrics

- **Cleaning run**: `nan_threshold`, `n_cols_before`, `n_cols_after`
- **FeatureEngineering run**: `email_buckets`, `amt_log`, `amt_decimal`, `tx_hour_day` flags; `n_features_after_fe`
- **FeatureSelection run**: `corr_threshold`
- **CrossValidation run**: all model hyperparameters, `n_folds`, per-fold `fold_auc` (`step=fold`), `cv_mean_auc`, `cv_std_auc`
- **FinalFit run**: hyperparameters, `cv_mean_auc`, `train_auc`, `val_auc`, `train_val_gap`, `fit_status` tag (`underfitting` / `balanced` / `overfitting`), and the full sklearn `Pipeline` artifact with a `signature` (so it can be served from the registry directly).

### Best-model summary

```
Model:        LightGBM (registered: ieee_fraud_best, alias=production, v1)
CV AUC:       0.96621  (std 0.0011)  — 3-fold StratifiedKFold
Train AUC:    0.99872  (full train, 15% hold-out)
Val AUC:      0.97011
Kaggle Public LB:   0.932448
Kaggle Private LB:  0.894971
```

## How to run

```bash
# 1. dependencies
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. data
mkdir -p data && kaggle competitions download -c ieee-fraud-detection -p data
unzip data/ieee-fraud-detection.zip -d data/ && rm data/*.zip

# 3. dagshub credentials
export MLFLOW_TRACKING_USERNAME=gurtm23
export MLFLOW_TRACKING_PASSWORD=<dagshub-token>

# 4. training (LR + LightGBM)
.venv/bin/python scripts/run_real.py

# 5. submission file is written to ./submission.csv
kaggle competitions submit -c ieee-fraud-detection -f submission.csv -m "lgbm pipeline"
```

For the per-model notebooks (RF / XGB / CatBoost), launch them locally
```bash
jupyter lab notebooks/
```
or upload them to Kaggle — they're self-contained (preprocessing inlined, `kaggle_secrets` bootstrap). Just upload → Add Data: IEEE-CIS → set the two MLflow secrets → Run All.
