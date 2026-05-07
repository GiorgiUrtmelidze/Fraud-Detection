"""
Helpers shared across model_experiment_*.ipynb notebooks.

I dump everything into transformer classes so the final estimator is a real
sklearn Pipeline -> can be pickled and run on the raw test set later
(no manual preprocessing needed at inference time).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# emails are very predictive in this dataset (saw it on a public kernel)
EMAIL_COLS = ["P_emaildomain", "R_emaildomain"]

# columns I noticed have lots of NaNs but still hold signal when bucketed
HIGH_NAN_KEEP = ["dist1", "dist2", "D1", "D2", "D3", "D4", "D5"]


def reduce_mem_usage(df, verbose=False):
    """Standard kaggle trick — downcast numeric dtypes to save RAM."""
    start = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == object or str(col_type) == "category":
            continue
        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] == "int":
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    end = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(f"mem: {start:.1f} -> {end:.1f} MB  ({100*(start-end)/start:.1f}% saved)")
    return df


class EmailDomainCleaner(BaseEstimator, TransformerMixin):
    """Buckets email domains into providers (yahoo, gmail, hotmail, ...)."""

    PROVIDER_MAP = {
        "gmail": "google", "googlemail": "google", "google": "google",
        "yahoo": "yahoo", "ymail": "yahoo", "rocketmail": "yahoo", "frontier": "yahoo",
        "hotmail": "microsoft", "outlook": "microsoft", "live": "microsoft",
        "msn": "microsoft", "windowsmail": "microsoft",
        "aol": "aol",
        "att": "att", "verizon": "verizon", "comcast": "comcast",
        "icloud": "apple", "me": "apple", "mac": "apple",
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in EMAIL_COLS:
            if col not in X.columns:
                continue
            base = X[col].astype("object").fillna("missing").str.split(".").str[0]
            X[col + "_bin"] = base.map(self.PROVIDER_MAP).fillna("other")
            X[col + "_suffix"] = (
                X[col].astype("object").fillna("missing.x").str.split(".").str[-1]
            )
        return X


class TransactionAmtFeats(BaseEstimator, TransformerMixin):
    """A few engineered features around TransactionAmt that public kernels flag."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "TransactionAmt" in X.columns:
            X["TransactionAmt_log"] = np.log1p(X["TransactionAmt"])
            X["TransactionAmt_decimal"] = (
                (X["TransactionAmt"] - X["TransactionAmt"].astype(int)) * 1000
            ).astype(int)
        if "TransactionDT" in X.columns:
            # not a real timestamp — it's seconds since some unknown reference
            X["TX_hour"] = (X["TransactionDT"] // 3600) % 24
            X["TX_day"] = (X["TransactionDT"] // (3600 * 24)) % 7
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Frequency / count encoding for object columns.
    I tried label encoding first but tree-based models like LGBM/XGB blow up
    in cardinality and frequency encoding gave better CV almost everywhere.
    """

    def __init__(self, min_count=2):
        self.min_count = min_count
        self.maps_ = {}

    def fit(self, X, y=None):
        self.maps_ = {}
        for col in X.select_dtypes(include=["object", "category"]).columns:
            vc = X[col].astype("object").value_counts(dropna=False)
            # collapse rare values so test set doesn't blow up unseen-cat counts
            self.maps_[col] = vc[vc >= self.min_count].to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col, m in self.maps_.items():
            if col not in X.columns:
                continue
            X[col] = X[col].astype("object").map(m).fillna(0).astype(np.float32)
        return X


class NaNFiller(BaseEstimator, TransformerMixin):
    """Fill numeric NaNs with -999 sentinel (works well for tree models)."""

    def __init__(self, sentinel=-999):
        self.sentinel = sentinel

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        num = X.select_dtypes(include=[np.number]).columns
        X[num] = X[num].fillna(self.sentinel)
        # any leftover object cols -> 'missing'
        obj = X.select_dtypes(include=["object", "category"]).columns
        for c in obj:
            X[c] = X[c].astype("object").fillna("missing")
        return X


class HighNullDropper(BaseEstimator, TransformerMixin):
    """Drop columns whose missing-rate exceeds threshold (fitted on train)."""

    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.keep_ = None

    def fit(self, X, y=None):
        miss = X.isna().mean()
        self.keep_ = miss[miss < self.threshold].index.tolist()
        # but always keep these — they're high-nan but still useful
        for c in HIGH_NAN_KEEP:
            if c in X.columns and c not in self.keep_:
                self.keep_.append(c)
        return self

    def transform(self, X):
        cols = [c for c in self.keep_ if c in X.columns]
        return X[cols].copy()


class CorrelatedDropper(BaseEstimator, TransformerMixin):
    """Drop one of each pair with |corr| > threshold."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.drop_ = []

    def fit(self, X, y=None):
        # only check numerics — string columns don't matter here
        num = X.select_dtypes(include=[np.number])
        # subsample rows to make this feasible on the full train set
        if len(num) > 50_000:
            num = num.sample(50_000, random_state=0)
        corr = num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.drop_ = [c for c in upper.columns if any(upper[c] > self.threshold)]
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.drop_ if c in X.columns])


def load_raw(path="../data"):
    """
    Load + merge transaction & identity tables on TransactionID.
    Path defaults to ../data/ to match repo layout. On Kaggle pass
    '/kaggle/input/ieee-fraud-detection'.
    """
    train_tx = pd.read_csv(f"{path}/train_transaction.csv")
    train_id = pd.read_csv(f"{path}/train_identity.csv")
    test_tx = pd.read_csv(f"{path}/test_transaction.csv")
    test_id = pd.read_csv(f"{path}/test_identity.csv")

    train = train_tx.merge(train_id, how="left", on="TransactionID")
    test = test_tx.merge(test_id, how="left", on="TransactionID")
    # kaggle ships test identity columns prefixed id-XX (dash) instead of id_XX
    test.columns = [c.replace("id-", "id_") for c in test.columns]

    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    return train, test
