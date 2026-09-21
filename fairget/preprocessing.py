"""Preprocessing for binary tabular classification.

Numeric features are min-max scaled and categorical features are one-hot
encoded.  The fitted mappings are retained and reused for evaluation and
prediction.  Target and sensitive-group mappings are included in the model
report; callers can set them explicitly with ``positive_label`` and
``privileged_group``.
"""

import numpy as np
import pandas as pd


class Preprocessor:
    """Fits encoders on training data and applies them consistently at predict
    time. Stateful so that predict() sees the same encoding as fit()."""

    def __init__(self, target, sensitive, positive_label=None,
                 privileged_group=None, drop_columns=None):
        if isinstance(sensitive, (list, tuple)):
            if len(sensitive) != 1:
                raise ValueError(
                    "fairget v1 supports exactly one binary sensitive column. "
                    f"Got {len(sensitive)}: {sensitive}. Pick one, or combine "
                    "them into a single binary column before calling fit()."
                )
            sensitive = sensitive[0]
        self.target = target
        self.sensitive = sensitive
        self.positive_label = positive_label
        self.privileged_group = privileged_group
        if isinstance(drop_columns, str):
            drop_columns = [drop_columns]
        self.drop_columns = list(dict.fromkeys(drop_columns or []))

        # Learned during fit
        self.feature_columns_ = None      # ordered list of engineered columns
        self.categorical_maps_ = {}       # col -> list of categories (one-hot order)
        self.numeric_ranges_ = {}         # col -> (min, max) for scaling
        self.target_positive_ = None
        self.sensitive_privileged_ = None
        self.target_classes_ = None
        self.sensitive_groups_ = None

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _is_numeric(series):
        return pd.api.types.is_numeric_dtype(series)

    def _binarize_target(self, series, fitting):
        if series.isna().any():
            raise ValueError(f"Target column '{self.target}' contains missing values.")
        vals = series.dropna().unique()
        if fitting and len(vals) != 2:
            raise ValueError(
                f"Target column '{self.target}' must have exactly 2 classes for "
                f"binary classification; found {len(vals)}: {sorted(vals)[:10]}. "
                "fairget v1 is binary-target only."
            )
        if fitting:
            self.target_classes_ = tuple(vals.tolist())
            if self.positive_label is not None:
                pos = self.positive_label
                if pos not in vals:
                    raise ValueError(
                        f"positive_label={pos!r} not found in target values {sorted(vals)}."
                    )
            else:
                # Default: the larger of the two sorted values (e.g. 1 over 0,
                # '>50K' over '<=50K' lexically). Reported back to the user.
                pos = sorted(vals, key=lambda v: str(v))[-1]
            self.target_positive_ = pos
        else:
            unknown = [v for v in vals if v not in self.target_classes_]
            if unknown:
                raise ValueError(
                    f"Target column '{self.target}' contains labels not seen during "
                    f"fit: {sorted(unknown, key=str)}."
                )
        return (series == self.target_positive_).astype(int).to_numpy()

    def _binarize_sensitive(self, series, fitting):
        if series.isna().any():
            raise ValueError(
                f"Sensitive column '{self.sensitive}' contains missing values."
            )
        vals = series.dropna().unique()
        if fitting and len(vals) != 2:
            raise ValueError(
                f"Sensitive column '{self.sensitive}' must be binary in fairget v1; "
                f"found {len(vals)} categories: {sorted(vals, key=str)[:10]}. "
                "Binarize it first (e.g. group into two categories, or pick two "
                "groups to compare)."
            )
        if fitting:
            self.sensitive_groups_ = tuple(vals.tolist())
            if self.privileged_group is not None:
                priv = self.privileged_group
                if priv not in vals:
                    raise ValueError(
                        f"privileged_group={priv!r} not in sensitive values {sorted(vals, key=str)}."
                    )
            else:
                priv = sorted(vals, key=lambda v: str(v))[-1]
            self.sensitive_privileged_ = priv
        else:
            unknown = [v for v in vals if v not in self.sensitive_groups_]
            if unknown:
                raise ValueError(
                    f"Sensitive column '{self.sensitive}' contains groups not seen "
                    f"during fit: {sorted(unknown, key=str)}."
                )
        # group 1 = privileged, group 0 = other
        return (series == self.sensitive_privileged_).astype(int).to_numpy()

    def _engineer_features(self, df, fitting):
        feature_df = df.drop(
            columns=[self.target, self.sensitive] + self.drop_columns,
            errors="ignore",
        )
        if fitting:
            self.feature_columns_ = []
            self.categorical_maps_ = {}
            self.numeric_ranges_ = {}

        pieces = []
        # Determine, at fit time, which columns are numeric vs categorical.
        if fitting:
            self._raw_feature_cols_ = list(feature_df.columns)

        for col in self._raw_feature_cols_:
            if col not in feature_df.columns:
                raise ValueError(f"Column '{col}' seen at fit time is missing at predict time.")
            series = feature_df[col]
            if series.isna().any():
                raise ValueError(f"Feature column '{col}' contains missing values.")
            if fitting:
                is_num = self._is_numeric(series)
            else:
                is_num = col in self.numeric_ranges_

            if is_num:
                if fitting:
                    lo, hi = float(series.min()), float(series.max())
                    self.numeric_ranges_[col] = (lo, hi)
                lo, hi = self.numeric_ranges_[col]
                rng = (hi - lo) if hi > lo else 1.0
                scaled = (series.to_numpy(dtype=float) - lo) / rng
                scaled = np.clip(scaled, 0.0, 1.0)
                pieces.append(scaled.reshape(-1, 1))
                if fitting:
                    self.feature_columns_.append(col)
            else:
                if fitting:
                    cats = sorted(series.dropna().astype(str).unique())
                    self.categorical_maps_[col] = cats
                cats = self.categorical_maps_[col]
                s = series.astype(str)
                unknown = sorted(set(s.unique()) - set(cats))
                if unknown:
                    raise ValueError(
                        f"Categorical column '{col}' contains categories not seen "
                        f"during fit: {unknown[:10]}."
                    )
                onehot = np.zeros((len(s), len(cats)), dtype=float)
                for j, cat in enumerate(cats):
                    onehot[(s == cat).to_numpy(), j] = 1.0
                pieces.append(onehot)
                if fitting:
                    self.feature_columns_.extend(f"{col}={cat}" for cat in cats)

        if not pieces:
            raise ValueError("No usable feature columns after preprocessing.")
        return np.hstack(pieces).astype(np.float32)

    # --- public API ----------------------------------------------------------

    def fit_transform(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("fit expects a pandas DataFrame.")
        missing = [c for c in (self.target, self.sensitive) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}.")
        unknown_drops = [c for c in self.drop_columns if c not in df.columns]
        if unknown_drops:
            raise ValueError(f"drop_columns contains unknown columns: {unknown_drops}.")
        X = self._engineer_features(df, fitting=True)
        y = self._binarize_target(df[self.target], fitting=True)
        s = self._binarize_sensitive(df[self.sensitive], fitting=True)
        return X, y, s

    def transform(self, df):
        if self.feature_columns_ is None:
            raise RuntimeError("Preprocessor.transform called before fit_transform.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("transform expects a pandas DataFrame.")
        X = self._engineer_features(df, fitting=False)
        # target/sensitive may be absent at predict time
        y = (self._binarize_target(df[self.target], fitting=False)
             if self.target in df.columns else None)
        s = (self._binarize_sensitive(df[self.sensitive], fitting=False)
             if self.sensitive in df.columns else None)
        return X, y, s

    @property
    def n_features(self):
        return len(self.feature_columns_)

    def summary(self):
        """Human-readable description of every preprocessing decision made."""
        negative_class = next(
            value for value in self.target_classes_
            if value != self.target_positive_
        )
        nonprivileged_group = next(
            value for value in self.sensitive_groups_
            if value != self.sensitive_privileged_
        )
        return {
            "n_features_engineered": self.n_features,
            "target_column": self.target,
            "target_positive_class": self.target_positive_,
            "target_classes": list(self.target_classes_),
            "target_class_encoding": {
                "0": negative_class,
                "1": self.target_positive_,
            },
            "sensitive_column": self.sensitive,
            "sensitive_privileged_group": self.sensitive_privileged_,
            "sensitive_groups": list(self.sensitive_groups_),
            "sensitive_group_encoding": {
                "0": nonprivileged_group,
                "1": self.sensitive_privileged_,
            },
            "numeric_columns": list(self.numeric_ranges_.keys()),
            "categorical_columns": list(self.categorical_maps_.keys()),
            "dropped_columns": self.drop_columns,
        }
