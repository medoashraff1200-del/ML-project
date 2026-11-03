from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ==========================================
# Feature definitions
# ==========================================

TARGET_REGRESSION = "ADR"
TARGET_CLASSIFICATION = "IsCanceled"

# Features assumed to be available when the booking is initially made.
# Excludes fields that may reflect later events or final outcomes.
BOOKING_TIME_FEATURES = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "DepositType",
    "Agent",
    "Company",
    "CustomerType",
]

# Clustering features: no target labels or final booking outcomes.
CLUSTERING_FEATURES = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "IsRepeatedGuest",
    "ADR",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]


# ==========================================
# Preprocessing builders
# ==========================================

def get_feature_types(
    df: pd.DataFrame,
    features: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Return numerical and categorical feature names."""
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    numerical_features = df[list(features)].select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = [
        column for column in features
        if column not in numerical_features
    ]

    return numerical_features, categorical_features


def build_preprocessor(
    df: pd.DataFrame,
    features: Sequence[str],
) -> ColumnTransformer:
    """Build a preprocessing transformer for the selected columns."""
    numerical_features, categorical_features = get_feature_types(
        df,
        features,
    )

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    transformers = []

    if numerical_features:
        transformers.append(
            ("numerical", numerical_pipeline, numerical_features)
        )

    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ==========================================
# Task data preparation
# ==========================================

def prepare_task_data(
    df: pd.DataFrame,
    task: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split data for regression or classification.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    if task == "regression":
        target = TARGET_REGRESSION
        features = BOOKING_TIME_FEATURES
    elif task == "classification":
        target = TARGET_CLASSIFICATION
        features = BOOKING_TIME_FEATURES
    else:
        raise ValueError(
            "task must be either 'regression' or 'classification'"
        )

    required_columns = set(features) | {target}
    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    task_df = df[list(features) + [target]].copy()

    # Remove rows with missing target only; feature missing values
    # are handled inside the preprocessing pipeline.
    task_df = task_df.dropna(subset=[target])

    X = task_df[list(features)]
    y = task_df[target]

    stratify = y if task == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    preprocessor = build_preprocessor(X_train, features)

    return X_train, X_test, y_train, y_test, preprocessor


# ==========================================
# Clustering data preparation
# ==========================================

def prepare_clustering_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, ColumnTransformer]:
    """Return clustering features and their preprocessing transformer."""
    missing = sorted(set(CLUSTERING_FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Missing clustering columns: {missing}")

    X = df[CLUSTERING_FEATURES].copy()
    preprocessor = build_preprocessor(X, CLUSTERING_FEATURES)

    return X, preprocessor