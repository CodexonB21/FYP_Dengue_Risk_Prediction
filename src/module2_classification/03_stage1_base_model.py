"""
Step 3: Train the Stage 1 base outbreak classifier (binary).

Uses epidemiological + temporal features only, saves the model under `models/`,
and writes evaluation plots/metrics to `results/`.

Run:
    python -m src.module2_classification.03_stage1_base_model
"""

from __future__ import annotations

import importlib
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.module2_classification.utils import (
    BASE_MODEL_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURE_IMPORTANCE_PATH,
    ID_COLUMNS,
    METRICS_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    ensure_module_dirs,
    load_feature_sets,
)
from src.utils import load_csv


def _prepare_xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    x = df[features].replace([np.inf, -np.inf], np.nan)
    y = df[TARGET_COLUMN]
    return x, y


def _ensure_train_test_data() -> None:
    if TRAIN_DATA_PATH.exists() and TEST_DATA_PATH.exists():
        return
    step2 = importlib.import_module("src.module2_classification.02_feature_selection")
    step2.run_feature_selection()


def train_base_classifier(
    train_df: pd.DataFrame,
    features: list[str],
) -> tuple[XGBClassifier, SimpleImputer]:
    """Fit an imputer + binary XGBoost classifier on Stage 1 features."""
    x_train, y_train = _prepare_xy(train_df, features)

    imputer = SimpleImputer(strategy="median")
    x_train_imputed = imputer.fit_transform(x_train)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train_imputed, y_train)
    return model, imputer


def evaluate_model(
    model: XGBClassifier,
    imputer: SimpleImputer,
    test_df: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    """Compute binary classification metrics on the held-out test set."""
    x_test, y_test = _prepare_xy(test_df, features)
    x_test_imputed = imputer.transform(x_test)

    y_pred = model.predict(x_test_imputed)
    y_prob = model.predict_proba(x_test_imputed)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def plot_feature_importance(model: XGBClassifier, features: list[str], path) -> None:
    importance = pd.Series(model.feature_importances_, index=features).sort_values()
    plt.figure(figsize=(8, 5))
    importance.plot(kind="barh", color="#2563eb")
    plt.title("Stage 1 Base Classifier — Feature Importance")
    plt.xlabel("XGBoost importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Outbreak"],
        yticklabels=["Normal", "Outbreak"],
    )
    plt.title("Stage 1 Confusion Matrix (Test Set)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_metrics(
    path,
    metrics: dict[str, float],
    features: list[str],
    train_size: int,
    test_size: int,
    y_test,
    y_pred,
) -> None:
    lines = [
        "Module 2 — Stage 1 Base Classifier (binary outbreak)",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Data",
        "-" * 40,
        f"Train rows: {train_size:,}",
        f"Test rows: {test_size:,}",
        f"Features ({len(features)}): {', '.join(features)}",
        "",
        "Test metrics",
        "-" * 40,
        f"Accuracy:  {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall:    {metrics['recall']:.4f}",
        f"F1:        {metrics['f1']:.4f}",
        f"ROC-AUC:   {metrics['roc_auc']:.4f}",
        "",
        "Classification report",
        "-" * 40,
        classification_report(
            y_test,
            y_pred,
            target_names=["Normal", "Outbreak"],
            zero_division=0,
        ),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_model_bundle(model, imputer, features: list[str], path) -> None:
    bundle = {
        "model": model,
        "imputer": imputer,
        "features": features,
        "target": TARGET_COLUMN,
        "positive_class": 1,
        "id_columns": list(ID_COLUMNS),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(bundle, fh)


def run_stage1_base_model() -> dict[str, float]:
    ensure_module_dirs()
    _ensure_train_test_data()

    feature_sets = load_feature_sets()
    features = feature_sets["stage1_features"]

    train_df = load_csv(TRAIN_DATA_PATH, parse_dates=["week_start_date"])
    test_df = load_csv(TEST_DATA_PATH, parse_dates=["week_start_date"])

    model, imputer = train_base_classifier(train_df, features)
    metrics = evaluate_model(model, imputer, test_df, features)

    x_test, y_test = _prepare_xy(test_df, features)
    y_pred = model.predict(imputer.transform(x_test.replace([np.inf, -np.inf], np.nan)))

    save_model_bundle(model, imputer, features, BASE_MODEL_PATH)
    plot_feature_importance(model, features, FEATURE_IMPORTANCE_PATH)
    plot_confusion_matrix(y_test, y_pred, CONFUSION_MATRIX_PATH)
    write_metrics(
        METRICS_PATH,
        metrics,
        features,
        len(train_df),
        len(test_df),
        y_test,
        y_pred,
    )

    print(f"Saved model: {BASE_MODEL_PATH}")
    print(f"Saved results under {RESULTS_DIR}:")
    print(f"  {FEATURE_IMPORTANCE_PATH.name}")
    print(f"  {CONFUSION_MATRIX_PATH.name}")
    print(f"  {METRICS_PATH.name}")
    print(
        "Test metrics — "
        f"F1={metrics['f1']:.3f}, "
        f"Recall={metrics['recall']:.3f}, "
        f"ROC-AUC={metrics['roc_auc']:.3f}"
    )
    return metrics


def main() -> None:
    run_stage1_base_model()


if __name__ == "__main__":
    main()
