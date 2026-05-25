"""
Step 3: Train the Stage 1 base risk classifier (low / medium / high).

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
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.module2_classification.utils import (
    BASE_MODEL_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURE_IMPORTANCE_PATH,
    ID_COLUMNS,
    METRICS_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    RISK_CLASSES,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    encode_risk_label,
    ensure_module_dirs,
    load_feature_sets,
)
from src.utils import load_csv


def _prepare_xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    x = df[features].replace([np.inf, -np.inf], np.nan)
    y = encode_risk_label(df[TARGET_COLUMN])
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
    """Fit an imputer + multiclass XGBoost classifier on Stage 1 features."""
    x_train, y_train = _prepare_xy(train_df, features)

    imputer = SimpleImputer(strategy="median")
    x_train_imputed = imputer.fit_transform(x_train)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(RISK_CLASSES),
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train_imputed, y_train, sample_weight=sample_weight)
    return model, imputer


def evaluate_model(
    model: XGBClassifier,
    imputer: SimpleImputer,
    test_df: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    """Compute multiclass metrics on the held-out test set."""
    x_test, y_test = _prepare_xy(test_df, features)
    x_test_imputed = imputer.transform(x_test)

    y_pred = model.predict(x_test_imputed)
    y_prob = model.predict_proba(x_test_imputed)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc_ovr": roc_auc_score(
            y_test,
            y_prob,
            multi_class="ovr",
            average="macro",
        ),
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
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(RISK_CLASSES))))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(RISK_CLASSES),
        yticklabels=list(RISK_CLASSES),
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
        "Module 2 — Stage 1 Base Classifier (low / medium / high)",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Data",
        "-" * 40,
        f"Train rows: {train_size:,}",
        f"Test rows: {test_size:,}",
        f"Classes: {', '.join(RISK_CLASSES)}",
        f"Features ({len(features)}): {', '.join(features)}",
        "",
        "Test metrics",
        "-" * 40,
        f"Accuracy:         {metrics['accuracy']:.4f}",
        f"Precision (macro): {metrics['precision_macro']:.4f}",
        f"Recall (macro):    {metrics['recall_macro']:.4f}",
        f"F1 (macro):        {metrics['f1_macro']:.4f}",
        f"F1 (weighted):     {metrics['f1_weighted']:.4f}",
        f"ROC-AUC (OvR):     {metrics['roc_auc_ovr']:.4f}",
        "",
        "Classification report",
        "-" * 40,
        classification_report(
            y_test,
            y_pred,
            target_names=list(RISK_CLASSES),
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
        "classes": list(RISK_CLASSES),
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
    y_pred = model.predict(imputer.transform(x_test))

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
        f"Accuracy={metrics['accuracy']:.3f}, "
        f"F1(macro)={metrics['f1_macro']:.3f}, "
        f"ROC-AUC={metrics['roc_auc_ovr']:.3f}"
    )
    return metrics


def main() -> None:
    run_stage1_base_model()


if __name__ == "__main__":
    main()
