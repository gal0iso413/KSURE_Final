"""
Step 9: Final Model Fit, Prediction Export, and Quick Interpretation
===================================================================

Purpose
- Bridge Step 8 tuned parameters to grading by training final model(s),
  exporting per-year multiclass probabilities for all rows, and producing
  a lean, business-ready sanity report.

Outputs
- result/predictions/yearly_multiclass_proba.csv
- result/step9_post/quick_report.md
- result/step9_post/plots/confusion_matrix_*.png (OOT if available, DEV fallback)
- result/step9_post/plots/feature_importance_*.png
- result/step9_post/models/* (JSON boosters)
- result/step9_post/manifest.json

Notes
- Uses the same feature selection and exclusions as Step 8 scripts.
- Supports both architectures:
  - individual: one model per target (risk_year1..4)
  - unified: one model with task_id feature
- Optional SHAP-like attributions via XGBoost pred_contribs on a small sample
  can be added later if needed; currently disabled to keep runs fast.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# Visualization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------


@dataclass
class Step9Config:
    dataset_path: str = "dataset/credit_risk_dataset_step4.csv"
    predictions_path: str = "result/predictions/yearly_multiclass_proba.csv"
    output_dir: str = "result/step9_post"
    best_params_dir: str = "result/step8_optuna"
    arch: str = "individual"  # individual|unified
    oot_frac: float = 0.2  # last fraction by 보험청약일자 for quick OOT sanity check
    shap_sample_size: int = 0  # 0 disables; else sample up to N rows per target for contribs
    date_column: str = "보험청약일자"
    key_column: str = "청약번호"
    random_state: int = 42


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_step4_data(dataset_path: str, date_column: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    if date_column in df.columns:
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception:
            pass
    return df


def get_feature_and_target_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    exclude_cols = [
        "사업자등록번호",
        "대상자명",
        "대상자등록이력일시",
        "대상자기본주소",
        "청약번호",
        "보험청약일자",
        "청약상태코드",
        "수출자대상자번호",
        "특별출연협약코드",
        "업종코드1",
    ]
    target_cols = [c for c in df.columns if c.startswith("risk_year")]
    feature_cols = [c for c in df.columns if c not in exclude_cols + target_cols]
    return feature_cols, target_cols, exclude_cols


def detect_best_params(cfg: Step9Config) -> Tuple[str, Dict[str, dict], Dict[str, any]]:
    """
    Returns (approach, best_params_by_target_or_unified, meta)
    - approach: 'individual' or 'unified'
    - best_params_by_target_or_unified:
        individual -> {"risk_year1": {..}, ..., "risk_year4": {..}}
        unified    -> {"unified": {..}}
    - meta: additional info from the best-params file
    """
    candidates_ordered: List[Tuple[str, str]] = []
    if cfg.arch == "individual":
        candidates_ordered = [
            ("individual", os.path.join(cfg.best_params_dir, "step8_best_params_individual.json")),
            ("individual", os.path.join(cfg.best_params_dir, "step8_best_params_individual_gpu.json")),
        ]
    elif cfg.arch == "unified":
        candidates_ordered = [
            ("unified", os.path.join(cfg.best_params_dir, "step8_best_params_unified.json")),
            ("unified", os.path.join(cfg.best_params_dir, "step8_best_params_unified_gpu.json")),
        ]
    else:
        candidates_ordered = [
            ("individual", os.path.join(cfg.best_params_dir, "step8_best_params_individual.json")),
            ("unified", os.path.join(cfg.best_params_dir, "step8_best_params_unified.json")),
            ("individual", os.path.join(cfg.best_params_dir, "step8_best_params_individual_gpu.json")),
            ("unified", os.path.join(cfg.best_params_dir, "step8_best_params_unified_gpu.json")),
        ]

    for approach, path in candidates_ordered:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = {k: v for k, v in data.items() if k not in {"best_params", "best_params_by_target"}}
            if approach == "individual" and "best_params_by_target" in data:
                return "individual", data["best_params_by_target"], meta
            if approach == "unified" and "best_params" in data:
                return "unified", {"unified": data["best_params"]}, meta
            # Fallback if schema differs
            if "best_params_by_target" in data:
                return "individual", data["best_params_by_target"], meta
            if "best_params" in data:
                return "unified", {"unified": data["best_params"]}, meta

    raise FileNotFoundError(
        "Could not find Step 8 best-params file. Expected one of: "
        "step8_best_params_individual(.json|_gpu.json) or step8_best_params_unified(.json|_gpu.json) in result/step8_optuna"
    )


def build_oot_mask(df: pd.DataFrame, date_column: str, frac: float) -> np.ndarray:
    n = len(df)
    if n == 0 or frac <= 0.0:
        return np.zeros(n, dtype=bool)
    if date_column in df.columns and np.issubdtype(df[date_column].dtype, np.datetime64):
        order = np.argsort(df[date_column].values)
    else:
        order = np.arange(n)
    cut = int(n * (1.0 - max(0.0, min(1.0, frac))))
    oot_idx = set(order[cut:])
    mask = np.array([i in oot_idx for i in range(n)], dtype=bool)
    return mask


def compute_metrics_per_target(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    mask = ~np.isnan(y_true)
    y_true = y_true[mask].astype(int)
    proba = proba[mask]
    if len(y_true) == 0:
        return {"f1_macro": 0.0, "balanced_accuracy": 0.0, "high_risk_recall": 0.0, "n_eval": 0}
    y_pred = np.argmax(proba, axis=1)
    try:
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        y_true_hr = (y_true >= 2).astype(int)
        y_pred_hr = (y_pred >= 2).astype(int)
        hr_recall = recall_score(y_true_hr, y_pred_hr, zero_division=0)
    except Exception:
        f1m, bal_acc, hr_recall = 0.0, 0.0, 0.0
    out.update({
        "f1_macro": float(f1m),
        "balanced_accuracy": float(bal_acc),
        "high_risk_recall": float(hr_recall),
        "n_eval": int(len(y_true)),
    })
    return out


def expand_proba_to_4_classes(model_classes: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """
    Ensures probability matrix has 4 columns for classes {0,1,2,3}.
    """
    num_rows = proba.shape[0]
    full = np.zeros((num_rows, 4), dtype=float)
    for idx, cls in enumerate(model_classes):
        if 0 <= int(cls) <= 3:
            full[:, int(cls)] = proba[:, idx]
    # Normalize safety (should already sum to 1 across present classes)
    row_sum = full.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        full = np.where(row_sum > 0, full / row_sum, full)
    return full


def xgb_classifier_from_params(params: Dict[str, any], random_state: int) -> xgb.XGBClassifier:
    base = dict(
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss",
    )
    # Remove None-valued keys
    base = {k: v for k, v in base.items() if v is not None}
    model = xgb.XGBClassifier(**base, **params)
    return model


def save_model_booster(model: xgb.XGBClassifier, path: str) -> None:
    try:
        booster = model.get_booster()
        booster.save_model(path)
    except Exception:
        # Fallback to sklearn pickle if booster not available
        import joblib

        joblib.dump(model, path + ".pkl")


def feature_importance_gain(model: xgb.XGBClassifier, feature_names: List[str]) -> pd.DataFrame:
    try:
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
    except Exception:
        score = {}
    # Normalize keys to feature names when possible
    # XGBoost returns feature keys like 'f0', 'f1' if not trained with named columns.
    # Here we trained with pandas DataFrame, so names should be preserved.
    rows = [(k, float(v)) for k, v in score.items() if k in feature_names]
    df = pd.DataFrame(rows, columns=["feature", "gain"]).sort_values("gain", ascending=False)
    return df


def try_configure_korean_font() -> None:
    try:
        # Prefer common Korean fonts on Windows and Linux
        for f in ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "AppleGothic"]:
            plt.rcParams["font.family"] = f
            break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def plot_confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray, path: str, title: str) -> None:
    if len(y_true) == 0:
        return
    labels = [0, 1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_feature_importance(fi_df: pd.DataFrame, path: str, title: str, top_n: int = 20) -> None:
    if fi_df is None or fi_df.empty:
        return
    df = fi_df.head(top_n)
    plt.figure(figsize=(7.5, 5.0))
    sns.barplot(data=df, x="gain", y="feature", color="#4C78A8")
    plt.xlabel("Gain (importance)")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Step 9: Final Model Fit, Prediction Export, Quick Interpretation")
    parser.add_argument("--dataset_path", type=str, default="dataset/credit_risk_dataset_step4.csv")
    parser.add_argument("--predictions_path", type=str, default="result/predictions/yearly_multiclass_proba.csv")
    parser.add_argument("--output_dir", type=str, default="result/step9_post")
    parser.add_argument("--best_params_dir", type=str, default="result/step8_optuna")
    parser.add_argument("--arch", type=str, default="individual", choices=["individual", "unified"])
    parser.add_argument("--oot_frac", type=float, default=0.2)
    parser.add_argument("--shap_sample_size", type=int, default=0)
    parser.add_argument("--date_column", type=str, default="보험청약일자")
    parser.add_argument("--key_column", type=str, default="청약번호")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    cfg = Step9Config(
        dataset_path=args.dataset_path,
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
        best_params_dir=args.best_params_dir,
        arch=args.arch,
        oot_frac=float(args.oot_frac),
        shap_sample_size=int(args.shap_sample_size),
        date_column=args.date_column,
        key_column=args.key_column,
        random_state=int(args.random_state),
    )

    ensure_dir(os.path.dirname(cfg.predictions_path))
    ensure_dir(cfg.output_dir)
    models_dir = os.path.join(cfg.output_dir, "models")
    ensure_dir(models_dir)
    plots_dir = os.path.join(cfg.output_dir, "plots")
    ensure_dir(plots_dir)
    try_configure_korean_font()

    # Load data and define features/targets
    df = load_step4_data(cfg.dataset_path, cfg.date_column)
    feature_cols, target_cols, exclude_cols = get_feature_and_target_columns(df)
    X_all = df[feature_cols].copy()

    # Detect best params and potentially architecture
    approach, best_params_map, meta = detect_best_params(cfg)

    # Build OOT mask once (global chronological split)
    oot_mask = build_oot_mask(df, cfg.date_column, cfg.oot_frac)

    # Prepare predictions frame with keys
    output_cols = [cfg.key_column]
    if cfg.key_column not in df.columns:
        raise ValueError(f"Key column {cfg.key_column} not found in dataset.")
    preds_df = df[[cfg.key_column]].copy()
    if cfg.date_column in df.columns:
        preds_df[cfg.date_column] = df[cfg.date_column]

    metrics_summary: Dict[str, Dict[str, float]] = {}
    model_manifest: Dict[str, any] = {
        "execution_date": datetime.now().isoformat(),
        "approach": approach,
        "use_gpu": False,
        "dataset_path": cfg.dataset_path,
        "predictions_path": cfg.predictions_path,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "exclude_cols": exclude_cols,
        "meta_from_step8": meta,
    }

    if approach == "individual":
        # Train one model per target and predict all rows
        for _, target in enumerate(sorted(target_cols)):
            # Guard to only accept expected targets risk_year1..4
            if not target.startswith("risk_year"):
                continue
            params = best_params_map.get(target, best_params_map.get(str(target), {}))
            model = xgb_classifier_from_params(params, cfg.random_state)

            mask = ~pd.isna(df[target])
            X_train = X_all.loc[mask]
            y_train = df[target].loc[mask].astype(int).values
            if len(y_train) == 0:
                # No labels; skip training and fill uniform probabilities
                uniform = np.full((len(df), 4), 0.25, dtype=float)
                try:
                    year_idx = int(str(target).replace("risk_year", ""))
                except Exception:
                    year_idx = 1
                for cls in range(4):
                    preds_df[f"proba_y{year_idx}_{cls}"] = uniform[:, cls]
                metrics_summary[target] = {"f1_macro": 0.0, "balanced_accuracy": 0.0, "high_risk_recall": 0.0, "n_eval": 0, "split": "none"}
                continue

            sample_weights = compute_sample_weight("balanced", y_train)
            model.fit(X_train, y_train, sample_weight=sample_weights)

            # Persist booster
            save_model_booster(model, os.path.join(models_dir, f"individual_{target}.json"))

            # Predict for all rows (production predictions)
            proba_all = model.predict_proba(X_all)
            proba_full = expand_proba_to_4_classes(getattr(model, "classes_", np.array([0, 1, 2, 3])), proba_all)
            # Map target name to year index explicitly
            try:
                year_idx = int(str(target).replace("risk_year", ""))
            except Exception:
                # Fallback to 1-based index if parsing fails (should not happen)
                year_idx = 1
            for cls in range(4):
                preds_df[f"proba_y{year_idx}_{cls}"] = proba_full[:, cls]

            # Honest OOT evaluation: train eval model on DEV-only labels, evaluate on OOT
            y_all = df[target].astype(float).values
            dev_mask = (~oot_mask)
            dev_labeled = dev_mask & (~np.isnan(y_all))
            oot_labeled = oot_mask & (~np.isnan(y_all))
            metrics_summary[target] = {"f1_macro": 0.0, "balanced_accuracy": 0.0, "high_risk_recall": 0.0, "n_eval": 0, "split": "none"}
            if dev_labeled.sum() > 0:
                eval_model = xgb_classifier_from_params(params, cfg.random_state)
                X_dev = X_all.loc[dev_labeled]
                y_dev = df[target].loc[dev_labeled].astype(int).values
                eval_sw = compute_sample_weight("balanced", y_dev)
                eval_model.fit(X_dev, y_dev, sample_weight=eval_sw)
                if oot_labeled.sum() > 0:
                    X_oot = X_all.loc[oot_labeled]
                    y_oot = df[target].loc[oot_labeled].astype(int).values
                    proba_eval = eval_model.predict_proba(X_oot)
                    metrics_summary[target] = compute_metrics_per_target(y_oot.astype(float), proba_eval)
                    metrics_summary[target]["n_eval"] = int(len(y_oot))
                    metrics_summary[target]["split"] = "oot"
                    y_pred_eval = np.argmax(proba_eval, axis=1)
                    plot_confusion_matrix_counts(
                        y_oot,
                        y_pred_eval,
                        os.path.join(plots_dir, f"confusion_matrix_{target}.png"),
                        title=f"혼동행렬 (OOT) - {target}"
                    )
                else:
                    # Fallback: evaluate on DEV (optimistic), still provide visibility
                    proba_eval_dev = eval_model.predict_proba(X_dev)
                    metrics_summary[target] = compute_metrics_per_target(y_dev.astype(float), proba_eval_dev)
                    metrics_summary[target]["n_eval"] = int(len(y_dev))
                    metrics_summary[target]["split"] = "dev"
                    y_pred_eval_dev = np.argmax(proba_eval_dev, axis=1)
                    plot_confusion_matrix_counts(
                        y_dev,
                        y_pred_eval_dev,
                        os.path.join(plots_dir, f"confusion_matrix_dev_{target}.png"),
                        title=f"혼동행렬 (DEV) - {target}"
                    )
            else:
                print(f"[Step9] Warning: No labeled DEV rows for {target}. Skipping evaluation.")

            # Feature importance (gain) plot
            fi_df = feature_importance_gain(model, feature_cols)
            plot_feature_importance(
                fi_df,
                os.path.join(plots_dir, f"feature_importance_{target}.png"),
                title=f"특성 중요도 (Gain) - {target}",
                top_n=20,
            )

    else:
        # Unified model with task_id
        frames: List[pd.DataFrame] = []
        labels: List[np.ndarray] = []
        task_id_map = {f"risk_year{i}": (i - 1) for i in range(1, 5)}
        for t in range(1, 5):
            target = f"risk_year{t}"
            if target not in df.columns:
                continue
            mask = ~pd.isna(df[target])
            if mask.sum() == 0:
                continue
            X_t = X_all.loc[mask].copy()
            X_t["task_id"] = task_id_map[target]
            frames.append(X_t)
            labels.append(df[target].loc[mask].astype(int).values)
        if not frames:
            raise ValueError("No labeled rows found for any target. Cannot train unified model.")

        X_stacked = pd.concat(frames, axis=0).reset_index(drop=True)
        y_stacked = np.concatenate(labels, axis=0)

        params = best_params_map.get("unified", {})
        model = xgb_classifier_from_params(params, cfg.random_state)
        sample_weights = compute_sample_weight("balanced", y_stacked)
        model.fit(X_stacked, y_stacked, sample_weight=sample_weights)

        # Persist booster
        save_model_booster(model, os.path.join(models_dir, "unified.json"))

        # Predict for each target by setting task_id (production predictions)
        for t in range(1, 5):
            X_pred = X_all.copy()
            X_pred["task_id"] = t - 1
            proba_all = model.predict_proba(X_pred)
            proba_full = expand_proba_to_4_classes(getattr(model, "classes_", np.array([0, 1, 2, 3])), proba_all)
            for cls in range(4):
                preds_df[f"proba_y{t}_{cls}"] = proba_full[:, cls]

            target = f"risk_year{t}"
            if target in df.columns:
                # Honest OOT evaluation with unified eval model trained on DEV
                y_all = df[target].astype(float).values
                dev_mask = (~oot_mask)
                dev_labeled = dev_mask & (~np.isnan(y_all))
                oot_labeled = oot_mask & (~np.isnan(y_all))
                metrics_summary[target] = {"f1_macro": 0.0, "balanced_accuracy": 0.0, "high_risk_recall": 0.0, "n_eval": 0, "split": "none"}
                if dev_labeled.sum() > 0:
                    # Build unified DEV-only training data
                    frames_dev: List[pd.DataFrame] = []
                    labels_dev: List[np.ndarray] = []
                    for tt in range(1, 5):
                        col = f"risk_year{tt}"
                        if col not in df.columns:
                            continue
                        y_all_t = df[col].astype(float).values
                        m = dev_mask & (~np.isnan(y_all_t))
                        if m.sum() == 0:
                            continue
                        X_t = X_all.loc[m].copy()
                        X_t["task_id"] = task_id_map[col]
                        frames_dev.append(X_t)
                        labels_dev.append(df[col].loc[m].astype(int).values)
                    if frames_dev:
                        X_stacked_dev = pd.concat(frames_dev, axis=0).reset_index(drop=True)
                        y_stacked_dev = np.concatenate(labels_dev, axis=0)
                        eval_model = xgb_classifier_from_params(params, cfg.random_state)
                        eval_sw = compute_sample_weight("balanced", y_stacked_dev)
                        eval_model.fit(X_stacked_dev, y_stacked_dev, sample_weight=eval_sw)
                        if oot_labeled.sum() > 0:
                            X_oot = X_all.loc[oot_labeled].copy()
                            X_oot["task_id"] = t - 1
                            y_oot = df[target].loc[oot_labeled].astype(int).values
                            proba_eval = eval_model.predict_proba(X_oot)
                            metrics_summary[target] = compute_metrics_per_target(y_oot.astype(float), proba_eval)
                            metrics_summary[target]["n_eval"] = int(len(y_oot))
                            metrics_summary[target]["split"] = "oot"
                            y_pred_eval = np.argmax(proba_eval, axis=1)
                            plot_confusion_matrix_counts(
                                y_oot,
                                y_pred_eval,
                                os.path.join(plots_dir, f"confusion_matrix_{target}.png"),
                                title=f"혼동행렬 (OOT) - {target}"
                            )
                        else:
                            # Fallback DEV
                            X_dev_t = X_all.loc[dev_labeled].copy()
                            X_dev_t["task_id"] = t - 1
                            y_dev_t = df[target].loc[dev_labeled].astype(int).values
                            proba_eval_dev = eval_model.predict_proba(X_dev_t)
                            metrics_summary[target] = compute_metrics_per_target(y_dev_t.astype(float), proba_eval_dev)
                            metrics_summary[target]["n_eval"] = int(len(y_dev_t))
                            metrics_summary[target]["split"] = "dev"
                            y_pred_eval_dev = np.argmax(proba_eval_dev, axis=1)
                            plot_confusion_matrix_counts(
                                y_dev_t,
                                y_pred_eval_dev,
                                os.path.join(plots_dir, f"confusion_matrix_dev_{target}.png"),
                                title=f"혼동행렬 (DEV) - {target}"
                            )
                    else:
                        print(f"[Step9] Warning: No unified DEV-labeled frames available for evaluation.")
                else:
                    print(f"[Step9] Warning: No labeled DEV rows for {target}. Skipping evaluation.")

        # Unified feature importance (overall)
        fi_df = feature_importance_gain(model, feature_cols + ["task_id"])
        if not fi_df.empty:
            fi_df = fi_df[fi_df["feature"] != "task_id"]
        plot_feature_importance(
            fi_df,
            os.path.join(plots_dir, "feature_importance_overall.png"),
            title="특성 중요도 (Gain) - Unified",
            top_n=20,
        )

    # Save predictions CSV
    preds_df.to_csv(cfg.predictions_path, index=False)

    # Quick report
    lines: List[str] = []
    lines.append("# Step 9 Quick Report")
    lines.append("")
    lines.append(f"- Execution: {datetime.now().isoformat()}")
    lines.append(f"- Approach: {approach} (GPU=off)")
    lines.append(f"- Dataset rows: {len(df):,}; Features: {len(feature_cols):,}")
    lines.append(f"- Predictions: {cfg.predictions_path}")
    lines.append("")
    lines.append("## OOT Metrics (last 20% by date unless overridden)")
    for target in sorted([c for c in metrics_summary.keys() if c.startswith("risk_year")]):
        m = metrics_summary[target]
        lines.append(
            f"- {target}: F1-macro={m.get('f1_macro', 0.0):.3f}, "
            f"BalancedAcc={m.get('balanced_accuracy', 0.0):.3f}, "
            f"HighRiskRecall={m.get('high_risk_recall', 0.0):.3f} (n={m.get('n_eval', 0)})"
        )
    lines.append("")
    lines.append("## Figures")
    lines.append(f"- Confusion matrices: {os.path.join(plots_dir, 'confusion_matrix_*.png')}")
    lines.append(f"- Feature importance: {os.path.join(plots_dir, 'feature_importance_*.png')}")

    ensure_dir(cfg.output_dir)
    with open(os.path.join(cfg.output_dir, "quick_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Manifest
    model_manifest["metrics_oot"] = metrics_summary
    with open(os.path.join(cfg.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(model_manifest, f, indent=2, ensure_ascii=False)

    print(f"✅ Step 9 completed. Predictions saved to: {cfg.predictions_path}")
    print(f"   Quick report: {os.path.join(cfg.output_dir, 'quick_report.md')}")
    print(f"   Manifest: {os.path.join(cfg.output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()


