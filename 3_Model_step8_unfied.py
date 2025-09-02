"""
XGBoost Risk Prediction Model - Step 8: Hyperparameter Tuning (Unified Multiclass)
==============================================================================

Objective, efficient, and explainable tuning for the unified multiclass model
using rolling-window CV with per-target matured folds and Step 6 imbalance
handling (class-balanced sample weights; argmax decision).

Primary objective:
- Maximize mean macro F1 across all targets and folds.

Notes:
- Rolling-window fold generation mirrors Step 5.
- Evaluation matches Step 7 unified evaluation style: train a single unified
  model (with `task_id`) per fold and evaluate per-target on that target's
  test split.
- No cascades, no calibration, no prior correction, no ordinal in tuning.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WINDOW_SIZE = 0.6
N_SPLITS = 5
RANDOM_STATE = 42

TRIALS = 80

RESULTS_DIR = 'result/step8_optuna'
DATASET_PATH = 'dataset/credit_risk_dataset_step4.csv'

# -----------------------------------------------------------------------------
# Utilities (aligned with Steps 5-7)
# -----------------------------------------------------------------------------

def load_step4_data():
    df = pd.read_csv(DATASET_PATH)
    if '보험청약일자' in df.columns:
        df = df.sort_values('보험청약일자').reset_index(drop=True)
    exclude_cols = [
        '사업자등록번호', '대상자명', '청약번호', '보험청약일자', '수출자대상자번호', '업종코드1'
    ]
    target_cols = [c for c in df.columns if c.startswith('risk_year')]
    feature_cols = [c for c in df.columns if c not in exclude_cols + target_cols]
    X = df[feature_cols]
    y = df[target_cols]
    return df, X, y, exclude_cols, target_cols


def generate_rolling_window_indices(X: pd.DataFrame, window_size: float = WINDOW_SIZE, n_splits: int = N_SPLITS):
    total = len(X)
    win = int(total * window_size)
    step = max(1, (total - win) // n_splits) if (total - win) > 0 else 1
    indices = []
    for fold in range(n_splits):
        start = fold * step
        end = start + win
        if end >= total:
            break
        test_start = end
        test_end = min(test_start + step, total)
        if test_end <= test_start:
            break
        indices.append((list(range(start, end)), list(range(test_start, test_end))))
    return indices

def compute_sample_weights_balanced(y: np.ndarray) -> np.ndarray:
    return compute_sample_weight('balanced', y)


def generate_target_folds_original_indices(X: pd.DataFrame, y_target: pd.Series,
                                           window_size: float = WINDOW_SIZE, n_splits: int = N_SPLITS):
    mask = ~pd.isna(y_target)
    if mask.sum() == 0:
        return []
    X_masked = X.loc[mask]
    masked_indices = X_masked.index.to_numpy()
    masked_folds = generate_rolling_window_indices(X_masked, window_size=window_size, n_splits=n_splits)
    orig_folds = []
    for train_idx, test_idx in masked_folds:
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        orig_train = masked_indices[train_idx].tolist()
        orig_test = masked_indices[test_idx].tolist()
        orig_folds.append((orig_train, orig_test))
    return orig_folds


# -----------------------------------------------------------------------------
# Optuna objective (Unified Multiclass)
# -----------------------------------------------------------------------------

def suggest_xgb_params(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 150, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
        'max_bin': trial.suggest_categorical('max_bin', [256, 512, 1024]),
    }
    return params

def stack_multitask_training_data(X_train: pd.DataFrame, y_train_df: pd.DataFrame, target_cols) -> Tuple[pd.DataFrame, np.ndarray]:
    frames = []
    labels = []
    for task_id, target_name in enumerate(target_cols):
        y_t = y_train_df[target_name]
        mask = ~pd.isna(y_t)
        if mask.sum() == 0:
            continue
        X_t = X_train.loc[mask].copy()
        X_t['task_id'] = task_id
        frames.append(X_t)
        labels.append(y_t.loc[mask].astype(int).values)
    if not frames:
        return None, None
    X_stacked = pd.concat(frames, axis=0).reset_index(drop=True)
    y_stacked = np.concatenate(labels, axis=0)
    return X_stacked, y_stacked


def objective_unified_multiclass(trial: optuna.Trial, X: pd.DataFrame, y: pd.DataFrame, target_cols, precomputed_folds: dict) -> float:
    params = suggest_xgb_params(trial)

    f1_macros = []

    # Evaluate unified model per target using that target's matured folds
    for target_name in target_cols:
        task_id = target_cols.index(target_name)
        folds = precomputed_folds.get(target_name, [])
        for train_idx, test_idx in folds:
            # Train unified multiclass on full training indices (no early stopping)
            X_train_fold = X.loc[train_idx]
            y_train_df_fold = y.loc[train_idx]
            X_stacked, y_stacked = stack_multitask_training_data(X_train_fold, y_train_df_fold[target_cols], target_cols)
            if X_stacked is None or len(y_stacked) == 0:
                continue

            sample_weights = compute_sample_weights_balanced(y_stacked)
            model = xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method='hist',
                eval_metric='mlogloss',
                **params,
            )
            model.fit(X_stacked, y_stacked, sample_weight=sample_weights)

            X_test = X.loc[test_idx].copy()
            y_test = y[target_name].loc[test_idx].astype(int).values
            X_test['task_id'] = task_id

            proba_test = model.predict_proba(X_test)
            y_pred = np.argmax(proba_test, axis=1)

            # Macro F1 across all classes
            f1m = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macros.append(f1m)

            # Early pruning hint
            trial.report(np.mean(f1_macros), step=len(f1_macros))
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Primary objective: mean macro F1
    return float(np.mean(f1_macros)) if f1_macros else 0.0


def tune_unified_multiclass(X: pd.DataFrame, y: pd.DataFrame, target_cols, precomputed_folds: dict, n_trials: int = TRIALS):
    study = optuna.create_study(
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner(n_warmup_steps=10),
        direction='maximize',
        study_name='unified_multiclass_study'
    )
    study.optimize(lambda t: objective_unified_multiclass(t, X, y, target_cols, precomputed_folds), n_trials=n_trials, show_progress_bar=False)
    return study


# No threshold computation is needed for multiclass argmax


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df, X, y, exclude_cols, target_cols = load_step4_data()

    print("🚀 STEP 8: Hyperparameter Tuning with Optuna (Unified Multiclass)")
    print("=" * 70)
    print(f"📊 Features: {X.shape[1]}")
    print(f"🎯 Targets: {target_cols}")

    # Precompute per-target matured folds once for efficiency
    precomputed_folds = {t: generate_target_folds_original_indices(X, y[t]) for t in target_cols}

    print("   🔧 Tuning unified multiclass model (class weights; argmax)…")
    study = tune_unified_multiclass(X, y, target_cols, precomputed_folds, n_trials=TRIALS)
    best_params = study.best_params
    best_score = study.best_value
    print(f"   ✅ Best mean F1-Macro: {best_score:.4f}")

    # Save results
    summary = {
        'execution_date': datetime.now().isoformat(),
        'approach': 'optuna_tuning_unified_multiclass',
        'window_size': WINDOW_SIZE,
        'n_splits': N_SPLITS,
        'trials': TRIALS,
        'targets': target_cols,
        'best_score_mean_f1_macro': best_score,
        'best_params': best_params,
    }

    with open(os.path.join(RESULTS_DIR, 'step8_optuna_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Also save a compact best-params file for downstream steps
    compact = {
        'execution_date': datetime.now().isoformat(),
        'approach': 'unified_multiclass',
        'targets': target_cols,
        'best_params': best_params,
        'best_score_metric': 'f1_macro',
    }
    with open(os.path.join(RESULTS_DIR, 'step8_best_params_unified.json'), 'w', encoding='utf-8') as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)

    print("\n🎉 STEP 8 TUNING COMPLETED!")
    print("=" * 70)
    print(f"✅ Results saved: {os.path.join(RESULTS_DIR, 'step8_optuna_summary.json')}")
    print(f"✅ Best params saved: {os.path.join(RESULTS_DIR, 'step8_best_params_unified.json')}")


if __name__ == '__main__':
    main()

