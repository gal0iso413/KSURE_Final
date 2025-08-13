"""
XGBoost Risk Prediction Model - Step 8 (GPU): Hyperparameter Tuning (Individual Per-Target Multiclass)
====================================================================================================

GPU-enabled variant of Step 8 for per-target multiclass tuning using XGBoost.
Uses rolling-window CV with per-target matured folds and Step 6 imbalance
handling (class-balanced sample weights; argmax decision).

Primary objective:
- Maximize mean macro F1 across all folds per target.

Notes:
- Rolling-window fold generation mirrors Step 5.
- Evaluation matches Step 7 individual evaluation style: for each target, train
  a separate model per fold using that target's training split and evaluate on
  that target's test split.
- No cascades, no calibration, no prior correction, no ordinal in tuning.
- This file mirrors `3_Model_step8_individual.py` but runs XGBoost with GPU
  acceleration (tree_method='gpu_hist', predictor='gpu_predictor').
"""

import os
import json
from datetime import datetime
from typing import Dict, Any

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
# Utilities (shared with previous steps)
# -----------------------------------------------------------------------------

def load_step4_data():
    df = pd.read_csv(DATASET_PATH)
    if '보험청약일자' in df.columns:
        df = df.sort_values('보험청약일자').reset_index(drop=True)
    exclude_cols = [
        '사업자등록번호', '대상자명', '대상자등록이력일시', '대상자기본주소',
        '청약번호', '보험청약일자', '청약상태코드', '수출자대상자번호',
        '특별출연협약코드', '업종코드1'
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
# Optuna objectives (Individual Per-Target Multiclass, GPU)
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


def objective_individual_multiclass_for_target(trial: optuna.Trial, X: pd.DataFrame, y_target: pd.Series, indices) -> float:
    params = suggest_xgb_params(trial)

    f1_macros = []
    for (train_idx, test_idx) in indices:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_full, y_test_full = y_target.iloc[train_idx], y_target.iloc[test_idx]

        # Clean NaNs
        train_mask = ~pd.isna(y_train_full)
        test_mask = ~pd.isna(y_test_full)
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train_full[train_mask].astype(int).values
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test_full[test_mask].astype(int).values
        if len(y_train_clean) == 0 or len(y_test_clean) == 0:
            continue

        sample_weights = compute_sample_weights_balanced(y_train_clean)
        model = xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            eval_metric='mlogloss',
            **params,
        )
        model.fit(X_train_clean, y_train_clean, sample_weight=sample_weights)

        proba_test = model.predict_proba(X_test_clean)
        y_pred = np.argmax(proba_test, axis=1)

        # Macro F1 across all classes
        f1m = f1_score(y_test_clean, y_pred, average='macro', zero_division=0)
        f1_macros.append(f1m)

        # Early pruning hint
        trial.report(np.mean(f1_macros), step=len(f1_macros))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(f1_macros)) if f1_macros else 0.0


def tune_individual_multiclass_for_target(X: pd.DataFrame, y_target: pd.Series, indices, n_trials: int = TRIALS):
    study = optuna.create_study(
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner(n_warmup_steps=10),
        direction='maximize',
        study_name='individual_multiclass_study_gpu'
    )
    study.optimize(lambda t: objective_individual_multiclass_for_target(t, X, y_target, indices), n_trials=n_trials, show_progress_bar=False)
    return study


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df, X, y, exclude_cols, target_cols = load_step4_data()

    print("🚀 STEP 8 (GPU): Hyperparameter Tuning with Optuna (Individual Per-Target Multiclass)")
    print("=" * 70)
    print(f"📊 Features: {X.shape[1]}")
    print(f"🎯 Targets: {target_cols}")

    all_results: Dict[str, Any] = {}
    for target in target_cols:
        print(f"\n🎯 TUNING TARGET: {target}")
        indices = generate_target_folds_original_indices(X, y[target], window_size=WINDOW_SIZE, n_splits=N_SPLITS)
        print(f"   Folds available: {len(indices)}")
        study = tune_individual_multiclass_for_target(X, y[target], indices, n_trials=TRIALS)
        best_params = study.best_params
        best_score = study.best_value
        print(f"   ✅ Best mean F1-Macro: {best_score:.4f}")

        all_results[target] = {
            'best_params': best_params,
            'best_score_mean_f1_macro': best_score,
        }

    # Save results
    summary = {
        'execution_date': datetime.now().isoformat(),
        'approach': 'optuna_tuning_individual_multiclass',
        'window_size': WINDOW_SIZE,
        'n_splits': N_SPLITS,
        'trials': TRIALS,
        'targets_tuned': len(all_results),
        'results': all_results,
    }

    with open(os.path.join(RESULTS_DIR, 'step8_optuna_summary_gpu_individual.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Also save a compact best-params file for downstream steps
    best_params_by_target = {t: info['best_params'] for t, info in all_results.items()}
    compact = {
        'execution_date': datetime.now().isoformat(),
        'approach': 'individual_multiclass',
        'targets': target_cols,
        'best_params_by_target': best_params_by_target,
        'best_score_metric': 'f1_macro',
        'accelerator': 'gpu',
    }
    with open(os.path.join(RESULTS_DIR, 'step8_best_params_individual_gpu.json'), 'w', encoding='utf-8') as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)

    print("\n🎉 STEP 8 (GPU) TUNING COMPLETED!")
    print("=" * 70)
    print(f"✅ Results saved: {os.path.join(RESULTS_DIR, 'step8_optuna_summary_gpu_individual.json')}")
    print(f"✅ Best params saved: {os.path.join(RESULTS_DIR, 'step8_best_params_individual_gpu.json')}")


if __name__ == '__main__':
    main()


