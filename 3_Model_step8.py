"""
XGBoost Risk Prediction Model - Step 8: Hyperparameter Tuning with Optuna
=======================================================================

Objective, efficient, and explainable tuning for the Two-Stage Cascade (Gate + Specialist)
using rolling-window CV and the same imbalance handling as Step 6.

Primary objective:
- Gate (2/3 vs 0/1): maximize mean high-risk PR-AUC (AP) across folds
- Specialist (3 vs 2): maximize mean PR-AUC for class 3 vs 2 across folds

Tie-breaks (applied when comparing top trials outside Optuna, not weighted in score):
- Higher mean high-risk recall (Gate) or class-3 recall (Specialist) at tuned thresholds
- Higher mean macro-F1 (for Gate thresholds applied downstream)
- Lower std of PR-AUC across folds

Notes:
- Rolling-window CV and fold generation mirror Step 5.
- EENS class-balanced weights, temporal Platt calibration, and prior correction are applied.
- Thresholds are optimized post-hoc per fold, but not part of the PR-AUC objective.
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

from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WINDOW_SIZE = 0.6
N_SPLITS = 5
RANDOM_STATE = 42
CALIBRATION_FRAC = 0.1
EENS_BETA = 0.999

GATE_TRIALS = 80
SPECIALIST_TRIALS = 80

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


def compute_class_balanced_weights_eens(y: np.ndarray, beta: float = EENS_BETA) -> np.ndarray:
    unique, counts = np.unique(y, return_counts=True)
    class_to_weight = {}
    for k, n_k in zip(unique, counts):
        effective_num = 1.0 - (beta ** n_k)
        class_to_weight[int(k)] = (1.0 - beta) / max(effective_num, 1e-12)
    w = np.array([class_to_weight[int(c)] for c in y], dtype=float)
    return w / (w.mean() + 1e-12)


def prior_correction_binary(p: np.ndarray, train_pos: float, ref_pos: float) -> np.ndarray:
    # Adjust binary probability using priors; keep numerically stable
    p = np.clip(p, 1e-12, 1 - 1e-12)
    q_pos = p * (ref_pos / max(train_pos, 1e-12))
    q_neg = (1.0 - p) * ((1.0 - ref_pos) / max(1.0 - train_pos, 1e-12))
    q_sum = q_pos + q_neg
    return np.clip(q_pos / np.maximum(q_sum, 1e-12), 1e-12, 1 - 1e-12)


def fit_platt(proba_1d: np.ndarray, y_true_bin: np.ndarray):
    # Fit logistic calibration for binary case on provided arrays
    lr = LogisticRegression(max_iter=1000)
    pos = y_true_bin.sum()
    if pos == 0 or pos == len(y_true_bin):
        return None
    lr.fit(proba_1d.reshape(-1, 1), y_true_bin)
    return lr


def apply_platt(lr, proba_1d: np.ndarray) -> np.ndarray:
    if lr is None:
        return proba_1d
    return lr.predict_proba(proba_1d.reshape(-1, 1))[:, 1]


def train_binary_with_calibration(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                                  params: Dict[str, Any], calibration_frac: float = CALIBRATION_FRAC,
                                  seed: int = RANDOM_STATE) -> Tuple[np.ndarray, float]:
    y_train = y_train.astype(int)
    w = compute_class_balanced_weights_eens(y_train.values)

    # Split calibration tail from training (temporal)
    cal_size = max(1, int(len(X_train) * calibration_frac)) if len(X_train) > 20 else 0
    if cal_size > 0:
        X_inner = X_train.iloc[:-cal_size]
        y_inner = y_train.iloc[:-cal_size]
        w_inner = w[:-cal_size]
        X_cal = X_train.iloc[-cal_size:]
        y_cal = y_train.iloc[-cal_size:]
    else:
        X_inner, y_inner, w_inner = X_train, y_train, w
        X_cal, y_cal = None, None

    model = xgb.XGBClassifier(
        random_state=seed,
        n_jobs=-1,
        tree_method='hist',
        eval_metric='logloss',
        **params,
    )

    eval_set = [(X_cal, y_cal)] if X_cal is not None else None
    try:
        model.fit(
            X_inner, y_inner,
            sample_weight=w_inner,
            eval_set=eval_set,
            verbose=False,
            early_stopping_rounds=50 if eval_set else None,
        )
    except Exception:
        model.fit(X_inner, y_inner, sample_weight=w_inner)

    proba_test = model.predict_proba(X_test)[:, 1]

    # Platt calibration on calibration set
    if X_cal is not None:
        proba_cal = model.predict_proba(X_cal)[:, 1]
        lr = fit_platt(proba_cal, y_cal.values)
        proba_test = apply_platt(lr, proba_test)

    # Priors for correction
    train_pos = float((y_inner == 1).mean()) if len(y_inner) > 0 else 0.5
    ref_pos = float((y_train == 1).mean())
    proba_test = prior_correction_binary(proba_test, train_pos, ref_pos)

    return proba_test, model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration is not None else params.get('n_estimators', 0)


# -----------------------------------------------------------------------------
# Optuna objectives
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


def objective_gate(trial: optuna.Trial, X: pd.DataFrame, y_target: pd.Series, indices) -> float:
    params = suggest_xgb_params(trial)

    pr_aucs = []
    for (train_idx, test_idx) in indices:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_full, y_test_full = y_target.iloc[train_idx], y_target.iloc[test_idx]

        # Clean NaNs
        train_mask = ~pd.isna(y_train_full)
        test_mask = ~pd.isna(y_test_full)
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train_full[train_mask]
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test_full[test_mask]
        if len(y_train_clean) == 0 or len(y_test_clean) == 0:
            continue

        # Gate labels: high-risk vs not
        y_gate_train = (y_train_clean >= 2).astype(int)
        y_gate_test = (y_test_clean >= 2).astype(int)

        proba_test, _ = train_binary_with_calibration(X_train_clean, y_gate_train, X_test_clean, params)

        # PR-AUC for high-risk detection (threshold-free)
        if y_gate_test.sum() == 0:
            continue
        pr_auc = average_precision_score(y_gate_test.values, proba_test)
        pr_aucs.append(pr_auc)

        # Early pruning hint
        trial.report(np.mean(pr_aucs), step=len(pr_aucs))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(pr_aucs)) if pr_aucs else 0.0


def objective_specialist(trial: optuna.Trial, X: pd.DataFrame, y_target: pd.Series, indices) -> float:
    params = suggest_xgb_params(trial)

    pr_aucs = []
    for (train_idx, test_idx) in indices:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_full, y_test_full = y_target.iloc[train_idx], y_target.iloc[test_idx]

        train_mask = ~pd.isna(y_train_full)
        test_mask = ~pd.isna(y_test_full)
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train_full[train_mask]
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test_full[test_mask]
        if len(y_train_clean) == 0 or len(y_test_clean) == 0:
            continue

        # Specialist labels: among high-risk only (2 vs 3)
        hr_train_mask = (y_train_clean >= 2)
        hr_test_mask = (y_test_clean >= 2)
        if hr_train_mask.sum() < 5 or hr_test_mask.sum() < 5:
            continue

        y_spec_train = (y_train_clean[hr_train_mask] == 3).astype(int)
        # Train on HR subset
        proba_spec, _ = train_binary_with_calibration(
            X_train_clean[hr_train_mask], y_spec_train, X_test_clean, params
        )

        # Evaluate PR-AUC on HR subset of test
        y_spec_test = (y_test_clean[hr_test_mask] == 3).astype(int)
        proba_spec_eval = proba_spec[hr_test_mask.values]
        if y_spec_test.sum() == 0:
            continue
        pr_auc = average_precision_score(y_spec_test.values, proba_spec_eval)
        pr_aucs.append(pr_auc)

        trial.report(np.mean(pr_aucs), step=len(pr_aucs))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(pr_aucs)) if pr_aucs else 0.0


# -----------------------------------------------------------------------------
# Tuning orchestrators
# -----------------------------------------------------------------------------

def tune_gate_for_target(X: pd.DataFrame, y_target: pd.Series, indices, n_trials: int = GATE_TRIALS):
    study = optuna.create_study(
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner(n_warmup_steps=10),
        direction='maximize',
        study_name='gate_study'
    )
    study.optimize(lambda t: objective_gate(t, X, y_target, indices), n_trials=n_trials, show_progress_bar=False)
    return study


def tune_specialist_for_target(X: pd.DataFrame, y_target: pd.Series, indices, n_trials: int = SPECIALIST_TRIALS):
    study = optuna.create_study(
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner(n_warmup_steps=10),
        direction='maximize',
        study_name='specialist_study'
    )
    study.optimize(lambda t: objective_specialist(t, X, y_target, indices), n_trials=n_trials, show_progress_bar=False)
    return study


def compute_thresholds_after_tuning(X: pd.DataFrame, y_target: pd.Series, indices,
                                    gate_params: Dict[str, Any], spec_params: Dict[str, Any]):
    # Derive average thresholds by maximizing F1 per fold
    gate_thresholds = []
    spec_thresholds = []

    for (train_idx, test_idx) in indices:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_full, y_test_full = y_target.iloc[train_idx], y_target.iloc[test_idx]
        train_mask = ~pd.isna(y_train_full)
        test_mask = ~pd.isna(y_test_full)
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train_full[train_mask]
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test_full[test_mask]
        if len(y_train_clean) == 0 or len(y_test_clean) == 0:
            continue

        # Gate
        y_gate_train = (y_train_clean >= 2).astype(int)
        y_gate_test = (y_test_clean >= 2).astype(int)
        proba_gate, _ = train_binary_with_calibration(X_train_clean, y_gate_train, X_test_clean, gate_params)

        best_f1, best_t = -1.0, 0.5
        for t in np.linspace(0.1, 0.9, 17):
            y_pred = (proba_gate >= t).astype(int)
            f1 = f1_score(y_gate_test.values, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        gate_thresholds.append(best_t)

        # Specialist (on HR subset)
        hr_train_mask = (y_train_clean >= 2)
        hr_test_mask = (y_test_clean >= 2)
        if hr_train_mask.sum() < 5 or hr_test_mask.sum() < 5:
            continue
        y_spec_train = (y_train_clean[hr_train_mask] == 3).astype(int)
        proba_spec, _ = train_binary_with_calibration(
            X_train_clean[hr_train_mask], y_spec_train, X_test_clean, spec_params
        )
        y_spec_test = (y_test_clean[hr_test_mask] == 3).astype(int)
        proba_spec_eval = proba_spec[hr_test_mask.values]

        best_f1_s, best_t_s = -1.0, 0.5
        for t in np.linspace(0.1, 0.9, 17):
            y_pred_s = (proba_spec_eval >= t).astype(int)
            f1_s = f1_score(y_spec_test.values, y_pred_s, zero_division=0)
            if f1_s > best_f1_s:
                best_f1_s, best_t_s = f1_s, t
        spec_thresholds.append(best_t_s)

    return float(np.mean(gate_thresholds)) if gate_thresholds else 0.5, \
           float(np.mean(spec_thresholds)) if spec_thresholds else 0.5


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df, X, y, exclude_cols, target_cols = load_step4_data()

    print("🚀 STEP 8: Hyperparameter Tuning with Optuna (Two-Stage Cascade)")
    print("=" * 70)
    print(f"📊 Features: {X.shape[1]}")
    print(f"🎯 Targets: {target_cols}")

    indices = generate_rolling_window_indices(X)
    all_results = {}

    for target in target_cols:
        print(f"\n🎯 TUNING TARGET: {target}")
        # Per-target non-null mask for awareness (folds still filter per fold)
        available = y[target].notna().sum()
        print(f"   Available labeled rows: {available}")

        # Gate tuning
        print("   🔧 Tuning Gate (high-risk vs not) ...")
        gate_study = tune_gate_for_target(X, y[target], indices, n_trials=GATE_TRIALS)
        best_gate = gate_study.best_params
        best_gate_score = gate_study.best_value
        print(f"   ✅ Best Gate PR-AUC: {best_gate_score:.4f}")

        # Specialist tuning
        print("   🔧 Tuning Specialist (3 vs 2) ...")
        spec_study = tune_specialist_for_target(X, y[target], indices, n_trials=SPECIALIST_TRIALS)
        best_spec = spec_study.best_params
        best_spec_score = spec_study.best_value
        print(f"   ✅ Best Specialist PR-AUC: {best_spec_score:.4f}")

        # Thresholds after tuning (for reporting only)
        gate_t, spec_t = compute_thresholds_after_tuning(X, y[target], indices, best_gate, best_spec)
        print(f"   🎚️ Gate threshold (avg): {gate_t:.3f} | Specialist threshold (avg): {spec_t:.3f}")

        all_results[target] = {
            'gate': {
                'best_params': best_gate,
                'best_pr_auc': best_gate_score,
            },
            'specialist': {
                'best_params': best_spec,
                'best_pr_auc': best_spec_score,
            },
            'avg_thresholds': {
                'gate': gate_t,
                'specialist': spec_t,
            }
        }

    # Save results
    summary = {
        'execution_date': datetime.now().isoformat(),
        'approach': 'optuna_tuning_two_stage_cascade',
        'window_size': WINDOW_SIZE,
        'n_splits': N_SPLITS,
        'calibration_frac': CALIBRATION_FRAC,
        'eens_beta': EENS_BETA,
        'trials': {
            'gate': GATE_TRIALS,
            'specialist': SPECIALIST_TRIALS,
        },
        'targets_tuned': len(all_results),
        'results': all_results,
    }

    with open(os.path.join(RESULTS_DIR, 'step8_optuna_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n🎉 STEP 8 TUNING COMPLETED!")
    print("=" * 70)
    print(f"✅ Results saved: {os.path.join(RESULTS_DIR, 'step8_optuna_summary.json')}")


if __name__ == '__main__':
    main()

