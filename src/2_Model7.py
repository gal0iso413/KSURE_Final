"""
XGBoost Risk Prediction Model - Step 7: Unified Hyperparameter Tuning
====================================================================

Configuration-driven hyperparameter tuning for both individual and unified 
multiclass models using Optuna with rolling-window CV and Step 5 imbalance
handling (class-balanced sample weights; argmax decision).

Key Improvements:
- Single unified script eliminates code duplication across 4 separate files
- Configuration-driven approach for architecture (individual/unified) and device (cpu/gpu)
- Maintains all original functionality while improving maintainability
- Easy experimentation through config changes

Primary objective:
- Maximize mean macro F1 across all targets and folds

Notes:
- Rolling-window fold generation mirrors Step 4 (Temporal Validation)
- Evaluation matches Step 6 (Model Architecture) evaluation styles
- No cascades, no calibration, no prior correction, no ordinal in tuning
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

import warnings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuration - MODIFY THIS SECTION TO CHANGE BEHAVIOR
# -----------------------------------------------------------------------------

TUNING_CONFIG = {
    'architecture': 'unified',    # 'individual' or 'unified'
    'device': 'cpu',              # 'cpu' or 'gpu'  
    'n_trials': 80,               # Number of Optuna trials
    'window_size': 0.6,           # Rolling window size
    'n_splits': 5,                # Number of CV splits
    'random_state': 42,           # Random seed for reproducibility
}

# Derived settings
RESULTS_DIR = 'result/step7_optuna'
DATASET_PATH = 'dataset/credit_risk_dataset_selected.csv'

# -----------------------------------------------------------------------------
# Utilities (shared across all configurations)
# -----------------------------------------------------------------------------

def load_selected_data():
    """
    🔄 CRITICAL FIX: Load TRAIN+VALIDATION for hyperparameter tuning (NO OOT LEAKAGE!)
    
    Hyperparameter tuning should optimize for validation performance, never OOT.
    This ensures final OOT evaluation remains unbiased.
    """
    print("📂 Loading TRAIN + VALIDATION data for hyperparameter tuning...")
    print("   ⚠️  OOT data is EXCLUDED to prevent leakage!")
    
    try:
        # Load split datasets
        train_df = pd.read_csv('../data/splits/train_data.csv')
        validation_df = pd.read_csv('../data/splits/validation_data.csv')
        
        # 🔄 CRITICAL: Use TRAIN + VALIDATION for hyperparameter tuning
        # This allows hyperparameter optimization while preserving OOT for final evaluation
        df = pd.concat([train_df, validation_df], ignore_index=True)
        
        print(f"✅ Hyperparameter tuning data loaded:")
        print(f"   - TRAIN: {len(train_df):,} rows")
        print(f"   - VALIDATION: {len(validation_df):,} rows") 
        print(f"   - COMBINED: {len(df):,} rows (for hyperparameter optimization)")
        print(f"   - OOT: EXCLUDED (preserves unbiased evaluation)")
        
    except FileNotFoundError:
        print("❌ ERROR: Split datasets not found!")
        print("   Please run '1_Split.py' first to create train/validation/oot splits")
        raise FileNotFoundError("Run 1_Split.py first to create proper data splits")
    
    if '보험청약일자' in df.columns:
        df = df.sort_values('보험청약일자').reset_index(drop=True)
    
    # Updated exclude columns for strict pipeline
    exclude_cols = [
        '사업자등록번호', '대상자명', '청약번호', '보험청약일자', '수출자대상자번호', '업종코드1', 'unique_id', 'data_split'
    ]
    target_cols = [c for c in df.columns if c.startswith('risk_year')]
    feature_cols = [c for c in df.columns if c not in exclude_cols + target_cols]
    
    print(f"📊 Hyperparameter tuning setup:")
    print(f"   - Features: {len(feature_cols)}")
    print(f"   - Targets: {len(target_cols)}")
    print(f"   - Period: {df['보험청약일자'].min()} to {df['보험청약일자'].max()}")
    
    X = df[feature_cols]
    y = df[target_cols]
    return df, X, y, exclude_cols, target_cols


def generate_rolling_window_indices(X: pd.DataFrame, window_size: float, n_splits: int):
    """Generate rolling window indices for consistent temporal validation."""
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
    """Compute sklearn 'balanced' sample weights."""
    return compute_sample_weight('balanced', y)


def generate_target_folds_original_indices(X: pd.DataFrame, y_target: pd.Series, 
                                         window_size: float, n_splits: int):
    """Generate rolling window folds for a specific target, mapping back to original indices."""
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


def suggest_xgb_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest XGBoost hyperparameters for Optuna optimization."""
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


def get_xgb_base_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get base XGBoost parameters based on configuration."""
    base_params = {
        'random_state': config['random_state'],
        'n_jobs': -1,
        'eval_metric': 'mlogloss',
        'verbosity': 0,
    }
    
    if config['device'] == 'gpu':
        base_params['tree_method'] = 'gpu_hist'
        base_params['predictor'] = 'gpu_predictor'
    else:
        base_params['tree_method'] = 'hist'
    
    return base_params


# -----------------------------------------------------------------------------
# Architecture-specific implementations
# -----------------------------------------------------------------------------

def stack_multitask_training_data(X_train: pd.DataFrame, y_train_df: pd.DataFrame, 
                                target_cols: list) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """Create stacked training data for unified multitask training."""
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


def objective_individual_multiclass(trial: optuna.Trial, X: pd.DataFrame, y_target: pd.Series, 
                                   indices: list, base_params: Dict[str, Any]) -> float:
    """Objective function for individual per-target multiclass tuning."""
    params = suggest_xgb_params(trial)
    params.update(base_params)
    
    f1_macros = []
    for train_idx, test_idx in indices:
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
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_clean, y_train_clean, sample_weight=sample_weights)
        
        proba_test = model.predict_proba(X_test_clean)
        y_pred = np.argmax(proba_test, axis=1)
        
        f1m = f1_score(y_test_clean, y_pred, average='macro', zero_division=0)
        f1_macros.append(f1m)
        
        # Early pruning hint
        trial.report(np.mean(f1_macros), step=len(f1_macros))
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return float(np.mean(f1_macros)) if f1_macros else 0.0


def objective_unified_multiclass(trial: optuna.Trial, X: pd.DataFrame, y: pd.DataFrame, 
                                target_cols: list, precomputed_folds: dict, 
                                base_params: Dict[str, Any]) -> float:
    """Objective function for unified multitask multiclass tuning."""
    params = suggest_xgb_params(trial)
    params.update(base_params)
    
    f1_macros = []
    
    # Evaluate unified model per target using that target's matured folds
    for target_name in target_cols:
        task_id = target_cols.index(target_name)
        folds = precomputed_folds.get(target_name, [])
        
        for train_idx, test_idx in folds:
            X_train_fold = X.loc[train_idx]
            y_train_df_fold = y.loc[train_idx]
            X_stacked, y_stacked = stack_multitask_training_data(X_train_fold, y_train_df_fold[target_cols], target_cols)
            
            if X_stacked is None or len(y_stacked) == 0:
                continue
            
            sample_weights = compute_sample_weights_balanced(y_stacked)
            model = xgb.XGBClassifier(**params)
            model.fit(X_stacked, y_stacked, sample_weight=sample_weights)
            
            X_test = X.loc[test_idx].copy()
            y_test = y[target_name].loc[test_idx].astype(int).values
            X_test['task_id'] = task_id
            
            proba_test = model.predict_proba(X_test)
            y_pred = np.argmax(proba_test, axis=1)
            
            f1m = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macros.append(f1m)
            
            # Early pruning hint
            trial.report(np.mean(f1_macros), step=len(f1_macros))
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    return float(np.mean(f1_macros)) if f1_macros else 0.0


# -----------------------------------------------------------------------------
# Tuning orchestrators
# -----------------------------------------------------------------------------

def tune_individual_architecture(X: pd.DataFrame, y: pd.DataFrame, target_cols: list, 
                                config: Dict[str, Any]) -> Dict[str, Any]:
    """Tune hyperparameters for individual per-target models."""
    print(f"🎯 TUNING INDIVIDUAL ARCHITECTURE ({config['device'].upper()})")
    print(f"   Trials: {config['n_trials']}, Targets: {len(target_cols)}")
    
    base_params = get_xgb_base_params(config)
    all_results = {}
    
    for target in target_cols:
        print(f"\n🎯 TUNING TARGET: {target}")
        indices = generate_target_folds_original_indices(
            X, y[target], 
            window_size=config['window_size'], 
            n_splits=config['n_splits']
        )
        print(f"   Folds available: {len(indices)}")
        
        study = optuna.create_study(
            sampler=TPESampler(seed=config['random_state']),
            pruner=MedianPruner(n_warmup_steps=10),
            direction='maximize',
            study_name=f'individual_{target}_study'
        )
        
        study.optimize(
            lambda trial: objective_individual_multiclass(trial, X, y[target], indices, base_params),
            n_trials=config['n_trials'],
            show_progress_bar=False
        )
        
        best_params = study.best_params
        best_score = study.best_value
        print(f"   ✅ Best mean F1-Macro: {best_score:.4f}")
        
        all_results[target] = {
            'best_params': best_params,
            'best_score_mean_f1_macro': best_score,
        }
    
    return all_results


def tune_unified_architecture(X: pd.DataFrame, y: pd.DataFrame, target_cols: list, 
                             config: Dict[str, Any]) -> Dict[str, Any]:
    """Tune hyperparameters for unified multitask model."""
    print(f"🎯 TUNING UNIFIED ARCHITECTURE ({config['device'].upper()})")
    print(f"   Trials: {config['n_trials']}, Targets: {len(target_cols)}")
    
    base_params = get_xgb_base_params(config)
    
    # Precompute per-target matured folds once for efficiency
    precomputed_folds = {
        t: generate_target_folds_original_indices(
            X, y[t], 
            window_size=config['window_size'], 
            n_splits=config['n_splits']
        ) 
        for t in target_cols
    }
    
    print("   🔧 Tuning unified multiclass model (class weights; argmax)…")
    study = optuna.create_study(
        sampler=TPESampler(seed=config['random_state']),
        pruner=MedianPruner(n_warmup_steps=10),
        direction='maximize',
        study_name='unified_multiclass_study'
    )
    
    study.optimize(
        lambda trial: objective_unified_multiclass(trial, X, y, target_cols, precomputed_folds, base_params),
        n_trials=config['n_trials'],
        show_progress_bar=False
    )
    
    best_params = study.best_params
    best_score = study.best_value
    print(f"   ✅ Best mean F1-Macro: {best_score:.4f}")
    
    return {
        'best_params': best_params,
        'best_score_mean_f1_macro': best_score,
        'targets': target_cols
    }


# -----------------------------------------------------------------------------
# Main execution and results saving
# -----------------------------------------------------------------------------

def save_results(results: Dict[str, Any], config: Dict[str, Any], target_cols: list):
    """Save tuning results with configuration-specific naming."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create detailed summary
    summary = {
        'execution_date': datetime.now().isoformat(),
        'approach': f'optuna_tuning_{config["architecture"]}_multiclass',
        'configuration': config,
        'window_size': config['window_size'],
        'n_splits': config['n_splits'],
        'trials': config['n_trials'],
        'targets': target_cols,
        'results': results,
    }
    
    # Architecture-specific result processing
    if config['architecture'] == 'individual':
        summary['targets_tuned'] = len(results)
    else:  # unified
        summary['best_score_mean_f1_macro'] = results.get('best_score_mean_f1_macro', 0.0)
        summary['best_params'] = results.get('best_params', {})
    
    # Save detailed summary
    summary_filename = f"step7_optuna_summary_{config['architecture']}_{config['device']}.json"
    with open(os.path.join(RESULTS_DIR, summary_filename), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Save compact best-params file for downstream steps
    if config['architecture'] == 'individual':
        best_params_by_target = {t: info['best_params'] for t, info in results.items()}
        compact = {
            'execution_date': datetime.now().isoformat(),
            'approach': 'individual_multiclass',
            'targets': target_cols,
            'best_params_by_target': best_params_by_target,
            'best_score_metric': 'f1_macro',
            'device': config['device'],
        }
    else:  # unified
        compact = {
            'execution_date': datetime.now().isoformat(),
            'approach': 'unified_multiclass',
            'targets': target_cols,
            'best_params': results.get('best_params', {}),
            'best_score_metric': 'f1_macro',
            'device': config['device'],
        }
    
    compact_filename = f"step7_best_params_{config['architecture']}_{config['device']}.json"
    with open(os.path.join(RESULTS_DIR, compact_filename), 'w', encoding='utf-8') as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)
    
    return summary_filename, compact_filename


def main():
    """Main execution function with configuration-driven behavior."""
    print("🚀 STEP 7: Unified Hyperparameter Tuning with Optuna")
    print("=" * 70)
    print(f"📋 Configuration: {TUNING_CONFIG['architecture'].title()} architecture on {TUNING_CONFIG['device'].upper()}")
    
    # Load data
    df, X, y, exclude_cols, target_cols = load_selected_data()
    print(f"📊 Features: {X.shape[1]}")
    print(f"🎯 Targets: {target_cols}")
    
    # Architecture-specific tuning
    if TUNING_CONFIG['architecture'] == 'individual':
        results = tune_individual_architecture(X, y, target_cols, TUNING_CONFIG)
    elif TUNING_CONFIG['architecture'] == 'unified':
        results = tune_unified_architecture(X, y, target_cols, TUNING_CONFIG)
    else:
        raise ValueError(f"Unknown architecture: {TUNING_CONFIG['architecture']}")
    
    # Save results
    summary_file, compact_file = save_results(results, TUNING_CONFIG, target_cols)
    
    print("\n🎉 STEP 7 HYPERPARAMETER TUNING COMPLETED!")
    print("=" * 70)
    print(f"✅ Architecture: {TUNING_CONFIG['architecture'].title()}")
    print(f"✅ Device: {TUNING_CONFIG['device'].upper()}")
    print(f"✅ Summary saved: {summary_file}")
    print(f"✅ Best params saved: {compact_file}")
    print(f"\n💡 To run different configuration:")
    print(f"   1. Modify TUNING_CONFIG at top of script")
    print(f"   2. Re-run: python 3_Model_step7_tune.py")


if __name__ == '__main__':
    main()
