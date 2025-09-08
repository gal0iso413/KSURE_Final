"""
XGBoost Risk Prediction Model - Step 6: Model Architecture Experiments (Two-Phase)
==================================================================================

Step 6 Implementation - Interpretable & Structured:
1. Phase 1: Unified vs Individual Models (both reuse Step 5 handling: class-balanced sample weights; argmax)
2. Phase 2: Best-of-Phase1 vs Ordinal Score + Fixed Cutpoints (interpretable; training-only OOF cutpoints)

Design Focus:
- Use Step 4 optimized features
- Use Step 4 Rolling Window CV for temporal robustness
- Apply Step 5 handling consistently for Phase 1
- No cascade, no ensembles
- Interpretability via single models and three numeric cutpoints for ordinal
 - Focus on high-risk recall, macro F1, balanced accuracy, kappa, high-risk PR-AUC, stability
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
from datetime import datetime
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    balanced_accuracy_score, cohen_kappa_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
import platform
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

# Configure matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Configuration parameters
WINDOW_SIZE = 0.6
N_SPLITS = 5
N_ESTIMATORS = 200
RANDOM_STATE = 42

# Configure Korean font for matplotlib
def setup_korean_font():
    """Set up Korean font for matplotlib visualizations"""
    system = platform.system()
    
    if system == "Windows":
        korean_fonts = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'Gulim', 'Dotum']
    elif system == "Darwin":  # macOS
        korean_fonts = ['AppleGothic', 'NanumGothic']
    else:  # Linux
        korean_fonts = ['NanumGothic', 'NanumBarunGothic', 'UnDotum']
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    korean_font = None
    
    for font in korean_fonts:
        if font in available_fonts:
            korean_font = font
            break
    
    if korean_font:
        plt.rcParams['font.family'] = korean_font
        print(f"✅ Korean font set: {korean_font}")
    else:
        print("⚠️  Preferred Korean fonts not found. Using fallback options...")
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
    
    plt.rcParams['axes.unicode_minus'] = False
    return korean_font

# Set up Korean font
setup_korean_font()

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_selected_data():
    """Load Step 2 optimized data with proper temporal sorting"""
    print("🚀 MODEL ARCHITECTURE EXPERIMENTS - STEP 6")
    print("=" * 70)
    
    base_df = pd.read_csv('../data/processed/credit_risk_dataset_selected.csv')
    df = base_df[base_df['data_split'] == 'development']
    print(f"✅ Step 2 dataset loaded: {df.shape}")
    
    # Sort by 보험청약일자 for temporal validation
    if '보험청약일자' in df.columns:
        df = df.sort_values('보험청약일자').reset_index(drop=True)
        print(f"✅ Data sorted by 보험청약일자 for temporal validation")
    
    exclude_cols = [
        '사업자등록번호', '대상자명', '청약번호', '보험청약일자', '수출자대상자번호', '업종코드1', 'unique_id', 'data_split'
    ]
    
    target_cols = [col for col in df.columns if col.startswith('risk_year')]
    feature_cols = [col for col in df.columns if col not in target_cols + exclude_cols]
    
    print(f"📊 Optimized features: {len(feature_cols)}")
    print(f"🎯 Target columns: {len(target_cols)}")
    
    X = df[feature_cols]
    y = df[target_cols]
    
    return df, X, y, exclude_cols, target_cols

def generate_rolling_window_indices(X, window_size=WINDOW_SIZE, n_splits=N_SPLITS):
    """Generate rolling window indices for consistent data splitting with min step size 1"""
    total_samples = len(X)
    window_samples = int(total_samples * window_size)
    remaining = total_samples - window_samples
    step_size = max(1, remaining // n_splits) if remaining > 0 else 1

    indices_list = []

    for fold in range(n_splits):
        start_idx = fold * step_size
        end_idx = start_idx + window_samples

        if end_idx >= total_samples:
            break

        train_idx = list(range(start_idx, end_idx))
        test_idx = list(range(end_idx, min(end_idx + step_size, total_samples)))

        if len(test_idx) == 0:
            break

        indices_list.append((train_idx, test_idx))

    return indices_list

def stack_multitask_training_data(X_train: pd.DataFrame, y_train_df: pd.DataFrame, target_cols: list) -> tuple:
    """Create a stacked training set with a numeric task_id feature for unified multi-task training."""
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

def generate_target_folds_original_indices(X: pd.DataFrame, y_target: pd.Series, window_size: float = WINDOW_SIZE, n_splits: int = N_SPLITS):
    """Generate rolling-window folds on the target-labeled subset, then map back to original indices.
    Returns list of (orig_train_idx_list, orig_test_idx_list).
    """
    mask = ~pd.isna(y_target)
    if mask.sum() == 0:
        return []
    X_masked = X.loc[mask]
    y_masked = y_target.loc[mask]
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

def calculate_comprehensive_metrics(y_true, y_pred, y_proba=None, high_risk_proba=None):
    """Classification metrics with imbalance awareness"""
    metrics = {
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    # High-risk recall (classes 2-3)
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 2:
        high_risk_mask = y_true >= 2  # Classes 2 and 3 (medium and high risk)
        if high_risk_mask.sum() > 0:
            high_risk_recall = recall_score(y_true[high_risk_mask], y_pred[high_risk_mask], average='macro')
            metrics['high_risk_recall'] = high_risk_recall
        else:
            metrics['high_risk_recall'] = 0.0
    else:
        metrics['high_risk_recall'] = 0.0
    
    # High-risk PR-AUC (classes 2 or 3 as positive)
    if high_risk_proba is not None:
        high_risk_true = (y_true >= 2).astype(int)
        if high_risk_true.sum() > 0:
            metrics['high_risk_pr_auc'] = average_precision_score(high_risk_true, high_risk_proba)
        else:
            metrics['high_risk_pr_auc'] = 0.0
    return metrics

def compute_sample_weights_balanced(y: np.ndarray) -> np.ndarray:
    """Compute sklearn 'balanced' sample weights and return as numpy array."""
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight('balanced', y)

def train_baseline_best_of_step6(*args, **kwargs):
    """Deprecated in Step 7."""
    return {}

def train_ordinal_score_with_cutpoints(X: pd.DataFrame, y_target: pd.Series) -> dict:
    """Ordinal process: single regressor on labels {0,1,2,3} with class weights; cutpoints learned on training-only OOF; predict via binning."""
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.model_selection import KFold
    results = {'metrics_list': [], 'all_true_values': [], 'all_predictions': [], 'all_probabilities': []}
    indices_list = generate_rolling_window_indices(X)
    for fold, (train_idx, test_idx) in enumerate(indices_list):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_target.iloc[train_idx], y_target.iloc[test_idx]
        train_mask = ~pd.isna(y_train)
        test_mask = ~pd.isna(y_test)
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask].astype(int).values
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask].astype(int).values
        if len(y_train_clean) == 0 or len(y_test_clean) == 0:
            continue
        # Out-of-fold scores on training to learn cutpoints
        kf = KFold(n_splits=5, shuffle=False)
        oof_scores = np.zeros(len(X_train_clean))
        for tr, va in kf.split(X_train_clean):
            X_tr, X_va = X_train_clean.iloc[tr], X_train_clean.iloc[va]
            y_tr, y_va = y_train_clean[tr], y_train_clean[va]
            w_tr = compute_sample_weights_balanced(y_tr)
            reg = xgb.XGBRegressor(
                n_estimators=N_ESTIMATORS,
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=-1,
                tree_method='hist',
                eval_metric='rmse'
            )
            reg.fit(X_tr, y_tr, sample_weight=w_tr)
            oof_scores[va] = reg.predict(X_va)
        # learn ordered cutpoints c1<c2<c3 by maximizing macro F1 on training-only OOF
        y_trn = y_train_clean
        s = oof_scores
        cands = np.quantile(s, [0.1, 0.3, 0.5, 0.7, 0.9])
        best = None
        for c1 in cands:
            for c2 in cands:
                if c2 <= c1:
                    continue
                for c3 in cands:
                    if c3 <= c2:
                        continue
                    y_pred_oof = np.digitize(s, bins=[c1, c2, c3])
                    f1 = f1_score(y_trn, y_pred_oof, average='macro', zero_division=0)
                    if (best is None) or (f1 > best[0]):
                        best = (f1, (c1, c2, c3))
        c1, c2, c3 = best[1] if best else (np.min(s), np.median(s), np.max(s))
        # Train on full training
        w_full = compute_sample_weights_balanced(y_train_clean)
        reg_full = xgb.XGBRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='rmse'
        )
        reg_full.fit(X_train_clean, y_train_clean, sample_weight=w_full)
        # Predict scores on test and bin
        score_test = reg_full.predict(X_test_clean)
        y_pred = np.digitize(score_test, bins=[c1, c2, c3])
        proba_test = None
        high_risk_proba = None
        m = calculate_comprehensive_metrics(y_test_clean, y_pred, proba_test, high_risk_proba)
        results['metrics_list'].append(m)
        results['all_true_values'].append(y_test_clean)
        results['all_predictions'].append(y_pred)
        results['all_probabilities'].append(np.zeros((len(y_pred), 4)))
    if results['metrics_list']:
        avg = {k: float(np.mean([mm[k] for mm in results['metrics_list']])) for k in results['metrics_list'][0].keys()}
        results['avg_metrics'] = avg
    else:
        results['avg_metrics'] = {}
    return results

def train_unified_multitask_multiclass(X: pd.DataFrame, y: pd.DataFrame, target_cols: list, n_estimators: int) -> dict:
    """Phase 1 unified model: single multiclass model across targets (task_id feature); class-balanced weights; argmax.

    Evaluation uses per-target matured rolling folds to ensure each target (e.g., risk_year4) has eligible test rows.
    Training remains unified on stacked data within each target's fold window.
    """
    print("🎯 Training Unified Multi-Task Model (single shared multiclass model; class weights; argmax)...")
    per_target_results = {t: {'metrics_list': [], 'all_true_values': [], 'all_predictions': [], 'all_probabilities': []} for t in target_cols}

    # Evaluate unified model per target using that target's matured folds
    for target_name in target_cols:
        target_task_id = target_cols.index(target_name)
        folds = generate_target_folds_original_indices(X, y[target_name])
        for train_idx, test_idx in folds:
            # Unified training on stacked data within this fold
            X_train = X.loc[train_idx]
            y_train_df = y.loc[train_idx]
            X_stacked, y_stacked = stack_multitask_training_data(X_train, y_train_df[target_cols], target_cols)
            if X_stacked is None or len(y_stacked) == 0:
                continue

            sample_weights = compute_sample_weights_balanced(y_stacked)
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=-1,
                tree_method='hist',
                eval_metric='mlogloss'
            )
            model.fit(X_stacked, y_stacked, sample_weight=sample_weights)

            # Evaluate on this target's test split
            X_test = X.loc[test_idx].copy()
            y_test = y[target_name].loc[test_idx].astype(int).values
            X_test['task_id'] = target_task_id

            proba_test = model.predict_proba(X_test)
            y_pred = np.argmax(proba_test, axis=1)
            high_risk_proba = proba_test[:, 2] + proba_test[:, 3]

            metrics = calculate_comprehensive_metrics(y_test, y_pred, proba_test, high_risk_proba)
            per_target_results[target_name]['metrics_list'].append(metrics)
            per_target_results[target_name]['all_true_values'].append(y_test)
            per_target_results[target_name]['all_predictions'].append(y_pred)
            per_target_results[target_name]['all_probabilities'].append(proba_test)

    for target_name, res in per_target_results.items():
        if res['metrics_list']:
            avg_metrics = {k: float(np.mean([m[k] for m in res['metrics_list']])) for k in res['metrics_list'][0].keys()}
            f1s = [m['f1_macro'] for m in res['metrics_list']]
            avg_metrics['stability_score'] = float(np.mean(f1s) - np.std(f1s))
            res['avg_metrics'] = avg_metrics
        else:
            res['avg_metrics'] = {}
    return per_target_results

def train_individual_multiclass(X: pd.DataFrame, y_target: pd.Series) -> dict:
    """Phase 1 individual per-target model: standard multiclass with class-balanced weights; argmax."""
    folds = generate_target_folds_original_indices(X, y_target)
    res = {'metrics_list': [], 'all_true_values': [], 'all_predictions': [], 'all_probabilities': []}
    for train_idx, test_idx in folds:
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y_target.loc[train_idx], y_target.loc[test_idx]
        y_train_clean = y_train.astype(int).values
        y_test_clean = y_test.astype(int).values

        if len(y_train_clean) == 0 or len(y_test_clean) == 0:
            continue

        sample_weights = compute_sample_weights_balanced(y_train_clean)
        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='mlogloss'
        )
        model.fit(X_train, y_train_clean, sample_weight=sample_weights)

        proba_test = model.predict_proba(X_test)
        y_pred = np.argmax(proba_test, axis=1)
        high_risk_proba = proba_test[:, 2] + proba_test[:, 3]

        metrics = calculate_comprehensive_metrics(y_test_clean, y_pred, proba_test, high_risk_proba)
        res['metrics_list'].append(metrics)
        res['all_true_values'].append(y_test_clean)
        res['all_predictions'].append(y_pred)
        res['all_probabilities'].append(proba_test)

    if res['metrics_list']:
        avg_metrics = {k: float(np.mean([m[k] for m in res['metrics_list']])) for k in res['metrics_list'][0].keys()}
        f1s = [m['f1_macro'] for m in res['metrics_list']]
        avg_metrics['stability_score'] = float(np.mean(f1s) - np.std(f1s))
        res['avg_metrics'] = avg_metrics
    else:
        res['avg_metrics'] = {}
    return res

## Removed: ordinal cumulative unified model for Phase 1 (replaced by multiclass unified model)

def phase1_unified_vs_individual(X: pd.DataFrame, y: pd.DataFrame, target_cols: list) -> dict:
    """Phase 1: Compare unified (single multiclass) vs individual (per-target multiclass) models."""
    print(f"\n📊 PHASE 1: UNIFIED VS INDIVIDUAL MODELS")
    print("-" * 50)
    results = {}
    # Unified multi-task (multiclass; class weights; argmax)
    unified_results = train_unified_multitask_multiclass(X, y, target_cols, n_estimators=N_ESTIMATORS)
    results['unified'] = unified_results
    # Individual per-target models (multiclass; class weights; argmax)
    print("🎯 Training Individual Models (per-target; multiclass; class weights; argmax)...")
    individual_results = {}
    for target_name in target_cols:
        res = train_individual_multiclass(X, y[target_name])
        individual_results[target_name] = res
    results['individual'] = individual_results
    return results

def select_stage1_winners(*args, **kwargs):
    """Deprecated in simplified Step 7."""
    return {}

def select_overall_stage1_winner(unified_results: dict, individual_results: dict) -> dict:
    """Select a single overall Stage 1 winner (Unified vs Individual) using aggregated metrics."""
    print(f"\n🏁 STAGE 1 OVERALL SELECTION (Unified vs Individual)")
    print("-" * 60)

    def valid_targets(results: dict) -> set:
        return {t for t, r in results.items() if isinstance(r, dict) and r.get('avg_metrics')}

    uni_targets = valid_targets(unified_results)
    ind_targets = valid_targets(individual_results)
    common = sorted(list(uni_targets & ind_targets))

    def agg_means(results: dict, targets: list) -> dict:
        if not targets:
            return {'high_risk_recall': 0.0, 'f1_macro': 0.0}
        hr = [results[t]['avg_metrics'].get('high_risk_recall', 0.0) for t in targets if results.get(t, {}).get('avg_metrics')]
        f1 = [results[t]['avg_metrics'].get('f1_macro', 0.0) for t in targets if results.get(t, {}).get('avg_metrics')]
        return {
            'high_risk_recall': float(np.mean(hr)) if hr else 0.0,
            'f1_macro': float(np.mean(f1)) if f1 else 0.0
        }

    if common:
        uni_mean = agg_means(unified_results, common)
        ind_mean = agg_means(individual_results, common)
        # Prefer F1 first, then High-Risk Recall as tie-breaker
        uni_key = (uni_mean['f1_macro'], uni_mean['high_risk_recall'])
        ind_key = (ind_mean['f1_macro'], ind_mean['high_risk_recall'])
        if uni_key >= ind_key:
            winner = {
                'winner': 'unified',
                'means': uni_mean,
                'coverage': {'unified': len(uni_targets), 'individual': len(ind_targets)},
                'targets_compared': common
            }
        else:
            winner = {
                'winner': 'individual',
                'means': ind_mean,
                'coverage': {'unified': len(uni_targets), 'individual': len(ind_targets)},
                'targets_compared': common
            }
    else:
        # No common coverage: pick by larger coverage; tie-break by internal means
        if len(uni_targets) > len(ind_targets):
            winner = {
                'winner': 'unified',
                'means': agg_means(unified_results, list(uni_targets)),
                'coverage': {'unified': len(uni_targets), 'individual': len(ind_targets)},
                'targets_compared': []
            }
        elif len(ind_targets) > len(uni_targets):
            winner = {
                'winner': 'individual',
                'means': agg_means(individual_results, list(ind_targets)),
                'coverage': {'unified': len(uni_targets), 'individual': len(ind_targets)},
                'targets_compared': []
            }
        else:
            # Equal coverage: compare their means over their own targets
            uni_mean = agg_means(unified_results, list(uni_targets))
            ind_mean = agg_means(individual_results, list(ind_targets))
            # Prefer F1 first, then High-Risk Recall as tie-breaker
            uni_key = (uni_mean['f1_macro'], uni_mean['high_risk_recall'])
            ind_key = (ind_mean['f1_macro'], ind_mean['high_risk_recall'])
            if uni_key >= ind_key:
                winner = {
                    'winner': 'unified',
                    'means': uni_mean,
                    'coverage': {'unified': len(uni_targets), 'individual': len(ind_targets)},
                    'targets_compared': []
                }
            else:
                winner = {
                    'winner': 'individual',
                    'means': ind_mean,
                    'coverage': {'unified': len(uni_targets), 'individual': len(ind_targets)},
                    'targets_compared': []
                }

    print(f"   Overall winner: {winner['winner']} (mean HR-Recall={winner['means']['high_risk_recall']:.4f}, mean F1={winner['means']['f1_macro']:.4f})")
    print(f"   Coverage: unified={winner['coverage']['unified']}, individual={winner['coverage']['individual']}; compared on {len(winner['targets_compared'])} targets")
    return winner

# Removed unused comparisons to streamline per-user request

def save_step6_results(phase1_results, phase2_results, final_selection):
    """Save comprehensive Step 6 results (simplified)."""
    print(f"\n💾 Saving Step 6 Results")
    print("-" * 30)
    
    results_dir = '../results/step6_model_architecture'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    all_results = {
        'phase1_unified_vs_individual': phase1_results,
        'phase2_best_vs_ordinal': phase2_results,
        'final_selection': final_selection,
        'execution_date': datetime.now().isoformat()
    }
    
    with open(f'{results_dir}/step6_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    # Create summary
    # Gather dynamic values for summary
    feature_count = None
    try:
        # Try to infer feature count from phase1 unified block (first available target metrics structure)
        # If not available, fallback to None
        feature_count = None
    except Exception:
        feature_count = None

    summary = {
        'execution_date': datetime.now().isoformat(),
        'approach': 'model_architecture_experiments_two_phase',
        'phases_tested': 2,
        'selection_policy': 'Stage1: Unified vs Individual; Stage2: Best-of-Stage1 vs Ordinal score + fixed cutpoints',
        'final_selection': final_selection,
        'key_components': {
            'step2_features': 'Optimized features',
            'step3_validation': 'Rolling Window CV',
            'step5_handling': 'Class-balanced weights (argmax) for Phase 1',
            'ordinal_process': 'Single regressor with training-only OOF cutpoints; interpretable'
        },
        # Note: For ordinal, probability-based metrics may be unavailable
        'evaluation_metrics': ['high_risk_recall', 'f1_macro', 'balanced_accuracy', 'cohen_kappa', 'high_risk_pr_auc', 'stability']
    }
    
    with open(f'{results_dir}/step6_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ Results saved to: {results_dir}")
    return results_dir

def create_stage_visualizations(phase1_results: dict, phase2_results: dict, results_dir: str) -> None:
    """Create Stage 1 and Stage 2 comparison visualizations."""
    print(f"\n📊 Creating Stage 1 and Stage 2 Visualizations")
    print("-" * 50)

    # Stage 1 comparison: Unified vs Individual per target
    rows = []
    unified = phase1_results.get('unified', {}) or {}
    individual = phase1_results.get('individual', {}) or {}
    targets = sorted(set(list(unified.keys()) + list(individual.keys())))
    for t in targets:
        um = (unified.get(t) or {}).get('avg_metrics', {})
        im = (individual.get(t) or {}).get('avg_metrics', {})
        if um:
            rows.append({'target': t, 'approach': 'Unified', 'f1_macro': um.get('f1_macro', 0.0), 'high_risk_recall': um.get('high_risk_recall', 0.0), 'balanced_accuracy': um.get('balanced_accuracy', 0.0)})
        if im:
            rows.append({'target': t, 'approach': 'Individual', 'f1_macro': im.get('f1_macro', 0.0), 'high_risk_recall': im.get('high_risk_recall', 0.0), 'balanced_accuracy': im.get('balanced_accuracy', 0.0)})

    if rows:
        df1 = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.barplot(data=df1, x='target', y='high_risk_recall', hue='approach', ax=axes[0])
        axes[0].set_title('Stage 1: High-Risk Recall')
        axes[0].set_xlabel('Target')
        axes[0].set_ylabel('High-Risk Recall')
        axes[0].grid(True, alpha=0.3)
        sns.barplot(data=df1, x='target', y='f1_macro', hue='approach', ax=axes[1])
        axes[1].set_title('Stage 1: F1-Macro')
        axes[1].set_xlabel('Target')
        axes[1].set_ylabel('F1-Macro')
        axes[1].grid(True, alpha=0.3)
        sns.barplot(data=df1, x='target', y='balanced_accuracy', hue='approach', ax=axes[2])
        axes[2].set_title('Stage 1: Balanced Accuracy')
        axes[2].set_xlabel('Target')
        axes[2].set_ylabel('Balanced Accuracy')
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'stage1_unified_vs_individual.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("⚠️ No Stage 1 data to plot")

    # Stage 2 comparison: Best-of-Stage1 vs Ordinal per target
    rows2 = []
    for t, pair in phase2_results.items():
        base = pair.get('best_of_stage1', {}) or {}
        ordn = pair.get('ordinal_score_cutpoints', {}) or {}
        bm = base.get('avg_metrics', {})
        om = ordn.get('avg_metrics', {})
        if bm:
            rows2.append({'target': t, 'approach': 'Best-of-Stage1', 'f1_macro': bm.get('f1_macro', 0.0), 'high_risk_recall': bm.get('high_risk_recall', 0.0), 'balanced_accuracy': bm.get('balanced_accuracy', 0.0)})
        if om:
            rows2.append({'target': t, 'approach': 'Ordinal', 'f1_macro': om.get('f1_macro', 0.0), 'high_risk_recall': om.get('high_risk_recall', 0.0), 'balanced_accuracy': om.get('balanced_accuracy', 0.0)})
    if rows2:
        df2 = pd.DataFrame(rows2)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.barplot(data=df2, x='target', y='high_risk_recall', hue='approach', ax=axes[0])
        axes[0].set_title('Stage 2: High-Risk Recall')
        axes[0].set_xlabel('Target')
        axes[0].set_ylabel('High-Risk Recall')
        axes[0].grid(True, alpha=0.3)
        sns.barplot(data=df2, x='target', y='f1_macro', hue='approach', ax=axes[1])
        axes[1].set_title('Stage 2: F1-Macro')
        axes[1].set_xlabel('Target')
        axes[1].set_ylabel('F1-Macro')
        axes[1].grid(True, alpha=0.3)
        sns.barplot(data=df2, x='target', y='balanced_accuracy', hue='approach', ax=axes[2])
        axes[2].set_title('Stage 2: Balanced Accuracy')
        axes[2].set_xlabel('Target')
        axes[2].set_ylabel('Balanced Accuracy')
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'stage2_best_vs_ordinal.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("⚠️ No Stage 2 data to plot")

def main():
    """Main execution - Step 6 (Two-Phase): Unified vs Individual; Best-of-Stage1 vs Ordinal"""
    print("🚀 STEP 6: MODEL ARCHITECTURE (TWO-PHASE)")
    print("=" * 60)
    
    # Load Step 4 data
    df, X, y, exclude_cols, target_cols = load_selected_data()
    
    print(f"📊 Testing Phase 1 and Phase 2 across {len(target_cols)} targets")
    print(f"🎯 Targets: {target_cols}")
    print(f"📈 Features: {X.shape[1]}")
    print(f"📅 Using Rolling Window CV (Step 3)")
    print(f"⚖️ Phase 1 uses Step 3 handling (class weights; argmax)")
    print(f"📏 Phase 2 Ordinal: single regressor + fixed cutpoints (training-only OOF)")
    
    # Phase 1: Unified vs Individual
    phase1_results = phase1_unified_vs_individual(X, y, target_cols)
    overall_stage1 = select_overall_stage1_winner(phase1_results.get('unified', {}), phase1_results.get('individual', {}))

    # Phase 2: Best-of-Stage1 vs Ordinal per target
    phase2_results = {}
    best_kind = overall_stage1.get('winner')
    for t in target_cols:
        print(f"\n🎯 Phase 2 for {t}: Best-of-Stage1 ({best_kind}) vs Ordinal")
        # Use the stored winner’s per-target result; if missing, fall back to the other approach
        if best_kind == 'unified':
            best_struct = (phase1_results.get('unified', {}) or {}).get(t, {})
            if not best_struct or not best_struct.get('avg_metrics'):
                # Fallback to individual for this target
                best_struct = (phase1_results.get('individual', {}) or {}).get(t, {})
        else:
            best_struct = (phase1_results.get('individual', {}) or {}).get(t, {})
            if not best_struct or not best_struct.get('avg_metrics'):
                # Fallback to unified for this target
                best_struct = (phase1_results.get('unified', {}) or {}).get(t, {})
        # Train ordinal on this target
        mask = ~pd.isna(y[t])
        base_X, base_y = X.loc[mask], y[t].loc[mask]
        ordn = train_ordinal_score_with_cutpoints(base_X, base_y)
        phase2_results[t] = {
            'best_of_stage1': best_struct,
            'ordinal_score_cutpoints': ordn
        }

    # Final selection per target (compare best-of-stage1 vs ordinal)
    final_selection = {}
    for t, pair in phase2_results.items():
        bm = (pair.get('best_of_stage1') or {}).get('avg_metrics', {})
        om = (pair.get('ordinal_score_cutpoints') or {}).get('avg_metrics', {})
        # Prefer F1 first, then High-Risk Recall as tie-breaker
        base_key = (bm.get('f1_macro', 0.0), bm.get('high_risk_recall', 0.0))
        ord_key = (om.get('f1_macro', 0.0), om.get('high_risk_recall', 0.0))
        final_selection[t] = {'winner': 'ordinal' if ord_key > base_key else 'best_of_stage1', 'metrics': om if ord_key > base_key else bm}

    # Save results & visuals
    results_dir = save_step6_results(phase1_results, phase2_results, final_selection)
    create_stage_visualizations(phase1_results, phase2_results, results_dir)
    
    print(f"\n🎉 STEP 6 COMPLETED!")
    print("=" * 60)
    print("✅ Phase 1: Unified vs Individual evaluated")
    print("✅ Phase 2: Best-of-Stage1 vs Ordinal evaluated")
    print("🏆 Final per-target winners:")
    for t, info in final_selection.items():
        print(f"   {t}: {info['winner']} (HR-Recall={info['metrics'].get('high_risk_recall',0):.4f}, F1={info['metrics'].get('f1_macro',0):.4f})")
    return final_selection

if __name__ == "__main__":
    best_architecture = main()
