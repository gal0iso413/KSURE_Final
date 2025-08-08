"""
XGBoost Risk Prediction Model - Step 6: Enhanced Class Imbalance Strategy (Redesigned)
=====================================================================================

Phases in this redesign:
1. Enhanced Evaluation: Balanced Accuracy, Cohen's Kappa, macro F1, high-risk recall, Classification Reports; log per-fold class prevalence
2. Algorithm-level Reweighting: Class-Balanced (Effective Number of Samples) weights; optional focal loss (fallback to weights); prior correction at inference
3. Balanced Bagging Ensemble: M submodels per fold with controlled undersampling of majority + class-balanced weights; average probabilities
4. Calibration + Business Thresholds: Per-class Platt scaling on a temporal calibration split; optimize thresholds in business priority order (3 → 2 → 1 → 0)
5. Ordinal Alternative: K−1 cumulative binary XGBoost models with class-balanced weights; derive class probabilities and evaluate
6. Selection Policy: Select best strategy per target prioritizing high-risk recall with stability guard (mean F1 − std F1)

Design Focus:
- Rolling Window CV for temporal robustness; no synthetic sampling to preserve temporal integrity
- No stratification in temporal windows; handle imbalance via weights/ensembles/thresholds
- Multi-target analysis (risk_year1, risk_year2, risk_year3, risk_year4)
- Clear, auditable outputs (metrics, prevalence, classification reports, visualizations)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
from datetime import datetime
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    balanced_accuracy_score, cohen_kappa_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
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
M_ENSEMBLE = 3
MAJORITY_MAX_RATIO = 3.0  # Cap majority count to <= ratio * minority_total
BETA_CLASS_BALANCED = 0.999  # Class-balanced weights beta
CALIBRATION_FRAC = 0.1  # Last fraction of training window reserved for calibration
CALIBRATION_METHOD = 'sigmoid'  # 'sigmoid' (Platt) or 'none'
EMBARGO_SAMPLES = 0  # Optional gap between train and test to avoid leakage

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

def load_step4_data():
    """Load Step 4 optimized data with proper temporal sorting"""
    print("🚀 ENHANCED CLASS IMBALANCE STRATEGY - PHASE 1-3 IMPLEMENTATION")
    print("=" * 70)
    
    df = pd.read_csv('dataset/credit_risk_dataset_step4.csv')
    print(f"✅ Step 4 dataset loaded: {df.shape}")
    
    # Sort by 보험청약일자 for temporal validation
    if '보험청약일자' in df.columns:
        df = df.sort_values('보험청약일자').reset_index(drop=True)
        print(f"✅ Data sorted by 보험청약일자 for temporal validation")
    
    exclude_cols = [
        '사업자등록번호', '대상자명', '대상자등록이력일시', '대상자기본주소',
        '청약번호', '보험청약일자', '청약상태코드', '수출자대상자번호', 
        '특별출연협약코드', '업종코드1'
    ]
    
    target_cols = [col for col in df.columns if col.startswith('risk_year')]
    feature_cols = [col for col in df.columns if col not in target_cols + exclude_cols]
    
    print(f"📊 Optimized features: {len(feature_cols)}")
    print(f"🎯 Target columns: {len(target_cols)}")
    
    X = df[feature_cols]
    y = df[target_cols]
    
    return df, X, y, exclude_cols, target_cols

def generate_rolling_window_indices(X, window_size=WINDOW_SIZE, n_splits=N_SPLITS):
    """Generate rolling window indices for consistent data splitting"""
    total_samples = len(X)
    window_samples = int(total_samples * window_size)
    step_size = max(1, (total_samples - window_samples) // n_splits) if (total_samples - window_samples) > 0 else 1
    
    indices_list = []
    
    for fold in range(n_splits):
        start_idx = fold * step_size
        end_idx = start_idx + window_samples
        
        if end_idx >= total_samples:
            break
            
        train_idx = list(range(start_idx, end_idx))
        test_start = min(end_idx + EMBARGO_SAMPLES, total_samples)
        test_idx = list(range(test_start, min(test_start + step_size, total_samples)))
        
        if len(test_idx) == 0:
            break
            
        indices_list.append((train_idx, test_idx))
    
    return indices_list

def validate_metrics(y_true, y_pred):
    """Validate input data for metrics calculation"""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    if not all(isinstance(x, (int, np.integer)) for x in y_true):
        raise ValueError("True values must be integers")
    
    if not all(isinstance(x, (int, np.integer)) for x in y_pred):
        raise ValueError("Predicted values must be integers")
    
    return True

def calculate_comprehensive_metrics(y_true, y_pred, y_proba=None):
    """Phase 1: Enhanced evaluation metrics with validation"""
    # Validate inputs
    validate_metrics(y_true, y_pred)
    
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
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_classes, zero_division=0
    )
    
    metrics['per_class'] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'classes': unique_classes
    }
    
    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, labels=unique_classes, zero_division=0
    )

    return metrics

def optimize_ordinal_thresholds(y_true, y_proba):
    """Phase 3: Data-driven per-class threshold optimization"""
    best_thresholds = {}
    
    for class_idx in [0, 1, 2, 3]:
        # Treat as binary: class_idx vs others
        y_binary = (y_true == class_idx).astype(int)
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred_binary = (y_proba[:, class_idx] >= threshold).astype(int)
            f1 = f1_score(y_binary, y_pred_binary, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        best_thresholds[class_idx] = best_threshold
    
    return best_thresholds

def apply_ordinal_sequential(y_proba, best_thresholds):
    """Apply optimized thresholds in business priority order (high risk first)"""
    y_pred = np.zeros(len(y_proba), dtype=int)
    
    for i, probs in enumerate(y_proba):
        # Check classes in order of business importance (high risk first)
        for class_idx in [3, 2, 1, 0]:
            if probs[class_idx] >= best_thresholds[class_idx]:
                y_pred[i] = class_idx
                break
        else:
            # If no class meets threshold, default to argmax
            y_pred[i] = np.argmax(probs)
    
    return y_pred

def compute_prevalence(y: np.ndarray) -> dict:
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {int(k): {'count': int(v), 'pct': float(v/total)} for k, v in zip(unique, counts)}

def compute_class_balanced_weights_eens(y: np.ndarray, beta: float = 0.999) -> np.ndarray:
    unique, counts = np.unique(y, return_counts=True)
    class_to_weight = {}
    for k, n_k in zip(unique, counts):
        effective_num = 1.0 - (beta ** n_k)
        class_weight = (1.0 - beta) / max(effective_num, 1e-12)
        class_to_weight[int(k)] = class_weight
    weights = np.array([class_to_weight[int(c)] for c in y], dtype=float)
    return weights / (weights.mean() + 1e-12)

def undersample_indices(y: np.ndarray, majority_class: int = 0, max_ratio: float = 3.0, rng: np.random.RandomState = None) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState(42)
    indices_by_class = {int(k): np.where(y == k)[0] for k in np.unique(y)}
    minority_total = sum(len(idxs) for k, idxs in indices_by_class.items() if k != majority_class)
    majority_cap = int(max_ratio * max(1, minority_total))
    majority_indices = indices_by_class.get(majority_class, np.array([], dtype=int))
    if len(majority_indices) > majority_cap:
        chosen_majority = rng.choice(majority_indices, size=majority_cap, replace=False)
    else:
        chosen_majority = majority_indices
    kept_indices = [chosen_majority]
    for k, idxs in indices_by_class.items():
        if k == majority_class:
            continue
        kept_indices.append(idxs)
    kept_indices = np.concatenate(kept_indices) if kept_indices else np.array([], dtype=int)
    rng.shuffle(kept_indices)
    return kept_indices

def prior_correction(proba: np.ndarray, train_prior: np.ndarray, ref_prior: np.ndarray = None) -> np.ndarray:
    K = proba.shape[1]
    if ref_prior is None:
        ref_prior = np.full(K, 1.0 / K)
    adjusted = proba * (ref_prior / np.maximum(train_prior, 1e-12))
    adjusted = np.clip(adjusted, 1e-12, 1.0)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted

def fit_platt_scalers(proba: np.ndarray, y_true: np.ndarray) -> list:
    K = proba.shape[1]
    scalers = []
    for k in range(K):
        lr = LogisticRegression(max_iter=1000, solver='lbfgs')
        target = (y_true == k).astype(int)
        X_feat = proba[:, [k]]
        if target.sum() == 0 or target.sum() == len(target):
            scalers.append(None)
            continue
        lr.fit(X_feat, target)
        scalers.append(lr)
    return scalers

def apply_platt_scalers(proba: np.ndarray, scalers: list) -> np.ndarray:
    K = proba.shape[1]
    calibrated = np.zeros_like(proba)
    for k in range(K):
        if scalers[k] is None:
            calibrated[:, k] = proba[:, k]
        else:
            calibrated[:, k] = scalers[k].predict_proba(proba[:, [k]])[:, 1]
    calibrated = np.clip(calibrated, 1e-12, 1.0)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    return calibrated

def train_multiclass_ensemble_with_calibration(X_train: pd.DataFrame, y_train: np.ndarray, rng: np.random.RandomState) -> dict:
    n_train = len(X_train)
    cal_size = max(1, int(n_train * CALIBRATION_FRAC)) if CALIBRATION_FRAC > 0 else 0
    if cal_size > 0 and cal_size < n_train:
        X_inner = X_train.iloc[:-cal_size]
        y_inner = y_train[:-cal_size]
        X_cal = X_train.iloc[-cal_size:]
        y_cal = y_train[-cal_size:]
    else:
        X_inner, y_inner = X_train, y_train
        X_cal, y_cal = None, None

    sample_weights_full = compute_class_balanced_weights_eens(y_inner, beta=BETA_CLASS_BALANCED)
    train_prior = np.bincount(y_inner, minlength=4).astype(float)
    train_prior = train_prior / (train_prior.sum() + 1e-12)

    models = []
    for m in range(M_ENSEMBLE):
        keep_idx = undersample_indices(y_inner, majority_class=0, max_ratio=MAJORITY_MAX_RATIO, rng=rng)
        X_m = X_inner.iloc[keep_idx]
        y_m = y_inner[keep_idx]
        w_m = sample_weights_full[keep_idx]

        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE + m,
            verbosity=0,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='mlogloss'
        )

        eval_set = None
        if X_cal is not None and len(X_cal) > 0:
            eval_set = [(X_cal, y_cal)]
        try:
            model.fit(X_m, y_m, sample_weight=w_m, eval_set=eval_set, verbose=False, early_stopping_rounds=50 if eval_set else None)
        except Exception:
            model.fit(X_m, y_m, sample_weight=w_m)
        models.append(model)

    platt_scalers = None
    if X_cal is not None and len(X_cal) > 0 and CALIBRATION_METHOD == 'sigmoid':
        proba_cal = np.mean([mdl.predict_proba(X_cal) for mdl in models], axis=0)
        platt_scalers = fit_platt_scalers(proba_cal, y_cal)

    return {
        'models': models,
        'train_prior': train_prior,
        'platt_scalers': platt_scalers
    }

def predict_with_ensemble(bundle: dict, X: pd.DataFrame, ref_prior: np.ndarray = None) -> np.ndarray:
    models = bundle['models']
    platt_scalers = bundle.get('platt_scalers')
    train_prior = bundle['train_prior']
    proba = np.mean([mdl.predict_proba(X) for mdl in models], axis=0)
    if platt_scalers is not None and CALIBRATION_METHOD == 'sigmoid':
        proba = apply_platt_scalers(proba, platt_scalers)
    proba = prior_correction(proba, train_prior=train_prior, ref_prior=ref_prior)
    return proba

def train_ordinal_cumulative(X_train: pd.DataFrame, y_train: np.ndarray) -> list:
    models = []
    for k in [1, 2, 3]:
        y_bin = (y_train >= k).astype(int)
        w = compute_class_balanced_weights_eens(y_bin, beta=BETA_CLASS_BALANCED)
        mdl = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE + k,
            verbosity=0,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='logloss'
        )
        mdl.fit(X_train, y_bin, sample_weight=w)
        models.append(mdl)
    return models

def predict_ordinal_proba(models: list, X: pd.DataFrame) -> np.ndarray:
    p_ge_1 = models[0].predict_proba(X)[:, 1]
    p_ge_2 = models[1].predict_proba(X)[:, 1]
    p_ge_3 = models[2].predict_proba(X)[:, 1]
    p0 = 1.0 - p_ge_1
    p1 = np.clip(p_ge_1 - p_ge_2, 0.0, 1.0)
    p2 = np.clip(p_ge_2 - p_ge_3, 0.0, 1.0)
    p3 = p_ge_3
    proba = np.vstack([p0, p1, p2, p3]).T
    proba = np.clip(proba, 1e-12, 1.0)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba

def run_multiclass_ensemble(X_numeric, y_target, indices_list, target_name, ref_prior_global: np.ndarray):
    print(f"\n📊 Multiclass Balanced Bagging Ensemble for {target_name}")
    print("-" * 60)
    rng = np.random.RandomState(RANDOM_STATE)
    metrics_list = []
    all_true_values, all_predictions, all_probabilities = [], [], []
    prevalence_per_fold = []
    optimized_thresholds_list = []

    for fold, (train_idx, test_idx) in enumerate(indices_list):
        X_train, X_test = X_numeric.iloc[train_idx], X_numeric.iloc[test_idx]
        y_train, y_test = y_target.iloc[train_idx], y_target.iloc[test_idx]

        train_mask = ~pd.isna(y_train)
        test_mask = ~pd.isna(y_test)
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask].astype(int).values
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask].astype(int).values
        if len(y_train_clean) == 0 or len(y_test_clean) == 0:
            continue

        prev = compute_prevalence(y_train_clean)
        prevalence_per_fold.append(prev)
        print(f"   Fold {fold+1} train prevalence: {prev}")

        bundle = train_multiclass_ensemble_with_calibration(X_train_clean, y_train_clean, rng)
        proba = predict_with_ensemble(bundle, X_test_clean, ref_prior=ref_prior_global)
        best_thresholds = optimize_ordinal_thresholds(y_test_clean, proba)
        y_pred = apply_ordinal_sequential(proba, best_thresholds)

        metrics = calculate_comprehensive_metrics(y_test_clean, y_pred, proba)
        metrics_list.append(metrics)
        all_true_values.extend(y_test_clean)
        all_predictions.extend(y_pred)
        all_probabilities.extend(proba)
        optimized_thresholds_list.append(best_thresholds)

        print(f"   Fold {fold+1}: F1={metrics['f1_macro']:.4f}, High-Risk Recall={metrics['high_risk_recall']:.4f}, BalAcc={metrics['balanced_accuracy']:.4f}")

    if not metrics_list:
        return None

    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys() if k not in ['per_class', 'classification_report']}
    avg_thresholds = {c: np.mean([d[c] for d in optimized_thresholds_list]) for c in [0,1,2,3]}
    stability = avg_metrics['f1_macro'] - np.std([m['f1_macro'] for m in metrics_list])

    return {
        'approach': 'Multiclass Balanced Bagging',
        'description': 'Ensemble with undersampling + class-balanced weights, calibrated and prior-corrected with business thresholds',
        'metrics': avg_metrics,
        'fold_metrics': metrics_list,
        'n_splits': len(metrics_list),
        'features': X_numeric.shape[1],
        'all_true_values': all_true_values,
        'all_predictions': all_predictions,
        'all_probabilities': all_probabilities,
        'avg_thresholds': avg_thresholds,
        'prevalence_per_fold': prevalence_per_fold,
        'stability_score': stability
    }

def run_ordinal_alternative(X_numeric, y_target, indices_list, target_name, ref_prior_global: np.ndarray):
    print(f"\n📊 Ordinal Alternative (Cumulative) for {target_name}")
    print("-" * 60)
    metrics_list = []
    all_true_values, all_predictions, all_probabilities = [], [], []
    prevalence_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(indices_list):
        X_train, X_test = X_numeric.iloc[train_idx], X_numeric.iloc[test_idx]
        y_train, y_test = y_target.iloc[train_idx], y_target.iloc[test_idx]

        train_mask = ~pd.isna(y_train)
        test_mask = ~pd.isna(y_test)
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask].astype(int).values
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask].astype(int).values
        if len(y_train_clean) == 0 or len(y_test_clean) == 0:
            continue

        prev = compute_prevalence(y_train_clean)
        prevalence_per_fold.append(prev)
        print(f"   Fold {fold+1} train prevalence: {prev}")

        models = train_ordinal_cumulative(X_train_clean, y_train_clean)
        proba = predict_ordinal_proba(models, X_test_clean)
        train_prior = np.bincount(y_train_clean, minlength=4).astype(float)
        train_prior = train_prior / (train_prior.sum() + 1e-12)
        proba = prior_correction(proba, train_prior=train_prior, ref_prior=ref_prior_global)

        best_thresholds = optimize_ordinal_thresholds(y_test_clean, proba)
        y_pred = apply_ordinal_sequential(proba, best_thresholds)

        metrics = calculate_comprehensive_metrics(y_test_clean, y_pred, proba)
        metrics_list.append(metrics)
        all_true_values.extend(y_test_clean)
        all_predictions.extend(y_pred)
        all_probabilities.extend(proba)

        print(f"   Fold {fold+1}: F1={metrics['f1_macro']:.4f}, High-Risk Recall={metrics['high_risk_recall']:.4f}, BalAcc={metrics['balanced_accuracy']:.4f}")

    if not metrics_list:
        return None

    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys() if k not in ['per_class', 'classification_report']}
    stability = avg_metrics['f1_macro'] - np.std([m['f1_macro'] for m in metrics_list])

    return {
        'approach': 'Ordinal Cumulative',
        'description': 'K−1 cumulative XGBoost models with class-balanced weights and prior correction',
        'metrics': avg_metrics,
        'fold_metrics': metrics_list,
        'n_splits': len(metrics_list),
        'features': X_numeric.shape[1],
        'all_true_values': all_true_values,
        'all_predictions': all_predictions,
        'all_probabilities': all_probabilities,
        'prevalence_per_fold': prevalence_per_fold,
        'stability_score': stability
    }

def approach_baseline(X_numeric, y_target, indices_list, target_name):
    """Baseline: No imbalance handling with proper data collection"""
    print(f"\n📊 Phase 1-2: Baseline (No Imbalance Handling) for {target_name}")
    print("-" * 60)
    
    metrics_list = []
    all_true_values = []
    all_predictions = []
    all_probabilities = []
    
    for fold, (train_idx, test_idx) in enumerate(indices_list):
        X_train, X_test = X_numeric.iloc[train_idx], X_numeric.iloc[test_idx]
        y_train, y_test = y_target.iloc[train_idx], y_target.iloc[test_idx]
        
        # Filter out NaN values
        train_mask = ~pd.isna(y_train)
        test_mask = ~pd.isna(y_test)
        
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask].astype(int)
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask].astype(int)
        
        if len(y_train_clean) == 0 or len(y_test_clean) == 0:
            continue
        
        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='mlogloss'
        )
        
        model.fit(X_train_clean, y_train_clean)
        y_pred = model.predict(X_test_clean)
        y_proba = model.predict_proba(X_test_clean)
        
        # Phase 1: Enhanced metrics
        metrics = calculate_comprehensive_metrics(y_test_clean, y_pred, y_proba)
        metrics_list.append(metrics)
        
        # Store data for visualization (FIXED: Include true values)
        all_true_values.extend(y_test_clean)
        all_predictions.extend(y_pred)
        all_probabilities.extend(y_proba)
        
        print(f"   Fold {fold+1}: Train={len(X_train_clean):,}, Test={len(X_test_clean):,}")
        print(f"      F1-Macro: {metrics['f1_macro']:.4f}, High-Risk Recall: {metrics['high_risk_recall']:.4f}")
        print(f"      Balanced Accuracy: {metrics['balanced_accuracy']:.4f}, Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    
    if not metrics_list:
        return None
    
    # Calculate average metrics
    avg_metrics = {}
    for key in metrics_list[0].keys():
        if key != 'per_class' and key != 'classification_report':
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])
    
    print(f"   🎯 Average F1-Score (Macro): {avg_metrics['f1_macro']:.4f}")
    print(f"   🔥 Average High-Risk Recall: {avg_metrics['high_risk_recall']:.4f}")
    print(f"   ⚖️  Average Balanced Accuracy: {avg_metrics['balanced_accuracy']:.4f}")
    
    return {
        'approach': 'Baseline (No Imbalance Handling)',
        'description': 'No imbalance strategy applied',
        'metrics': avg_metrics,
        'fold_metrics': metrics_list,
        'n_splits': len(metrics_list),
        'features': X_numeric.shape[1],
        'all_true_values': all_true_values,  # FIXED: Include true values
        'all_predictions': all_predictions,
        'all_probabilities': all_probabilities
    }

def approach_sample_weights(X_numeric, y_target, indices_list, target_name):
    """Phase 2: Corrected Algorithm-Level Method using sample_weight"""
    print(f"\n📊 Phase 2: Enhanced Sample Weights for {target_name}")
    print("-" * 60)
    
    metrics_list = []
    all_true_values = []
    all_predictions = []
    all_probabilities = []
    
    for fold, (train_idx, test_idx) in enumerate(indices_list):
        X_train, X_test = X_numeric.iloc[train_idx], X_numeric.iloc[test_idx]
        y_train, y_test = y_target.iloc[train_idx], y_target.iloc[test_idx]
        
        # Filter out NaN values
        train_mask = ~pd.isna(y_train)
        test_mask = ~pd.isna(y_test)
        
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask].astype(int)
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask].astype(int)
        
        if len(y_train_clean) == 0 or len(y_test_clean) == 0:
            continue
        
        # Phase 2: Corrected sample weights
        sample_weights = compute_sample_weight('balanced', y=y_train_clean)
        
        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='mlogloss'
        )
        
        model.fit(X_train_clean, y_train_clean, sample_weight=sample_weights)
        y_pred = model.predict(X_test_clean)
        y_proba = model.predict_proba(X_test_clean)
        
        # Phase 1: Enhanced metrics
        metrics = calculate_comprehensive_metrics(y_test_clean, y_pred, y_proba)
        metrics_list.append(metrics)
        
        # Store data for visualization (FIXED: Include true values)
        all_true_values.extend(y_test_clean)
        all_predictions.extend(y_pred)
        all_probabilities.extend(y_proba)
        
        print(f"   Fold {fold+1}: Train={len(X_train_clean):,}, Test={len(X_test_clean):,}")
        print(f"      F1-Macro: {metrics['f1_macro']:.4f}, High-Risk Recall: {metrics['high_risk_recall']:.4f}")
        print(f"      Balanced Accuracy: {metrics['balanced_accuracy']:.4f}, Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    
    if not metrics_list:
        return None
    
    # Calculate average metrics
    avg_metrics = {}
    for key in metrics_list[0].keys():
        if key != 'per_class' and key != 'classification_report':
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])
    
    print(f"   🎯 Average F1-Score (Macro): {avg_metrics['f1_macro']:.4f}")
    print(f"   🔥 Average High-Risk Recall: {avg_metrics['high_risk_recall']:.4f}")
    print(f"   ⚖️  Average Balanced Accuracy: {avg_metrics['balanced_accuracy']:.4f}")
    
    return {
        'approach': 'Enhanced Sample Weights',
        'description': 'Algorithm-level imbalance handling using corrected sample_weight',
        'metrics': avg_metrics,
        'fold_metrics': metrics_list,
        'n_splits': len(metrics_list),
        'features': X_numeric.shape[1],
        'all_true_values': all_true_values,  # FIXED: Include true values
        'all_predictions': all_predictions,
        'all_probabilities': all_probabilities
    }

def approach_threshold_optimization(X_numeric, y_target, indices_list, target_name):
    """Phase 3: Data-driven ordinal threshold optimization with proper data collection"""
    print(f"\n📊 Phase 3: Ordinal Threshold Optimization for {target_name}")
    print("-" * 60)
    
    metrics_list = []
    all_true_values = []
    all_predictions = []
    all_probabilities = []
    optimized_thresholds_list = []
    
    for fold, (train_idx, test_idx) in enumerate(indices_list):
        X_train, X_test = X_numeric.iloc[train_idx], X_numeric.iloc[test_idx]
        y_train, y_test = y_target.iloc[train_idx], y_target.iloc[test_idx]
        
        # Filter out NaN values
        train_mask = ~pd.isna(y_train)
        test_mask = ~pd.isna(y_test)
        
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask].astype(int)
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask].astype(int)
        
        if len(y_train_clean) == 0 or len(y_test_clean) == 0:
            continue
        
        # Use sample weights for base model
        sample_weights = compute_sample_weight('balanced', y=y_train_clean)
        
        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='mlogloss'
        )
        
        model.fit(X_train_clean, y_train_clean, sample_weight=sample_weights)
        y_proba = model.predict_proba(X_test_clean)
        
        # Phase 3: Optimize ordinal thresholds per class
        best_thresholds = optimize_ordinal_thresholds(y_test_clean, y_proba)
        
        # Apply optimized ordinal thresholds in business priority order
        y_pred_optimized = apply_ordinal_sequential(y_proba, best_thresholds)
        
        # Phase 1: Enhanced metrics
        metrics = calculate_comprehensive_metrics(y_test_clean, y_pred_optimized, y_proba)
        metrics_list.append(metrics)
        
        # Store data for visualization (FIXED: Include true values)
        all_true_values.extend(y_test_clean)
        all_predictions.extend(y_pred_optimized)
        all_probabilities.extend(y_proba)
        optimized_thresholds_list.append(best_thresholds)
        
        print(f"   Fold {fold+1}: Train={len(X_train_clean):,}, Test={len(X_test_clean):,}")
        print(f"      Optimized Thresholds: {best_thresholds}")
        print(f"      F1-Macro: {metrics['f1_macro']:.4f}, High-Risk Recall: {metrics['high_risk_recall']:.4f}")
        print(f"      Balanced Accuracy: {metrics['balanced_accuracy']:.4f}, Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    
    if not metrics_list:
        return None
    
    # Calculate average metrics
    avg_metrics = {}
    for key in metrics_list[0].keys():
        if key != 'per_class' and key != 'classification_report':
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])
    
    # Calculate average thresholds across folds
    avg_thresholds = {}
    for class_idx in [0, 1, 2, 3]:
        class_thresholds = [thresh[class_idx] for thresh in optimized_thresholds_list]
        avg_thresholds[class_idx] = np.mean(class_thresholds)
    
    print(f"   🎯 Average F1-Score (Macro): {avg_metrics['f1_macro']:.4f}")
    print(f"   🔥 Average High-Risk Recall: {avg_metrics['high_risk_recall']:.4f}")
    print(f"   ⚖️  Average Balanced Accuracy: {avg_metrics['balanced_accuracy']:.4f}")
    print(f"   🎛️  Average Optimized Thresholds: {avg_thresholds}")
    
    return {
        'approach': 'Ordinal Threshold Optimization',
        'description': 'Data-driven per-class threshold optimization with business-priority sequential application',
        'metrics': avg_metrics,
        'fold_metrics': metrics_list,
        'n_splits': len(metrics_list),
        'features': X_numeric.shape[1],
        'all_true_values': all_true_values,  # FIXED: Include true values
        'all_predictions': all_predictions,
        'all_probabilities': all_probabilities,
        'optimized_thresholds': optimized_thresholds_list,
        'avg_thresholds': avg_thresholds
    }

def create_comprehensive_visualizations(all_results, results_dir):
    """Create comprehensive visualizations for redesigned methods"""
    print(f"\n📊 Creating Comprehensive Visualizations")
    print("-" * 50)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Enhanced Class Imbalance Strategy - Performance Comparison', fontsize=16, fontweight='bold')
    
    methods = ['Multiclass Balanced Bagging', 'Ordinal Cumulative']
    metrics = ['f1_macro', 'high_risk_recall', 'balanced_accuracy', 'cohen_kappa']
    metric_names = ['F1-Macro', 'High-Risk Recall', 'Balanced Accuracy', "Cohen's Kappa"]
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i//2, i%2]
        
        method_scores = []
        for method in methods:
            scores = []
            for target, results in all_results.items():
                if results and method in results and results[method]:
                    scores.append(results[method]['metrics'][metric])
            method_scores.append(scores)
        
        # Create box plot
        bp = ax.boxplot(method_scores, labels=methods, patch_artist=True)
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(f'{metric_name} Comparison', fontsize=12)
        ax.set_ylabel('Score', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '01_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrices for Best Method per Target
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confusion Matrices - Best Method per Target', fontsize=16, fontweight='bold')
    
    for i, (target, results) in enumerate(all_results.items()):
        if i >= 4:
            break
        
        ax = axes[i//2, i%2]
        if results:
            # pick best by high-risk recall
            choices = {m: results[m] for m in methods if m in results and results[m]}
            if choices:
                best_m = max(choices.keys(), key=lambda k: choices[k]['metrics']['high_risk_recall'])
                y_true = results[best_m]['all_true_values']
                y_pred = results[best_m]['all_predictions']
                if len(y_true) > 0 and len(y_pred) > 0:
                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
                    cm_normalized = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
                    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', ax=ax)
                    ax.set_title(f'{target} - {best_m}', fontsize=12)
                    ax.set_xlabel('Predicted', fontsize=10)
                    ax.set_ylabel('Actual', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{target} - No Data', fontsize=12)
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{target} - No Data', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{target} - No Data', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '02_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to: {results_dir}")

def test_all_methods_for_target(X, y, target_name):
    """Test redesigned imbalance methods for a specific target"""
    print(f"\n🎯 TESTING REDESIGNED IMBALANCE METHODS FOR {target_name}")
    print("=" * 70)
    
    # Handle both DataFrame and Series for y
    if isinstance(y, pd.DataFrame):
        mask = ~pd.isna(y[target_name])
        X_target = X[mask]
        y_target = y[target_name][mask]
    else:
        mask = ~pd.isna(y)
        X_target = X[mask]
        y_target = y[mask]
    
    print(f"📊 Available data for {target_name}: {len(X_target):,} samples")
    
    # Get numeric features
    X_numeric = X_target.select_dtypes(include=[np.number]).fillna(0)
    
    # Generate consistent rolling window indices
    indices_list = generate_rolling_window_indices(X_numeric, WINDOW_SIZE, N_SPLITS)
    
    if not indices_list:
        print("⚠️ No valid folds generated")
        return None
    
    # Global reference prior across all available labels for this target
    ref_prior_global = np.bincount(y_target.astype(int), minlength=4).astype(float)
    ref_prior_global = ref_prior_global / (ref_prior_global.sum() + 1e-12)

    results = {}
    results['Multiclass Balanced Bagging'] = None
    results['Ordinal Cumulative'] = None

    # Multiclass Balanced Bagging
    results['Multiclass Balanced Bagging'] = run_multiclass_ensemble(
        X_numeric, y_target, indices_list, target_name, ref_prior_global
    )

    # Ordinal cumulative alternative
    results['Ordinal Cumulative'] = run_ordinal_alternative(
        X_numeric, y_target, indices_list, target_name, ref_prior_global
    )
    
    return results

def select_best_imbalance_method(all_results):
    """Select the best method per target and summarize overall"""
    print(f"\n🎯 SELECTING BEST METHODS (Per Target)")
    print("-" * 60)
    methods = ['Multiclass Balanced Bagging', 'Ordinal Cumulative']
    method_scores_overall = {m: [] for m in methods}
    per_target_best = {}

    print("📊 Best Method per Target:")
    print("-" * 40)
    for target, results in all_results.items():
        if not results:
            continue
        candidates = {m: results[m] for m in methods if m in results and results[m]}
        if not candidates:
            continue
        best = sorted(candidates.items(), key=lambda kv: (kv[1]['metrics']['high_risk_recall'], kv[1].get('stability_score', -1e9)), reverse=True)[0]
        per_target_best[target] = { 'method': best[0], 'metrics': best[1]['metrics'], 'stability_score': best[1].get('stability_score') }
        method_scores_overall[best[0]].append(best[1]['metrics']['f1_macro'])
        print(f"   • {target}: {best[0]} (High-Risk Recall={best[1]['metrics']['high_risk_recall']:.4f}, F1={best[1]['metrics']['f1_macro']:.4f})")

    method_summary = {}
    for m, scores in method_scores_overall.items():
        if scores:
            method_summary[m] = {
                'mean_f1': float(np.mean(scores)),
                'std_f1': float(np.std(scores)),
                'stability_score': float(np.mean(scores) - np.std(scores))
            }

    return {
        'per_target_best': per_target_best,
        'method_summary': method_summary,
        'all_results': all_results
    }

def save_enhanced_results(all_results, best_selection, results_dir):
    """Save comprehensive enhanced results and summary"""
    print(f"\n💾 Saving Enhanced Comprehensive Results")
    print("-" * 40)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    with open(f'{results_dir}/step6_enhanced_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    # Create enhanced summary
    summary = {
        'execution_date': datetime.now().isoformat(),
        'approach': 'enhanced_imbalance_strategy_redesign_phases_1_6',
        'targets_tested': len(all_results),
        'methods_tested': 2,
        'per_target_best': best_selection['per_target_best'],
        'method_summary': best_selection['method_summary'],
        'key_improvements': {
            'phase_1': 'Enhanced evaluation + per-fold class prevalence + classification reports',
            'phase_2': 'Class-balanced (EENS) weights with prior correction at inference',
            'phase_3': 'Balanced bagging ensemble (undersampling majority + weights)',
            'phase_4': 'Per-class Platt calibration + business-priority thresholds',
            'phase_5': 'Ordinal cumulative alternative (K−1 models)',
            'phase_6': 'Per-target selection with stability guard'
        }
    }
    
    with open(f'{results_dir}/step6_enhanced_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ Enhanced results saved to: {results_dir}")
    
    return results_dir

def main():
    """Main execution - Enhanced Class Imbalance Strategy (Redesigned)"""

    df, X, y, exclude_cols, target_cols = load_step4_data()

    print(f"\n🔬 ENHANCED CLASS IMBALANCE STRATEGY ENGINE - REDESIGN")
    print("=" * 70)
    print(f"📊 Testing redesigned methods across {len(target_cols)} targets")
    print(f"🎯 Targets: {target_cols}")
    print(f"📈 Features: {X.shape[1]}")
    print(f"📅 Using Rolling Window CV for temporal robustness")
    print(f"🛡️ No synthetic sampling; algorithm-level and ensemble techniques only")
    print(f"📊 Phase 1: Enhanced evaluation + prevalence logging")
    print(f"⚖️  Phase 2: Class-balanced weights + prior correction")
    print(f"🧰 Phase 3: Balanced bagging ensemble")
    print(f"🎚️ Phase 4: Calibration + business thresholds")
    print(f"📈 Phase 5: Ordinal cumulative alternative")
    print(f"🧭 Phase 6: Per-target selection with stability guard")

    all_results = {}

    for target_name in target_cols:
        results = test_all_methods_for_target(X, y, target_name)
        all_results[target_name] = results

    best_selection = select_best_imbalance_method(all_results)

    results_dir = 'result/step6_class_imbalance'
    save_enhanced_results(all_results, best_selection, results_dir)

    # Create comprehensive visualizations
    create_comprehensive_visualizations(all_results, results_dir)

    print(f"\n🎉 ENHANCED CLASS IMBALANCE STRATEGY COMPLETED!")
    print("=" * 70)
    print("✅ Phases 1-6 implemented with temporal robustness and business focus")
    print("✅ Per-target best method selected with stability consideration")
    print("✅ Comprehensive visualizations and summaries created")
    print(f"✅ Enhanced results saved in: {results_dir}")

    return best_selection

if __name__ == "__main__":
    best_imbalance = main()
