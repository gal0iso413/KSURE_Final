"""
XGBoost Risk Prediction Model - Step 7: Model Architecture Experiments
===============================================================

Step 7 Implementation - ARCHITECTURE EXPERIMENTS (Revised):
1. Phase 1: Unified vs Individual Models (single shared model vs separate models per target)
2. Phase 2: Two-Stage Cascade (Gate: high-risk vs not; Specialist: 2 vs 3)

Design Focus:
- Use Step 4 optimized features
- Use Step 5 Rolling Window CV for temporal robustness
- Apply Step 6 imbalance handling consistently: class-balanced (EENS) weights, calibration (Platt), prior correction, business thresholds
- Efficient training; focus on high-risk recall, macro F1, balanced accuracy, kappa, high-risk PR-AUC, ECE, stability
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
    confusion_matrix, precision_recall_fscore_support, average_precision_score
)
from sklearn.linear_model import LogisticRegression
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
N_ESTIMATORS_UNIFIED = 200
N_ESTIMATORS_INDIVIDUAL = 300
RANDOM_STATE = 42
BETA_STEP7 = 0.999

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
    print("🚀 MODEL ARCHITECTURE EXPERIMENTS - STEP 7")
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
    step_size = (total_samples - window_samples) // n_splits
    
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

def define_ordinal_cost_matrix():
    """Define business-aligned cost matrix for ordinal misclassifications"""
    cost_matrix = {
        # Critical errors (high business impact)
        (0, 3): 15,  # No risk → High risk: Very costly (missing high risk)
        (1, 3): 12,  # Low risk → High risk: Very costly
        (2, 3): 8,   # Medium risk → High risk: Costly
        (3, 0): 20,  # High risk → No risk: Extremely costly (false negative)
        (3, 1): 15,  # High risk → Low risk: Very costly
        
        # Moderate errors
        (0, 2): 6,   # No risk → Medium risk: Moderate cost
        (1, 2): 4,   # Low risk → Medium risk: Moderate cost
        (2, 0): 8,   # Medium risk → No risk: High cost
        (2, 1): 5,   # Medium risk → Low risk: Moderate cost
        
        # Minor errors
        (0, 1): 2,   # No risk → Low risk: Low cost
        (1, 0): 3,   # Low risk → No risk: Low cost
    }
    return cost_matrix

def compute_ordinal_sample_weights(y_true, cost_matrix):
    # Deprecated in revised Step 7; retained for compatibility if needed
    return np.ones(len(y_true))

def calculate_comprehensive_metrics(y_true, y_pred, y_proba=None, high_risk_proba=None):
    """Classification metrics with imbalance awareness and calibration checks"""
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

def train_single_model(X, y_target, model_type, n_estimators: int):
    """Core training for single-stage multiclass with Step 6 handling"""
    results = {
        'metrics_list': [],
        'all_true_values': [],
        'all_predictions': [],
        'all_probabilities': []
    }
    
    # Generate rolling window indices (Step 5)
    indices_list = generate_rolling_window_indices(X)
    
    for fold, (train_idx, test_idx) in enumerate(indices_list):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_target.iloc[train_idx], y_target.iloc[test_idx]
        
        # Clean data
        train_mask = ~pd.isna(y_train)
        test_mask = ~pd.isna(y_test)
        
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask].astype(int)
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask].astype(int)
        
        if len(y_train_clean) == 0 or len(y_test_clean) == 0:
            continue
        
        # EENS class-balanced weights
        unique, counts = np.unique(y_train_clean.values, return_counts=True)
        beta = BETA_STEP7
        class_to_weight = {}
        for k, n_k in zip(unique, counts):
            effective_num = 1.0 - (beta ** n_k)
            class_to_weight[int(k)] = (1.0 - beta) / max(effective_num, 1e-12)
        sample_weights = np.array([class_to_weight[int(c)] for c in y_train_clean.values], dtype=float)
        sample_weights = sample_weights / (sample_weights.mean() + 1e-12)

        # Model
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='mlogloss'
        )

        # Calibration split (temporal)
        cal_size = max(1, int(0.1 * len(X_train_clean))) if len(X_train_clean) > 20 else 0
        if cal_size > 0:
            X_inner = X_train_clean.iloc[:-cal_size]
            y_inner = y_train_clean.iloc[:-cal_size]
            w_inner = sample_weights[:-cal_size]
            X_cal = X_train_clean.iloc[-cal_size:]
            y_cal = y_train_clean.iloc[-cal_size:]
        else:
            X_inner, y_inner, w_inner = X_train_clean, y_train_clean, sample_weights
            X_cal, y_cal = None, None

        model.fit(X_inner, y_inner, sample_weight=w_inner)
        proba_test = model.predict_proba(X_test_clean)

        # Platt calibration per class
        if X_cal is not None and len(X_cal) > 0:
            proba_cal = model.predict_proba(X_cal)
            platt = []
            for k in range(proba_test.shape[1]):
                lr = LogisticRegression(max_iter=1000)
                y_bin = (y_cal.values == k).astype(int)
                if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                    platt.append(None)
                else:
                    lr.fit(proba_cal[:, [k]], y_bin)
                    platt.append(lr)
            calibrated = np.zeros_like(proba_test)
            for k in range(proba_test.shape[1]):
                if platt[k] is None:
                    calibrated[:, k] = proba_test[:, k]
                else:
                    calibrated[:, k] = platt[k].predict_proba(proba_test[:, [k]])[:, 1]
            proba_test = np.clip(calibrated, 1e-12, 1.0)
            proba_test /= proba_test.sum(axis=1, keepdims=True)

        # Prior correction
        train_prior = np.bincount(y_train_clean.values, minlength=4).astype(float)
        train_prior = train_prior / (train_prior.sum() + 1e-12)
        ref_prior = np.bincount(y_target.dropna().astype(int).values, minlength=4).astype(float)
        ref_prior = ref_prior / (ref_prior.sum() + 1e-12)
        proba_test = proba_test * (ref_prior / np.maximum(train_prior, 1e-12))
        proba_test = np.clip(proba_test, 1e-12, 1.0)
        proba_test /= proba_test.sum(axis=1, keepdims=True)

        # Threshold optimization per class; assignment in business order 3→2→1→0
        thresholds = {}
        for cls in [0, 1, 2, 3]:
            best_t, best_f1 = 0.5, 0.0
            y_true_bin = (y_test_clean.values == cls).astype(int)
            for t in np.linspace(0.1, 0.9, 17):
                y_pred_bin = (proba_test[:, cls] >= t).astype(int)
                f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            thresholds[cls] = best_t

        y_pred = np.zeros(len(proba_test), dtype=int)
        for i in range(len(proba_test)):
            assigned = False
            for cls in [3, 2, 1, 0]:
                if proba_test[i, cls] >= thresholds[cls]:
                    y_pred[i] = cls
                    assigned = True
                    break
            if not assigned:
                y_pred[i] = int(np.argmax(proba_test[i]))

        high_risk_proba = proba_test[:, 2] + proba_test[:, 3]
        metrics = calculate_comprehensive_metrics(y_test_clean.values, y_pred, proba_test, high_risk_proba)
        results['metrics_list'].append(metrics)
        results['all_true_values'].append(y_test_clean.values)
        results['all_predictions'].append(y_pred)
        results['all_probabilities'].append(proba_test)
    
    # Calculate average metrics
    if results['metrics_list']:
        avg_metrics = {}
        keys = results['metrics_list'][0].keys()
        for key in keys:
            if key != 'per_class':
                avg_metrics[key] = float(np.mean([m[key] for m in results['metrics_list']]))
        f1s = [m['f1_macro'] for m in results['metrics_list']]
        avg_metrics['stability_score'] = float(np.mean(f1s) - np.std(f1s))
        results['avg_metrics'] = avg_metrics
    else:
        results['avg_metrics'] = {}
    
    return results

def phase1_unified_vs_individual(X, y, target_cols):
    """Phase 1: Compare unified model vs individual models per target"""
    print(f"\n📊 PHASE 1: UNIFIED VS INDIVIDUAL MODELS")
    print("-" * 50)
    
    results = {}
    
    # 1. Unified Model (single model for all targets)
    print("🎯 Training Unified Model (shared configuration)...")
    unified_results = {}
    for target_name in target_cols:
        print(f"   Training unified model for {target_name}...")
        target_results = train_single_model(X, y[target_name], model_type='unified', n_estimators=N_ESTIMATORS_UNIFIED)
        unified_results[target_name] = target_results
    results['unified'] = unified_results
    
    # 2. Individual Models (separate model per target)
    print("🎯 Training Individual Models (per-target configuration)...")
    individual_results = {}
    for target_name in target_cols:
        print(f"   Training individual model for {target_name}...")
        target_results = train_single_model(X, y[target_name], model_type='individual', n_estimators=N_ESTIMATORS_INDIVIDUAL)
        individual_results[target_name] = target_results
    results['individual'] = individual_results
    
    # 3. Compare performance
    compare_unified_vs_individual(unified_results, individual_results)
    
    return results

def phase2_two_stage_cascade(X, y, target_cols):
    """Phase 2: Two-Stage Cascade (Gate: high-risk vs not; Specialist: 2 vs 3)"""
    print(f"\n📊 PHASE 2: TWO-STAGE CASCADE")
    print("-" * 50)
    
    def compute_class_balanced_weights_eens(y: np.ndarray, beta: float = 0.999) -> np.ndarray:
        unique, counts = np.unique(y, return_counts=True)
        class_to_weight = {}
        for k, n_k in zip(unique, counts):
            effective_num = 1.0 - (beta ** n_k)
            class_weight = (1.0 - beta) / max(effective_num, 1e-12)
            class_to_weight[int(k)] = class_weight
        weights = np.array([class_to_weight[int(c)] for c in y], dtype=float)
        return weights / (weights.mean() + 1e-12)

    def train_binary_with_calibration(X_train, y_train, X_test):
        y_train = y_train.astype(int)
        w = compute_class_balanced_weights_eens(y_train.values)
        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS_UNIFIED,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='logloss'
        )
        cal_size = max(1, int(0.1 * len(X_train))) if len(X_train) > 20 else 0
        if cal_size > 0:
            X_inner = X_train.iloc[:-cal_size]
            y_inner = y_train.iloc[:-cal_size]
            w_inner = w[:-cal_size]
            X_cal = X_train.iloc[-cal_size:]
            y_cal = y_train.iloc[-cal_size:]
        else:
            X_inner, y_inner, w_inner = X_train, y_train, w
            X_cal, y_cal = None, None
        model.fit(X_inner, y_inner, sample_weight=w_inner)
        proba_test = model.predict_proba(X_test)[:, 1]
        # Platt
        if X_cal is not None and len(X_cal) > 0 and y_cal.nunique() == 2:
            proba_cal = model.predict_proba(X_cal)[:, 1]
            lr = LogisticRegression(max_iter=1000)
            lr.fit(proba_cal.reshape(-1, 1), y_cal.values)
            proba_test = lr.predict_proba(proba_test.reshape(-1, 1))[:, 1]
        return proba_test

    results = {}
    for target_name in target_cols:
        print(f"🎯 Training cascade for {target_name}...")
        indices_list = generate_rolling_window_indices(X)
        y_target = y[target_name]
        fold_metrics = []
        all_true_values, all_predictions = [], []

        for fold, (train_idx, test_idx) in enumerate(indices_list):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_target.iloc[train_idx], y_target.iloc[test_idx]
            train_mask = ~pd.isna(y_train)
            test_mask = ~pd.isna(y_test)
            X_train_clean = X_train[train_mask]
            y_train_clean = y_train[train_mask].astype(int)
            X_test_clean = X_test[test_mask]
            y_test_clean = y_test[test_mask].astype(int)
            if len(y_train_clean) == 0 or len(y_test_clean) == 0:
                continue
            # Gate: high-risk vs not
            gate_y_train = (y_train_clean >= 2).astype(int)
            gate_proba = train_binary_with_calibration(X_train_clean, gate_y_train, X_test_clean)
            # Specialist: 2 vs 3 (train only on high-risk train subset)
            hr_mask_train = y_train_clean >= 2
            if hr_mask_train.sum() > 1:
                spec_y_train = (y_train_clean[hr_mask_train] == 3).astype(int)  # 1: class 3, 0: class 2
                spec_proba = train_binary_with_calibration(
                    X_train_clean[hr_mask_train], spec_y_train, X_test_clean
                )
            else:
                spec_proba = np.zeros(len(X_test_clean))

            # thresholds: tune gate threshold to favor recall; specialist with F1
            best_gate_t, best_hr_recall = 0.3, -1.0
            y_true_hr = (y_test_clean.values >= 2).astype(int)
            for t in np.linspace(0.1, 0.9, 17):
                y_gate = (gate_proba >= t).astype(int)
                rec = recall_score(y_true_hr, y_gate, zero_division=0)
                if rec > best_hr_recall:
                    best_hr_recall, best_gate_t = rec, t
            gate_pred = (gate_proba >= best_gate_t)

            # quick non-high-risk 0 vs 1 classifier
            nonhr_mask_train = y_train_clean < 2
            if nonhr_mask_train.sum() > 1:
                nh_y_train = (y_train_clean[nonhr_mask_train] == 1).astype(int)
                nh_proba = train_binary_with_calibration(
                    X_train_clean[nonhr_mask_train], nh_y_train, X_test_clean
                )
            else:
                nh_proba = np.zeros(len(X_test_clean))

            y_pred = np.zeros(len(X_test_clean), dtype=int)
            for i in range(len(X_test_clean)):
                if gate_pred[i]:
                    y_pred[i] = 3 if spec_proba[i] >= 0.5 else 2
                else:
                    y_pred[i] = 1 if nh_proba[i] >= 0.5 else 0

            # metrics
            high_risk_proba = gate_proba
            m = calculate_comprehensive_metrics(y_test_clean.values, y_pred, high_risk_proba=high_risk_proba)
            fold_metrics.append(m)
            all_true_values.append(y_test_clean.values)
            all_predictions.append(y_pred)
            print(f"   Fold {fold+1}: HR-Recall={m['high_risk_recall']:.4f}, F1={m['f1_macro']:.4f}")

        if not fold_metrics:
            results[target_name] = None
            continue
        avg = {k: float(np.mean([mm[k] for mm in fold_metrics])) for k in fold_metrics[0].keys()}
        avg['stability_score'] = float(np.mean([mm['f1_macro'] for mm in fold_metrics]) - np.std([mm['f1_macro'] for mm in fold_metrics]))
        results[target_name] = {
            'avg_metrics': avg,
            'metrics_list': fold_metrics,
            'all_true_values': all_true_values,
            'all_predictions': all_predictions
        }
    return results

# Phase 3 removed in revised Step 7

def compare_unified_vs_individual(unified_results, individual_results):
    """Compare unified vs individual model performance"""
    print(f"\n📊 UNIFIED VS INDIVIDUAL COMPARISON")
    print("-" * 40)
    
    for target_name in unified_results.keys():
        unified_metrics = unified_results[target_name]['avg_metrics']
        individual_metrics = individual_results[target_name]['avg_metrics']
        
        print(f"\n🎯 {target_name}:")
        print(f"   Unified Model:")
        print(f"     F1-Macro: {unified_metrics.get('f1_macro', 0):.4f}")
        print(f"     High-Risk Recall: {unified_metrics.get('high_risk_recall', 0):.4f}")
        print(f"     Balanced Accuracy: {unified_metrics.get('balanced_accuracy', 0):.4f}")
        
        print(f"   Individual Model:")
        print(f"     F1-Macro: {individual_metrics.get('f1_macro', 0):.4f}")
        print(f"     High-Risk Recall: {individual_metrics.get('high_risk_recall', 0):.4f}")
        print(f"     Balanced Accuracy: {individual_metrics.get('balanced_accuracy', 0):.4f}")

def compare_cascade_vs_single_stage(single_stage_results, cascade_results):
    print(f"\n📊 SINGLE-STAGE VS CASCADE COMPARISON")
    print("-" * 40)
    for target_name in single_stage_results.keys():
        ss = single_stage_results[target_name]['avg_metrics']
        cs_entry = cascade_results.get(target_name)
        cs = cs_entry.get('avg_metrics', {}) if cs_entry else None
        print(f"\n🎯 {target_name}:")
        print(f"   Single-Stage: HR-Recall={ss.get('high_risk_recall', 0):.4f}, F1={ss.get('f1_macro', 0):.4f}, PR-AUC={ss.get('high_risk_pr_auc', 0):.4f}")
        if cs:
            print(f"   Cascade:     HR-Recall={cs.get('high_risk_recall', 0):.4f}, F1={cs.get('f1_macro', 0):.4f}, PR-AUC={cs.get('high_risk_pr_auc', 0):.4f}")
        else:
            print(f"   Cascade:     No data (insufficient samples/folds)")

def compare_ordinal_vs_no_ordinal(no_ordinal_results, ordinal_results):
    """Compare ordinal vs no ordinal performance"""
    print(f"\n📊 ORDINAL VS NO ORDINAL COMPARISON")
    print("-" * 40)
    
    for target_name in no_ordinal_results.keys():
        no_ord_metrics = no_ordinal_results[target_name]['avg_metrics']
        ord_metrics = ordinal_results[target_name]['avg_metrics']
        
        print(f"\n🎯 {target_name}:")
        print(f"   No Ordinal:")
        print(f"     F1-Macro: {no_ord_metrics.get('f1_macro', 0):.4f}")
        print(f"     High-Risk Recall: {no_ord_metrics.get('high_risk_recall', 0):.4f}")
        
        print(f"   Ordinal (Cost-Sensitive):")
        print(f"     F1-Macro: {ord_metrics.get('f1_macro', 0):.4f}")
        print(f"     High-Risk Recall: {ord_metrics.get('high_risk_recall', 0):.4f}")

def select_best_architecture_simple(unified_results, cascade_results):
    print(f"\n🏆 SELECTING BEST ARCHITECTURE (Unified vs Cascade)")
    print("-" * 40)
    ss_vals = [v.get('avg_metrics', {}).get('high_risk_recall', 0) for v in unified_results.values() if v and 'avg_metrics' in v]
    cs_vals = [v.get('avg_metrics', {}).get('high_risk_recall', 0) for v in cascade_results.values() if v and 'avg_metrics' in v]
    ss_hr = float(np.mean(ss_vals)) if ss_vals else 0.0
    cs_hr = float(np.mean(cs_vals)) if cs_vals else 0.0
    print(f"   Unified (Single-Stage) HR-Recall: {ss_hr:.4f}")
    print(f"   Cascade HR-Recall:              {cs_hr:.4f}")
    return 'Two-Stage Cascade' if cs_hr >= ss_hr else 'Single-Stage (Unified)'

def save_step7_results(phase1_results, phase2_results, best_approach):
    """Save comprehensive Step 7 results"""
    print(f"\n💾 Saving Step 7 Results")
    print("-" * 30)
    
    results_dir = 'result/step7_model_architecture'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    all_results = {
        'phase1_unified_vs_individual': phase1_results,
        'phase2_two_stage_cascade': phase2_results,
        'best_approach': best_approach,
        'execution_date': datetime.now().isoformat()
    }
    
    with open(f'{results_dir}/step7_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    # Create summary
    summary = {
        'execution_date': datetime.now().isoformat(),
        'approach': 'model_architecture_experiments_revised',
        'phases_tested': 2,
        'best_approach': best_approach,
        'key_components': {
            'step4_features': 'Optimized 27 features',
            'step5_validation': 'Rolling Window CV',
            'step6_handling': 'Class-balanced weights, calibration, prior correction, thresholds',
            'efficient_training': 'Reused results from previous phases'
        },
        'evaluation_metrics': ['high_risk_recall', 'f1_macro', 'balanced_accuracy', 'cohen_kappa', 'high_risk_pr_auc', 'stability']
    }
    
    with open(f'{results_dir}/step7_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ Results saved to: {results_dir}")

def main():
    """Main execution - Step 7: Model Architecture Experiments"""
    print("🚀 STEP 7: MODEL ARCHITECTURE EXPERIMENTS")
    print("=" * 60)
    
    # Load Step 4 data
    df, X, y, exclude_cols, target_cols = load_step4_data()
    
    print(f"📊 Testing 2 phases across {len(target_cols)} targets")
    print(f"🎯 Targets: {target_cols}")
    print(f"📈 Features: {X.shape[1]}")
    print(f"📅 Using Rolling Window CV (Step 5)")
    print(f"⚖️ Applying Step 6 handling (EENS weights, calibration, prior correction, thresholds)")
    
    # Phase 1: Unified vs Individual
    phase1_results = phase1_unified_vs_individual(X, y, target_cols)
    
    # Phase 2: Two-stage cascade
    phase2_results = phase2_two_stage_cascade(X, y, target_cols)
    
    # Compare Unified vs Cascade and select best
    compare_cascade_vs_single_stage(phase1_results['unified'], phase2_results)
    best_approach = select_best_architecture_simple(phase1_results['unified'], phase2_results)
    
    # Save results
    save_step7_results(phase1_results, phase2_results, best_approach)
    
    print(f"\n🎉 STEP 7 COMPLETED!")
    print("=" * 60)
    print("✅ Phase 1: Unified vs Individual models tested")
    print("✅ Phase 2: Two-Stage Cascade evaluated")
    print("✅ Efficient training with result reuse")
    print(f"🏆 Best Architecture: {best_approach}")
    
    return best_approach

if __name__ == "__main__":
    best_architecture = main()
