"""
XGBoost Risk Prediction Model - Step 6: Class Imbalance Strategy (Simplified)
============================================================================

Scope in this simplified Step 6:
1. Algorithm-level Reweighting: Class-balanced sample weights on a single XGBoost model; predict via argmax

Shared Evaluation Layer:
- Balanced Accuracy, Cohen's Kappa, Macro F1, High-Risk Recall
- Per-class metrics and classification reports
- Per-fold class prevalence logging

Design Focus:
- Rolling Window CV for temporal robustness; no synthetic sampling
- Single-model pipeline for explainability; no ensembles, no ordinal, no thresholds, no calibration
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
    print("🚀 CLASS IMBALANCE STRATEGY (SIMPLIFIED) - PHASE 1-2 IMPLEMENTATION")
    print("=" * 70)
    
    df = pd.read_csv('dataset/credit_risk_dataset_step4.csv')
    print(f"✅ Step 4 dataset loaded: {df.shape}")
    
    # Sort by 보험청약일자 for temporal validation
    if '보험청약일자' in df.columns:
        df = df.sort_values('보험청약일자').reset_index(drop=True)
        print(f"✅ Data sorted by 보험청약일자 for temporal validation")
    
    exclude_cols = [
        '사업자등록번호', '대상자명', '청약번호', '보험청약일자', '수출자대상자번호', '업종코드1'
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
    """Evaluation layer: comprehensive metrics with validation"""
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

# Deprecated leakage-prone threshold optimizer removed in simplified Step 6

# Deprecated sequential threshold application removed in simplified Step 6

def compute_prevalence(y: np.ndarray) -> dict:
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {int(k): {'count': int(v), 'pct': float(v/total)} for k, v in zip(unique, counts)}

def compute_class_balanced_weights_eens(*args, **kwargs):
    """Deprecated in simplified Step 6."""
    raise NotImplementedError("EENS weighting is not used in simplified Step 6")

def undersample_indices(*args, **kwargs):
    """Deprecated in simplified Step 6."""
    raise NotImplementedError("Undersampling is not used in simplified Step 6")

def prior_correction(*args, **kwargs):
    """Deprecated in simplified Step 6."""
    raise NotImplementedError("Prior correction is not used in simplified Step 6")



def train_multiclass_ensemble_with_calibration(*args, **kwargs):
    """Deprecated in simplified Step 6."""
    raise NotImplementedError("Ensemble is not used in simplified Step 6")

def predict_with_ensemble(*args, **kwargs):
    """Deprecated in simplified Step 6."""
    raise NotImplementedError("Ensemble is not used in simplified Step 6")

def train_ordinal_cumulative(*args, **kwargs):
    """Deprecated in simplified Step 6."""
    raise NotImplementedError("Ordinal alternative is deferred from Step 6")

def predict_ordinal_proba(*args, **kwargs):
    """Deprecated in simplified Step 6."""
    raise NotImplementedError("Ordinal alternative is deferred from Step 6")

def run_multiclass_ensemble(*args, **kwargs):
    """Deprecated in simplified Step 6."""
    return None

def run_ordinal_alternative(*args, **kwargs):
    """Deprecated in simplified Step 6."""
    return None

def approach_baseline(X_numeric, y_target, indices_list, target_name):
    """Original baseline: No imbalance handling, no calibration"""
    print(f"\n📊 Original Baseline (No Imbalance Handling) for {target_name}")
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
        
        # Evaluation
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
        'approach': 'Original',
        'description': 'No imbalance handling, no calibration',
        'metrics': avg_metrics,
        'fold_metrics': metrics_list,
        'n_splits': len(metrics_list),
        'features': X_numeric.shape[1],
        'all_true_values': all_true_values,  # FIXED: Include true values
        'all_predictions': all_predictions,
        'all_probabilities': all_probabilities
    }

def approach_sample_weights(X_numeric, y_target, indices_list, target_name):
    """Algorithm-level reweighting via sample weights"""
    print(f"\n📊 Algorithm-Level Reweighting (Class-balanced sample weights) for {target_name}")
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
        
        # Class-balanced sample weights
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
        
        # Evaluation
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
        'approach': 'Algorithm-Level Reweighting',
        'description': 'Algorithm-level imbalance handling using class-balanced sample weights',
        'metrics': avg_metrics,
        'fold_metrics': metrics_list,
        'n_splits': len(metrics_list),
        'features': X_numeric.shape[1],
        'all_true_values': all_true_values,  # FIXED: Include true values
        'all_predictions': all_predictions,
        'all_probabilities': all_probabilities
    }



def approach_threshold_optimization(*args, **kwargs):
    """Deprecated in simplified Step 6."""
    return None

def create_comprehensive_visualizations(all_results, results_dir):
    """Create comprehensive visualizations for redesigned methods"""
    print(f"\n📊 Creating Comprehensive Visualizations")
    print("-" * 50)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Class Imbalance Strategy - Performance Comparison', fontsize=16, fontweight='bold')
    
    methods = ['Original', 'Algorithm-Level Reweighting']
    metrics = ['f1_macro', 'high_risk_recall', 'balanced_accuracy', 'cohen_kappa']
    metric_names = ['F1-Macro', 'High-Risk Recall', 'Balanced Accuracy', "Cohen's Kappa"]
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i//2, i%2]
        
        method_scores = []
        labels_to_plot = []
        for method in methods:
            scores = []
            for target, results in all_results.items():
                if results and method in results and results[method]:
                    scores.append(results[method]['metrics'][metric])
            if scores:
                method_scores.append(scores)
            labels_to_plot.append(method)

        if method_scores:
            bp = ax.boxplot(method_scores, labels=labels_to_plot, patch_artist=True)
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(labels_to_plot)]):
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
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
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
    """Test simplified imbalance methods for a specific target"""
    print(f"\n🎯 TESTING SIMPLIFIED IMBALANCE METHODS FOR {target_name}")
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
    
    # Get numeric features (preserve NaNs for XGBoost's native missing handling)
    X_numeric = X_target.select_dtypes(include=[np.number])
    
    # Generate consistent rolling window indices
    indices_list = generate_rolling_window_indices(X_numeric, WINDOW_SIZE, N_SPLITS)
    
    if not indices_list:
        print("⚠️ No valid folds generated")
        return None
    
    # Global reference prior across all available labels for this target
    ref_prior_global = np.bincount(y_target.astype(int), minlength=4).astype(float)
    ref_prior_global = ref_prior_global / (ref_prior_global.sum() + 1e-12)

    results = {}
    results['Original'] = approach_baseline(X_numeric, y_target, indices_list, target_name)
    results['Algorithm-Level Reweighting'] = approach_sample_weights(X_numeric, y_target, indices_list, target_name)
    
    return results

def select_best_imbalance_method(all_results):
    """Select the best simplified method per target and summarize overall"""
    print(f"\n🎯 SELECTING BEST METHODS (Per Target)")
    print("-" * 60)
    methods = ['Original', 'Algorithm-Level Reweighting']
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
        best = sorted(candidates.items(), key=lambda kv: (kv[1]['metrics']['high_risk_recall'], kv[1]['metrics']['f1_macro']), reverse=True)[0]
        per_target_best[target] = { 'method': best[0], 'metrics': best[1]['metrics'] }
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
    
    # Create simplified summary
    summary = {
        'execution_date': datetime.now().isoformat(),
        'approach': 'class_imbalance_strategy_simplified_phases_1_2',
        'targets_tested': len(all_results),
        'methods_tested': 3,
        'per_target_best': best_selection['per_target_best'],
        'method_summary': best_selection['method_summary'],
        'key_improvements': {
            'algorithm_reweighting': 'Algorithm-level reweighting (class-balanced sample weights)',
            'deferred': 'Ensembles, business thresholds, ordinal cumulative, probability calibration, and data-level oversampling (SMOTE) moved to later steps'
        }
    }
    
    with open(f'{results_dir}/step6_enhanced_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ Enhanced results saved to: {results_dir}")
    
    return results_dir

def main():
    """Main execution - Simplified Class Imbalance Strategy (Phases 1-2)"""

    df, X, y, exclude_cols, target_cols = load_step4_data()

    print(f"\n🔬 CLASS IMBALANCE STRATEGY ENGINE - SIMPLIFIED")
    print("=" * 70)
    print(f"📊 Testing simplified methods across {len(target_cols)} targets")
    print(f"🎯 Targets: {target_cols}")
    print(f"📈 Features: {X.shape[1]}")
    print(f"📅 Using Rolling Window CV for temporal robustness")
    print(f"🛡️ No synthetic sampling; single-model pipeline only")
    print(f"⚖️  Algorithm-level reweighting: Class-balanced sample weights (argmax)")

    all_results = {}

    for target_name in target_cols:
        results = test_all_methods_for_target(X, y, target_name)
        all_results[target_name] = results

    best_selection = select_best_imbalance_method(all_results)

    results_dir = 'result/step6_class_imbalance'
    save_enhanced_results(all_results, best_selection, results_dir)

    # Create comprehensive visualizations
    create_comprehensive_visualizations(all_results, results_dir)

    print(f"\n🎉 SIMPLIFIED CLASS IMBALANCE STRATEGY COMPLETED!")
    print("=" * 70)
    print("✅ Phases 1-2 implemented with temporal robustness and explainability")
    print("✅ Per-target best method selected")
    print("✅ Comprehensive visualizations and summaries created")
    print(f"✅ Results saved in: {results_dir}")

    return best_selection

if __name__ == "__main__":
    best_imbalance = main()
