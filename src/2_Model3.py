"""
XGBoost Risk Prediction Model - Step 3: Unified Temporal Validation
==================================================================

Step 3 Implementation - Validation Strategies (Max 5):
1. Random Split (shuffle=True) - Performance baseline ignoring temporal characteristics
2. Stratified Random Split (shuffle=True, stratify=y) - Baseline with class balance
3. Simple Temporal Holdout (shuffle=False) - Basic temporal validation (past vs future)
4. Expanding Window CV - Evaluates stability as more data accumulates over time
5. Rolling Window CV - Evaluates adaptation to recent trends by discarding old data

Design Focus:
- Proper data sorting by 보험청약일자 before temporal validation
- Consistent parameters across all targets (same split ratios, fold counts)
- Multi-target analysis (risk_year1, risk_year2, risk_year3, risk_year4)
- Unified validation strategy selection for all targets
- Clear distinction between temporal and non-temporal approaches
\n+Metric Focus:
- Report F1-Score (Macro) and High-Risk Recall (classes >= 2) instead of Accuracy
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
from datetime import datetime
from typing import Optional
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import platform
import warnings
import logging
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure matplotlib for Korean fonts
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
    else:
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
    
    plt.rcParams['axes.unicode_minus'] = False
    return korean_font

# Set up visualization
setup_korean_font()
plt.style.use('default')
sns.set_palette("husl")

def calculate_high_risk_recall(y_true, y_pred):
    """Compute high-risk recall for classes >= 2 (medium/high risk)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 2:
        high_risk_mask = y_true >= 2
        if np.sum(high_risk_mask) > 0:
            return recall_score(y_true[high_risk_mask], y_pred[high_risk_mask], average='macro', zero_division=0)
    return 0.0

def load_and_prepare_selected_data():
    """Load and prepare Step 2 data with proper temporal sorting"""
    print("🚀 CORRECTED TEMPORAL VALIDATION")
    print("=" * 50)
    print("📂 Loading Step 2 Optimized Dataset")
    print("-" * 40)
    
    # Load dataset from Step 2
    df = pd.read_csv('../data/processed/credit_risk_dataset_selected.csv')
    print(f"✅ Step 2 dataset loaded: {df.shape}")
    
    # CRITICAL: Sort by 보험청약일자 for temporal validation
    if '보험청약일자' in df.columns:
        df = df.sort_values('보험청약일자').reset_index(drop=True)
        print(f"✅ Data sorted by 보험청약일자 for temporal validation")
    else:
        print("⚠️ 보험청약일자 not found - using index order for temporal validation")
    
    # Define exclude columns (matching step2.py)
    exclude_cols = [
        '사업자등록번호', '대상자명', '청약번호', '보험청약일자', '수출자대상자번호', '업종코드1'
    ]
    
    # Separate features and targets
    target_cols = [col for col in df.columns if col.startswith('risk_year')]
    feature_cols = [col for col in df.columns if col not in target_cols + exclude_cols]
    
    print(f"📋 Excluded columns: {len(exclude_cols)}")
    print(f"🎯 Target columns: {len(target_cols)}")
    print(f"📊 Optimized features: {len(feature_cols)}")
    
    # Show target availability
    print(f"\n🎯 Target Variable Availability:")
    for target in target_cols:
        non_null_count = df[target].notna().sum()
        print(f"   • {target}: {non_null_count:,} records ({non_null_count/len(df)*100:.1f}%)")
    
    X = df[feature_cols]
    y = df[target_cols]
    
    return df, X, y, exclude_cols, target_cols

def approach1_random_split(X, y, test_size=0.2):
    """Approach 1: Random Split (shuffle=True) - Performance baseline ignoring temporal characteristics"""
    print(f"\n📊 Approach 1: Random Split (test_size={test_size})")
    print("-" * 60)
    print("🎯 Purpose: Performance baseline when ignoring time-series characteristics")
    
    # Use first target for validation
    y_single = y.iloc[:, 0] if len(y.shape) > 1 else y
    
    # Remove NaN targets
    mask = ~pd.isna(y_single)
    X_clean = X[mask]
    y_clean = y_single[mask]
    
    # Get numeric features (preserve NaNs for XGBoost's native missing handling)
    X_numeric = X_clean.select_dtypes(include=[np.number])
    
    if len(X_numeric.columns) == 0:
        print("⚠️ No numeric features for validation")
        return None
    
    # Random train/test split (shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y_clean, test_size=test_size, random_state=42, shuffle=True
    )
    
    print(f"   • Training samples: {len(X_train):,}")
    print(f"   • Test samples: {len(X_test):,}")
    print(f"   • Features: {X_train.shape[1]}")
    
    # Train and evaluate
    model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
        tree_method='hist',
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'high_risk_recall': calculate_high_risk_recall(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0)
    }
    
    print(f"   🎯 F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"   🔥 High-Risk Recall: {metrics['high_risk_recall']:.4f}")
    
    return {
        'approach': 'Random Split',
        'description': 'Performance baseline ignoring temporal characteristics',
        'metrics': metrics,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': X_train.shape[1]
    }

def approach2_stratified_random_split(X, y, test_size=0.2):
    """Approach 2: Stratified Random Split (shuffle=True, stratify=y) - Baseline with class balance"""
    print(f"\n📊 Approach 2: Stratified Random Split (test_size={test_size})")
    print("-" * 60)
    print("🎯 Purpose: Baseline with class balance (non-temporal)")
    
    # Use first target for validation
    y_single = y.iloc[:, 0] if len(y.shape) > 1 else y
    
    # Remove NaN targets
    mask = ~pd.isna(y_single)
    X_clean = X[mask]
    y_clean = y_single[mask]
    
    # Get numeric features (preserve NaNs)
    X_numeric = X_clean.select_dtypes(include=[np.number])
    
    if len(X_numeric.columns) == 0:
        print("⚠️ No numeric features for validation")
        return None
    
    # Stratified random split (shuffle=True, stratify)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y_clean, test_size=test_size, random_state=42, shuffle=True, stratify=y_clean
        )
    except ValueError as e:
        print(f"⚠️ Stratified split failed ({e}); falling back to non-stratified random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y_clean, test_size=test_size, random_state=42, shuffle=True
        )
    
    print(f"   • Training samples: {len(X_train):,}")
    print(f"   • Test samples: {len(X_test):,}")
    print(f"   • Features: {X_train.shape[1]}")
    
    # Train and evaluate
    model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
        tree_method='hist',
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'high_risk_recall': calculate_high_risk_recall(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0)
    }
    
    print(f"   🎯 F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"   🔥 High-Risk Recall: {metrics['high_risk_recall']:.4f}")
    
    return {
        'approach': 'Stratified Random Split',
        'description': 'Baseline with class balance (non-temporal)',
        'metrics': metrics,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': X_train.shape[1]
    }

def approach3_temporal_holdout(X, y, test_size=0.2):
    """Approach 3: Simple Temporal Holdout (shuffle=False) - Basic temporal validation (past vs future)"""
    print(f"\n📊 Approach 3: Simple Temporal Holdout (test_size={test_size})")
    print("-" * 60)
    print("🎯 Purpose: Basic temporal validation separating past from future")

    # Use first target for validation
    y_single = y.iloc[:, 0] if len(y.shape) > 1 else y

    # Remove NaN targets
    mask = ~pd.isna(y_single)
    X_clean = X[mask]
    y_clean = y_single[mask]

    # Get numeric features (preserve NaNs)
    X_numeric = X_clean.select_dtypes(include=[np.number])

    if len(X_numeric.columns) == 0:
        print("⚠️ No numeric features for validation")
        return None

    # Temporal train/test split (shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y_clean, test_size=test_size, random_state=42, shuffle=False
    )

    print(f"   • Training samples: {len(X_train):,} (earlier data)")
    print(f"   • Test samples: {len(X_test):,} (later data)")
    print(f"   • Features: {X_train.shape[1]}")

    # Train and evaluate
    model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
        tree_method='hist',
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'high_risk_recall': calculate_high_risk_recall(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0)
    }

    print(f"   🎯 F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"   🔥 High-Risk Recall: {metrics['high_risk_recall']:.4f}")

    return {
        'approach': 'Simple Temporal Holdout',
        'description': 'Basic temporal validation separating past from future',
        'metrics': metrics,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': X_train.shape[1]
    }

def approach4_expanding_window_cv(X, y, n_splits=5):
    """Approach 4: Expanding Window CV - Evaluates stability as more data accumulates"""
    print(f"\n📊 Approach 4: Expanding Window CV (n_splits={n_splits})")
    print("-" * 60)
    print("🎯 Purpose: Evaluates stability as more data accumulates over time")
    
    # Use first target for validation
    y_single = y.iloc[:, 0] if len(y.shape) > 1 else y
    
    # Remove NaN targets
    mask = ~pd.isna(y_single)
    X_clean = X[mask]
    y_clean = y_single[mask]
    
    # Get numeric features (preserve NaNs)
    X_numeric = X_clean.select_dtypes(include=[np.number])
    
    if len(X_numeric.columns) == 0:
        print("⚠️ No numeric features for validation")
        return None
    
    # Use TimeSeriesSplit for expanding window
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    metrics_list = []
    train_sizes = []
    
    print(f"   • Total samples: {len(X_numeric):,}")
    print(f"   • Features: {X_numeric.shape[1]}")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_numeric)):
        X_train, X_test = X_numeric.iloc[train_idx], X_numeric.iloc[test_idx]
        y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
        
        train_sizes.append(len(X_train))
        
        # Train and evaluate
        model = xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='mlogloss'
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'high_risk_recall': calculate_high_risk_recall(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0)
        }
        metrics_list.append(metrics)
        
        print(f"   Fold {fold+1}: Train={len(X_train):,}, Test={len(X_test):,}, F1={metrics['f1_macro']:.4f}")
    
    # Calculate average metrics
    avg_metrics = {}
    for key in metrics_list[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in metrics_list])
    
    print(f"   🎯 Average F1-Score (Macro): {avg_metrics['f1_macro']:.4f}")
    print(f"   🔥 Average High-Risk Recall: {avg_metrics['high_risk_recall']:.4f}")
    
    return {
        'approach': 'Expanding Window CV',
        'description': 'Evaluates stability as more data accumulates over time',
        'metrics': avg_metrics,
        'fold_metrics': metrics_list,
        'train_sizes': train_sizes,
        'n_splits': n_splits,
        'features': X_numeric.shape[1]
    }

def approach5_rolling_window_cv(X, y, window_size=0.6, n_splits=5):
    """Approach 5: Rolling Window CV - Evaluates adaptation to recent trends"""
    print(f"\n📊 Approach 5: Rolling Window CV (window_size={window_size}, n_splits={n_splits})")
    print("-" * 60)
    print("🎯 Purpose: Evaluates adaptation to recent trends by discarding old data")
    
    # Use first target for validation
    y_single = y.iloc[:, 0] if len(y.shape) > 1 else y
    
    # Remove NaN targets
    mask = ~pd.isna(y_single)
    X_clean = X[mask]
    y_clean = y_single[mask]
    
    # Get numeric features (preserve NaNs)
    X_numeric = X_clean.select_dtypes(include=[np.number])
    
    if len(X_numeric.columns) == 0:
        print("⚠️ No numeric features for validation")
        return None
    
    total_samples = len(X_numeric)
    window_samples = int(total_samples * window_size)
    step_size = (total_samples - window_samples) // n_splits
    
    print(f"   • Total samples: {total_samples:,}")
    print(f"   • Window size: {window_samples:,} samples")
    print(f"   • Step size: {step_size:,} samples")
    print(f"   • Features: {X_numeric.shape[1]}")
    
    metrics_list = []
    train_sizes = []
    
    for fold in range(n_splits):
        start_idx = fold * step_size
        end_idx = start_idx + window_samples
        
        if end_idx >= total_samples:
            break
            
        train_idx = list(range(start_idx, end_idx))
        test_idx = list(range(end_idx, min(end_idx + step_size, total_samples)))
        
        if len(test_idx) == 0:
            break
            
        X_train, X_test = X_numeric.iloc[train_idx], X_numeric.iloc[test_idx]
        y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
        
        train_sizes.append(len(X_train))
        
        # Train and evaluate
        model = xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='mlogloss'
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'high_risk_recall': calculate_high_risk_recall(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0)
        }
        metrics_list.append(metrics)
        
        print(f"   Fold {fold+1}: Train={len(X_train):,}, Test={len(X_test):,}, F1={metrics['f1_macro']:.4f}")
    
    if not metrics_list:
        print("⚠️ No valid folds generated")
        return None
    
    # Calculate average metrics
    avg_metrics = {}
    for key in metrics_list[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in metrics_list])
    
    print(f"   🎯 Average F1-Score (Macro): {avg_metrics['f1_macro']:.4f}")
    print(f"   🔥 Average High-Risk Recall: {avg_metrics['high_risk_recall']:.4f}")
    
    return {
        'approach': 'Rolling Window CV',
        'description': 'Evaluates adaptation to recent trends by discarding old data',
        'metrics': avg_metrics,
        'fold_metrics': metrics_list,
        'train_sizes': train_sizes,
        'n_splits': len(metrics_list),
        'features': X_numeric.shape[1]
    }

def test_all_approaches_for_target(X, y, target_name):
    """Test all 5 validation approaches for a specific target"""
    print(f"\n🎯 TESTING ALL APPROACHES FOR {target_name}")
    print("=" * 70)
    
    # Filter data for this specific target
    mask = ~pd.isna(y[target_name])
    X_target = X[mask]
    y_target = y[target_name][mask]
    
    print(f"📊 Available data for {target_name}: {len(X_target):,} samples")
    
    # Test all 5 approaches
    results = {}

    # Approach 1: Random Split
    results['random_split'] = approach1_random_split(X_target, y_target, test_size=0.2)

    # Approach 2: Stratified Random Split
    results['stratified_random'] = approach2_stratified_random_split(X_target, y_target, test_size=0.2)

    # Approach 3: Simple Temporal Holdout
    results['temporal_holdout'] = approach3_temporal_holdout(X_target, y_target, test_size=0.2)

    # Approach 4: Expanding Window CV
    results['expanding_window'] = approach4_expanding_window_cv(X_target, y_target, n_splits=5)

    # Approach 5: Rolling Window CV
    results['rolling_window'] = approach5_rolling_window_cv(X_target, y_target, window_size=0.6, n_splits=5)
    
    return results

def create_comprehensive_visualization(all_results, results_dir):
    """Create comprehensive visualization for all targets and approaches"""
    print(f"\n📊 Creating Comprehensive Visualizations")
    print("-" * 50)
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    if not all_results:
        print("⚠️ No results to visualize")
        return
    
    # Extract data for visualization
    targets = list(all_results.keys())
    approaches = ['Random Split', 'Stratified Random Split', 'Simple Temporal Holdout', 'Expanding Window CV', 'Rolling Window CV']
    
    # Create F1-Score comparison across all targets
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Step 3: Temporal Validation Strategy Comparison - All Targets', fontsize=16, fontweight='bold')
    
    for idx, target in enumerate(targets):
        ax = [ax1, ax2, ax3, ax4][idx]
        
        f1_scores = []
        approach_names = []
        
        for approach_key in ['random_split', 'stratified_random', 'temporal_holdout', 'expanding_window', 'rolling_window']:
            if all_results[target][approach_key] is not None:
                f1_scores.append(all_results[target][approach_key]['metrics']['f1_macro'])
                approach_names.append(all_results[target][approach_key]['approach'])
        
        if f1_scores:
            color_palette = ['lightcoral', 'lightblue', 'plum', 'lightgreen', 'gold']
            bars = ax.bar(approach_names, f1_scores, color=color_palette[:len(approach_names)], alpha=0.8)
            ax.set_ylabel('F1-Score (Macro)')
            ax.set_title(f'{target} Performance Comparison')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/step3_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Comprehensive visualization created successfully")

def select_best_unified_approach(all_results):
    """Select the best unified validation approach across all targets"""
    print(f"\n🎯 SELECTING BEST UNIFIED VALIDATION APPROACH")
    print("-" * 60)
    
    # Calculate average performance for each approach across all targets
    approach_scores = {
        'Random Split': [],
        'Stratified Random Split': [],
        'Simple Temporal Holdout': [],
        'Expanding Window CV': [],
        'Rolling Window CV': []
    }
    
    print("📊 Performance Summary by Approach:")
    print("-" * 40)
    
    for target, results in all_results.items():
        print(f"\n🎯 {target}:")
        for approach_key, approach_name in [
            ('random_split', 'Random Split'),
            ('stratified_random', 'Stratified Random Split'),
            ('temporal_holdout', 'Simple Temporal Holdout'),
            ('expanding_window', 'Expanding Window CV'),
            ('rolling_window', 'Rolling Window CV')
        ]:
            if results[approach_key] is not None:
                f1_score = results[approach_key]['metrics']['f1_macro']
                approach_scores[approach_name].append(f1_score)
                print(f"   • {approach_name}: {f1_score:.4f}")
    
    # Calculate average and stability metrics
    approach_summary = {}
    for approach, scores in approach_scores.items():
        if scores:
            approach_summary[approach] = {
                'mean_f1': np.mean(scores),
                'std_f1': np.std(scores),
                'min_f1': np.min(scores),
                'max_f1': np.max(scores),
                'stability_score': np.mean(scores) - np.std(scores)  # Higher is better
            }
    
    # Sort by stability score (prioritizing consistency over peak performance)
    sorted_approaches = sorted(approach_summary.items(), 
                              key=lambda x: x[1]['stability_score'], reverse=True)
    
    print(f"\n🏆 APPROACH RANKING (by Stability Score):")
    print("-" * 50)
    for i, (approach, metrics) in enumerate(sorted_approaches, 1):
        print(f"   {i}. {approach}:")
        print(f"      • Mean F1-Score: {metrics['mean_f1']:.4f}")
        print(f"      • Std F1-Score: {metrics['std_f1']:.4f}")
        print(f"      • Range: {metrics['min_f1']:.4f} - {metrics['max_f1']:.4f}")
        print(f"      • Stability Score: {metrics['stability_score']:.4f}")
    
    best_approach = sorted_approaches[0][0]
    best_metrics = sorted_approaches[0][1]
    
    print(f"\n🏆 SELECTED UNIFIED APPROACH: {best_approach}")
    print(f"   🎯 Mean F1-Score: {best_metrics['mean_f1']:.4f}")
    print(f"   📊 Stability Score: {best_metrics['stability_score']:.4f}")
    print(f"   📈 Performance Range: {best_metrics['min_f1']:.4f} - {best_metrics['max_f1']:.4f}")
    
    return {
        'best_approach': best_approach,
        'approach_summary': approach_summary,
        'all_results': all_results
    }

def save_results(all_results, best_selection, results_dir):
    """Save comprehensive results and summary"""
    print(f"\n💾 Saving Comprehensive Results")
    print("-" * 40)
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    with open(f'{results_dir}/step3_comprehensive_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary = {
        'execution_date': datetime.now().isoformat(),
        'approach': 'corrected_temporal_validation',
        'targets_tested': len(all_results),
        'strategies_tested': 5,
        'best_unified_approach': best_selection['best_approach'],
        'approach_summary': best_selection['approach_summary'],
        'key_insights': {
            'data_sorted': 'Data sorted by 보험청약일자 for temporal validation',
            'consistent_parameters': 'Same split ratios and fold counts across all targets',
            'unified_strategy': 'One best approach selected for all targets',
            'stability_priority': 'Stability and consistency prioritized over peak performance'
        }
    }
    
    with open(f'{results_dir}/step3_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to: {results_dir}")
    print(f"   📄 Comprehensive results: step3_comprehensive_results.json")
    print(f"   📋 Summary: step3_summary.json")
    print(f"   📊 Visualization: step3_comprehensive_comparison.png")
    
    return results_dir

def main():
    """Main execution - Corrected Temporal Validation"""
    
    # Load and prepare Step 2 data with proper temporal sorting
    df, X, y, exclude_cols, target_cols = load_and_prepare_selected_data()
    
    print(f"\n🔬 CORRECTED TEMPORAL VALIDATION ENGINE")
    print("=" * 60)
    print(f"📊 Testing 5 validation strategies across {len(target_cols)} targets")
    print(f"🎯 Targets: {target_cols}")
    print(f"📈 Features: {X.shape[1]}")
    print(f"📅 Data sorted by 보험청약일자 for temporal validation")
    
    # Test all approaches for each target
    all_results = {}
    
    for target_name in target_cols:
        results = test_all_approaches_for_target(X, y, target_name)
        all_results[target_name] = results
    
    # Create comprehensive visualization
    results_dir = 'result/step3_temporal_validation'
    create_comprehensive_visualization(all_results, results_dir)
    
    # Select best unified approach
    best_selection = select_best_unified_approach(all_results)
    
    # Save results
    save_results(all_results, best_selection, results_dir)
    
    print(f"\n🎉 CORRECTED TEMPORAL VALIDATION COMPLETED!")
    print("=" * 60)
    print("✅ All 4 validation strategies tested across all targets")
    print("✅ Data properly sorted for temporal validation")
    print("✅ Consistent parameters maintained across targets")
    print("✅ Best unified approach selected for all targets")
    print(f"✅ Results saved in: {results_dir}")
    
    return best_selection

if __name__ == "__main__":
    best_validation = main() 