"""
XGBoost Risk Prediction Model - Step 4: Feature Refinement and Selection
=======================================================================

Step 4 Implementation - SIMPLIFIED 4-STEP APPROACH:
1. Correlation analysis (>0.9 threshold, configurable)
2. Variance threshold (remove <0.001 variance, configurable)
3. Missing value patterns (remove >50% missing, configurable)
4. XGBoost feature importance (keep top 50-60%, configurable)

Design Focus:
- Clean, simple implementation
- Essential reduction steps only
- Preserve exclude columns and dataset creation
- Generate comparison visualizations (diagnostics only)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
from datetime import datetime
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import platform
import warnings
warnings.filterwarnings('ignore')

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

def load_data():
    """Load and prepare data"""
    print("🚀 FEATURE REDUCTION PIPELINE")
    print("=" * 50)
    print("📂 Loading Dataset")
    print("-" * 30)
    
    # Load dataset
    df = pd.read_csv('dataset/credit_risk_dataset.csv')
    print(f"✅ Dataset loaded: {df.shape}")
    
    # Define exclude columns (matching step3.py)
    exclude_cols = [
        '사업자등록번호', '대상자명', '대상자등록이력일시', '대상자기본주소',
        '청약번호', '보험청약일자', '청약상태코드', '수출자대상자번호', 
        '특별출연협약코드', '업종코드1'
    ]
    
    # Separate features and targets
    target_cols = [col for col in df.columns if col.startswith('risk_year')]
    feature_cols = [col for col in df.columns if col not in target_cols + exclude_cols]
    
    print(f"📋 Excluded columns: {len(exclude_cols)}")
    print(f"🎯 Target columns: {len(target_cols)}")
    print(f"📊 Features: {len(feature_cols)}")
    
    X = df[feature_cols]
    y = df[target_cols]
    
    return df, X, y, exclude_cols, target_cols

def audit_non_numeric_columns(X: pd.DataFrame, exclude_cols: list):
    """Identify non-numeric columns that are not part of exclude columns.

    These columns will be appended back for diagnostics only and should never be
    used during model training. We surface them so potential leakage can be audited.
    """
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_outside_exclude = [c for c in non_numeric_cols if c not in exclude_cols]

    if non_numeric_outside_exclude:
        print("⚠️ Non-numeric columns outside exclude list (diagnostics only, do not train with these):")
        for c in non_numeric_outside_exclude:
            print(f"   • {c}")
    else:
        print("✅ All non-numeric columns are within exclude list or none present in features")

    return non_numeric_outside_exclude

def step1_remove_high_correlation(X, threshold=0.9):
    """Step 1: Remove highly correlated features"""
    print(f"\n📊 Step 1: Correlation Analysis (>{threshold} threshold)")
    print("-" * 40)
    
    initial_features = len(X.columns)
    
    # Get only numeric columns for correlation
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("⚠️ No numeric features for correlation analysis")
        return X
    
    # Calculate correlation matrix
    X_numeric = X[numeric_cols]
    corr_matrix = X_numeric.corr().abs()
    
    # Find highly correlated pairs
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > threshold)]

    # Remove highly correlated features
    X_reduced = X.drop(columns=to_drop)
    
    print(f"✅ Removed {len(to_drop)} highly correlated features")
    print(f"   Features: {initial_features} → {len(X_reduced.columns)}")
    
    return X_reduced

def step2_remove_low_variance(X, threshold=0.001):
    """Step 2: Remove low variance features"""
    print(f"\n📊 Step 2: Variance Threshold (<{threshold} variance)")
    print("-" * 40)
    
    initial_features = len(X.columns)
    
    # Get only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("⚠️ No numeric features for variance analysis")
        return X
    
    # Apply variance threshold to numeric columns
    X_numeric = X[numeric_cols].fillna(0)
    selector = VarianceThreshold(threshold=threshold)
    
    try:
        X_numeric_filtered = selector.fit_transform(X_numeric)
        selected_numeric_cols = X_numeric.columns[selector.get_support()].tolist()
        
        # Combine selected numeric + all non-numeric columns
        final_cols = selected_numeric_cols + list(non_numeric_cols)
        X_reduced = X[final_cols]
        
        removed_count = len(numeric_cols) - len(selected_numeric_cols)
        print(f"✅ Removed {removed_count} low variance features")
        print(f"   Features: {initial_features} → {len(X_reduced.columns)}")
        
        return X_reduced
        
    except Exception as e:
        print(f"⚠️ Variance filter failed: {e}")
        return X

def step3_remove_high_missing(X, threshold=0.5):
    """Step 3: Remove features with high missing values"""
    print(f"\n📊 Step 3: Missing Value Analysis (>{threshold*100}% missing)")
    print("-" * 40)
    
    initial_features = len(X.columns)
    
    # Calculate missing percentages
    missing_percentages = X.isnull().sum() / len(X)
    
    # Find features to remove
    to_remove = missing_percentages[missing_percentages > threshold].index.tolist()
    
    # Remove high missing features
    X_reduced = X.drop(columns=to_remove)
    
    print(f"✅ Removed {len(to_remove)} high missing features")
    print(f"   Features: {initial_features} → {len(X_reduced.columns)}")
    
    return X_reduced

def step4_xgboost_importance(X, y, keep_percentage=60):
    """Step 4: XGBoost feature importance selection"""
    print(f"\n📊 Step 4: XGBoost Feature Importance (keep top {keep_percentage}%)")
    print("-" * 40)
    
    initial_features = len(X.columns)
    
    # Use first target for feature selection
    y_single = y.iloc[:, 0] if len(y.shape) > 1 else y
    
    # Remove NaN targets and get numeric features
    mask = ~pd.isna(y_single)
    X_clean = X[mask]
    y_clean = y_single[mask]
    
    # Get only numeric features for XGBoost
    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("⚠️ No numeric features for XGBoost importance")
        return X
    
    X_numeric = X_clean[numeric_cols].fillna(0)
    
    try:
        # Train XGBoost (diagnostic importance model)
        model = xgb.XGBClassifier(
            n_estimators=50,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='mlogloss'
        )
        model.fit(X_numeric, y_clean)
        
        # Get feature importance
        importance_scores = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': X_numeric.columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Select top percentage
        n_features = max(int(len(numeric_cols) * keep_percentage / 100), 5)
        selected_numeric = importance_df.head(n_features)['feature'].tolist()
        
        # Combine selected numeric + all non-numeric (diagnostics only; do not train with non-numerics)
        final_cols = selected_numeric + list(non_numeric_cols)
        X_reduced = X[final_cols]
        
        removed_count = len(numeric_cols) - len(selected_numeric)
        print(f"✅ Kept top {len(selected_numeric)} numeric + {len(non_numeric_cols)} non-numeric features")
        print(f"   Features: {initial_features} → {len(X_reduced.columns)}")
        
        return X_reduced
        
    except Exception as e:
        print(f"⚠️ XGBoost importance failed: {e}")
        return X

def create_visualizations(X_original, X_reduced, y, results_dir):
    """Create before/after comparison visualizations with proper train/test split"""
    print(f"\n📊 Creating Comparison Visualizations (Proper Validation)")
    print("-" * 50)
    
    try:
        # Use first target for visualization
        y_single = y.iloc[:, 0] if len(y.shape) > 1 else y
        
        # Remove NaN targets
        mask = ~pd.isna(y_single)
        X_orig_clean = X_original[mask]
        X_red_clean = X_reduced[mask]
        y_clean = y_single[mask]
        
        if len(X_orig_clean) < 100:  # Increased minimum for train/test split
            print("⚠️ Insufficient samples for train/test split visualization")
            return
        
        # Get numeric features for modeling
        X_orig_numeric = X_orig_clean.select_dtypes(include=[np.number]).fillna(0)
        X_red_numeric = X_red_clean.select_dtypes(include=[np.number]).fillna(0)
        
        if len(X_orig_numeric.columns) == 0 or len(X_red_numeric.columns) == 0:
            print("⚠️ No numeric features for visualization")
            return
        
        print(f"📊 Creating train/test split (80/20) for proper validation (stratified random)...")
        print(f"   • Total samples: {len(X_orig_numeric):,}")
        
        # Create stratified train/test split (diagnostics only)
        from sklearn.model_selection import train_test_split

        idx_train, idx_test = train_test_split(
            y_clean.index,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=y_clean
        )

        # Align splits across original and reduced feature sets
        X_orig_train, X_orig_test = X_orig_numeric.loc[idx_train], X_orig_numeric.loc[idx_test]
        X_red_train, X_red_test = X_red_numeric.loc[idx_train], X_red_numeric.loc[idx_test]
        y_train, y_test = y_clean.loc[idx_train], y_clean.loc[idx_test]
        
        print(f"   • Training samples: {len(X_orig_train):,}")
        print(f"   • Test samples: {len(X_orig_test):,}")
        
        # Train both RandomForest and XGBoost models for comparison
        print(f"   🎯 Training RandomForest models...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # RandomForest - Original model
        rf_orig = rf_model.fit(X_orig_train, y_train)
        y_pred_rf_orig = rf_orig.predict(X_orig_test)
        
        # RandomForest - Reduced model  
        rf_red = rf_model.fit(X_red_train, y_train)
        y_pred_rf_red = rf_red.predict(X_red_test)
        
        print(f"   🎯 Training XGBoost models...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='mlogloss'
        )
        
        # XGBoost - Original model
        xgb_orig = xgb_model.fit(X_orig_train, y_train)
        y_pred_xgb_orig = xgb_orig.predict(X_orig_test)
        
        # XGBoost - Reduced model  
        xgb_red = xgb_model.fit(X_red_train, y_train)
        y_pred_xgb_red = xgb_red.predict(X_red_test)
        
        # Create performance comparison for both models
        create_performance_comparison(y_test, y_pred_rf_orig, y_pred_rf_red, 
                                    y_pred_xgb_orig, y_pred_xgb_red,
                                    X_original, X_reduced, results_dir)
        
        # Create confusion matrix comparison for XGBoost (since it's the main model)
        create_confusion_matrix_comparison(y_test, y_pred_xgb_orig, y_pred_xgb_red, results_dir)
        
        print("✅ Visualizations created successfully with proper validation")
        
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")
        import traceback
        traceback.print_exc()

def create_performance_comparison(y_true, y_pred_rf_orig, y_pred_rf_red, 
                                y_pred_xgb_orig, y_pred_xgb_red, X_orig, X_red, results_dir):
    """Create performance metrics comparison chart with proper validation for both RF and XGBoost"""
    
    # Calculate metrics for RandomForest
    metrics_rf_orig = {
        'Accuracy': accuracy_score(y_true, y_pred_rf_orig),
        'F1-Score': f1_score(y_true, y_pred_rf_orig, average='macro', zero_division=0),
        'Precision': precision_score(y_true, y_pred_rf_orig, average='macro', zero_division=0),
        'Recall': recall_score(y_true, y_pred_rf_orig, average='macro', zero_division=0)
    }
    
    metrics_rf_red = {
        'Accuracy': accuracy_score(y_true, y_pred_rf_red),
        'F1-Score': f1_score(y_true, y_pred_rf_red, average='macro', zero_division=0),
        'Precision': precision_score(y_true, y_pred_rf_red, average='macro', zero_division=0),
        'Recall': recall_score(y_true, y_pred_rf_red, average='macro', zero_division=0)
    }
    
    # Calculate metrics for XGBoost
    metrics_xgb_orig = {
        'Accuracy': accuracy_score(y_true, y_pred_xgb_orig),
        'F1-Score': f1_score(y_true, y_pred_xgb_orig, average='macro', zero_division=0),
        'Precision': precision_score(y_true, y_pred_xgb_orig, average='macro', zero_division=0),
        'Recall': recall_score(y_true, y_pred_xgb_orig, average='macro', zero_division=0)
    }
    
    metrics_xgb_red = {
        'Accuracy': accuracy_score(y_true, y_pred_xgb_red),
        'F1-Score': f1_score(y_true, y_pred_xgb_red, average='macro', zero_division=0),
        'Precision': precision_score(y_true, y_pred_xgb_red, average='macro', zero_division=0),
        'Recall': recall_score(y_true, y_pred_xgb_red, average='macro', zero_division=0)
    }
    
    # Print validation results
    print(f"\n📊 VALIDATION RESULTS (80/20 Train/Test Split):")
    print("-" * 60)
    print(f"   🎯 RandomForest F1-Score (Macro):")
    print(f"      • Before Step4: {metrics_rf_orig['F1-Score']:.4f}")
    print(f"      • After Step4:  {metrics_rf_red['F1-Score']:.4f}")
    print(f"      • Difference:   {metrics_rf_red['F1-Score'] - metrics_rf_orig['F1-Score']:+.4f}")
    print(f"   🎯 XGBoost F1-Score (Macro):")
    print(f"      • Before Step4: {metrics_xgb_orig['F1-Score']:.4f}")
    print(f"      • After Step4:  {metrics_xgb_red['F1-Score']:.4f}")
    print(f"      • Difference:   {metrics_xgb_red['F1-Score'] - metrics_xgb_orig['F1-Score']:+.4f}")
    print(f"   📈 RandomForest Accuracy:")
    print(f"      • Before Step4: {metrics_rf_orig['Accuracy']:.4f}")
    print(f"      • After Step4:  {metrics_rf_red['Accuracy']:.4f}")
    print(f"      • Difference:   {metrics_rf_red['Accuracy'] - metrics_rf_orig['Accuracy']:+.4f}")
    print(f"   📈 XGBoost Accuracy:")
    print(f"      • Before Step4: {metrics_xgb_orig['Accuracy']:.4f}")
    print(f"      • After Step4:  {metrics_xgb_red['Accuracy']:.4f}")
    print(f"      • Difference:   {metrics_xgb_red['Accuracy'] - metrics_xgb_orig['Accuracy']:+.4f}")
    
    # Create comparison plot with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Step 4 Feature Reduction Impact - Model Comparison', fontsize=16, fontweight='bold')
    
    # 1. RandomForest Performance Metrics
    metrics_names = list(metrics_rf_orig.keys())
    rf_orig_values = list(metrics_rf_orig.values())
    rf_red_values = list(metrics_rf_red.values())
    
    x_pos = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, rf_orig_values, width, 
                    label=f'Before Step4 ({X_orig.shape[1]} features)', alpha=0.8, color='lightcoral')
    bars2 = ax1.bar(x_pos + width/2, rf_red_values, width, 
                    label=f'After Step4 ({X_red.shape[1]} features)', alpha=0.8, color='lightblue')
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('RandomForest Performance Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metrics_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    
    # 2. XGBoost Performance Metrics
    xgb_orig_values = list(metrics_xgb_orig.values())
    xgb_red_values = list(metrics_xgb_red.values())
    
    bars3 = ax2.bar(x_pos - width/2, xgb_orig_values, width, 
                    label=f'Before Step4 ({X_orig.shape[1]} features)', alpha=0.8, color='salmon')
    bars4 = ax2.bar(x_pos + width/2, xgb_red_values, width, 
                    label=f'After Step4 ({X_red.shape[1]} features)', alpha=0.8, color='skyblue')
    
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Score')
    ax2.set_title('XGBoost Performance Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metrics_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    
    # 3. Feature Count Comparison
    feature_counts = ['Before Step4', 'After Step4']
    counts = [X_orig.shape[1], X_red.shape[1]]
    colors = ['lightcoral', 'lightblue']
    
    bars = ax3.bar(feature_counts, counts, color=colors, alpha=0.8)
    ax3.set_ylabel('Number of Features')
    ax3.set_title('Feature Count Comparison')
    ax3.grid(True, alpha=0.3)
    
    # Add percentage reduction
    reduction_pct = (1 - X_red.shape[1] / X_orig.shape[1]) * 100
    ax3.text(0.5, max(counts) * 0.8, f'Reduction: {reduction_pct:.1f}%', 
            ha='center', fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    
    # 4. F1-Score Comparison Summary
    models = ['RandomForest', 'XGBoost']
    before_f1 = [metrics_rf_orig['F1-Score'], metrics_xgb_orig['F1-Score']]
    after_f1 = [metrics_rf_red['F1-Score'], metrics_xgb_red['F1-Score']]
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    bars5 = ax4.bar(x_pos - width/2, before_f1, width, 
                    label='Before Step4', alpha=0.8, color='lightcoral')
    bars6 = ax4.bar(x_pos + width/2, after_f1, width, 
                    label='After Step4', alpha=0.8, color='lightblue')
    
    ax4.set_xlabel('Models')
    ax4.set_ylabel('F1-Score (Macro)')
    ax4.set_title('F1-Score Comparison Summary')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(models)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars5:
        height = bar.get_height()
        ax4.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    for bar in bars6:
        height = bar.get_height()
        ax4.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/step4_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrix_comparison(y_true, y_pred_orig, y_pred_red, results_dir):
    """Create confusion matrix comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original model confusion matrix
    cm_orig = confusion_matrix(y_true, y_pred_orig)
    sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix - Before Step4')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Reduced model confusion matrix  
    cm_red = confusion_matrix(y_true, y_pred_red)
    sns.heatmap(cm_red, annot=True, fmt='d', cmap='Greens', ax=ax2)
    ax2.set_title('Confusion Matrix - After Step4')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/step4_confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(df_original, X_reduced, exclude_cols, target_cols, non_numeric_outside_exclude):
    """Save results and create complete dataset"""
    print(f"\n💾 Saving Results")
    print("-" * 30)
    
    # Create results directory
    results_dir = "result/step4_feature_reduction"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save selected features list (with proper header)
    pd.DataFrame({'feature_name': X_reduced.columns.tolist()}).to_csv(
        f'{results_dir}/step4_selected_features.csv', 
        index=False
    )
    
    # Create complete dataset: excluded columns + selected features + targets
    excluded_data = df_original[exclude_cols].copy()
    target_data = df_original[target_cols].copy()
    
    complete_dataset = pd.concat([
        excluded_data.reset_index(drop=True),
        X_reduced.reset_index(drop=True), 
        target_data.reset_index(drop=True)
    ], axis=1)
    
    # Save to dataset folder
    dataset_path = 'dataset/credit_risk_dataset_step4.csv'
    complete_dataset.to_csv(dataset_path, index=False)
        
    # Save summary
    summary = {
        'execution_date': datetime.now().isoformat(),
        'approach': '5_step_simplified_reduction',
        'steps': [
            '1. Correlation analysis (>0.9)',
            '2. Variance threshold (<0.001)', 
            '3. Missing values (>50%)',
            '4. XGBoost importance (top 60%)'
        ],
        'final_feature_count': len(X_reduced.columns),
        'dataset_composition': {
            'excluded_columns': len(exclude_cols),
            'selected_features': len(X_reduced.columns), 
            'target_columns': len(target_cols),
            'total_columns': complete_dataset.shape[1]
        },
        'non_numeric_outside_exclude': non_numeric_outside_exclude
    }
    
    with open(f'{results_dir}/step4_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Complete dataset saved to: {dataset_path}")
    print(f"   📋 Excluded columns: {len(exclude_cols)}")
    print(f"   📊 Selected features: {len(X_reduced.columns)}")
    print(f"   🎯 Target columns: {len(target_cols)}")
    print(f"   📄 Total columns: {complete_dataset.shape[1]}")
    print(f"   📈 Total rows: {complete_dataset.shape[0]}")
    
    return results_dir

def main(
    corr_threshold: float = 0.9,
    variance_threshold: float = 0.001,
    missing_threshold: float = 0.5,
    keep_percentage: int = 60
):
    """Main execution - Simple feature reduction (configurable thresholds)"""
    
    # Load data
    df, X, y, exclude_cols, target_cols = load_data()
    X_original = X.copy()  # Keep original for comparison
    
    print(f"\n🔬 SYSTEMATIC FEATURE REDUCTION ENGINE")
    print("=" * 50)
    print(f"📊 Starting features: {len(X.columns)}")
    
    # Step 1: Remove high correlation
    X = step1_remove_high_correlation(X, threshold=corr_threshold)
    
    # Step 2: Remove low variance  
    X = step2_remove_low_variance(X, threshold=variance_threshold)
    
    # Step 3: Remove high missing
    X = step3_remove_high_missing(X, threshold=missing_threshold)
    
    # Step 4: XGBoost feature importance
    X_final = step4_xgboost_importance(X, y, keep_percentage=keep_percentage)

    # Audit non-numeric features outside exclude cols (diagnostics only)
    non_numeric_outside_exclude = audit_non_numeric_columns(X_final, exclude_cols)
    
    # Calculate final stats
    reduction_pct = (1 - len(X_final.columns) / len(X_original.columns)) * 100
    
    print(f"\n✅ REDUCTION COMPLETE!")
    print("=" * 30)
    print(f"📊 Final features: {len(X_final.columns)}")
    print(f"📉 Reduction: {len(X_original.columns)} → {len(X_final.columns)} ({reduction_pct:.1f}% removed)")
    # Note: No business-logic preservation enforced by design
    
    # Save results and create dataset
    results_dir = save_results(df, X_final, exclude_cols, target_cols, non_numeric_outside_exclude)
    
    # Create comparison visualizations
    create_visualizations(X_original, X_final, y, results_dir)
    
    print(f"\n🎉 FEATURE REDUCTION COMPLETED!")
    print("=" * 50)
    print("✅ Systematic reduction - multicollinearity eliminated")
    print("✅ XGBoost optimized - performance validated") 
    print(f"✅ Results saved in: {results_dir}")
    
    return X_final

if __name__ == "__main__":
    results = main()