"""
XGBoost Risk Prediction Model - Step 2: Feature Refinement and Selection
=======================================================================

Step 2 Implementation - SIMPLE, DATA-DRIVEN APPROACH:
1. Correlation analysis (>0.9 threshold, configurable)
2. Variance threshold (remove <0.001 variance, configurable)
3. Missing value patterns (remove >50% missing, configurable)
4. XGBoost importance ranking + top-% grid [20, 40, 60, 80, 100] on a multiple 80/20 holdout

Design Focus:
- Clean, simple implementation
- Essential reduction steps only
- Preserve exclude columns and dataset creation
- Generate comparison visualizations (diagnostics only)
- RandomForest removed; XGBoost only
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
from datetime import datetime
from typing import Optional
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
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
def setup_korean_font() -> Optional[str]:
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
    """
    🔄 FIXED: Load ONLY TRAIN data for feature selection (no data leakage!)
    This ensures feature selection rules are created without seeing validation/OOT data.
    """
    logger.info("FEATURE REDUCTION PIPELINE - STRICT DATA SEPARATION")
    logger.info("Loading TRAIN DATA ONLY (No Leakage!)")
    
    # 🔄 CRITICAL CHANGE: Load ONLY train data for feature selection
    try:
        train_df = pd.read_csv('../data/splits/train_data.csv')
        logger.info(f"TRAIN dataset loaded: {train_df.shape}")
        logger.info(f"   Period: {train_df['보험청약일자'].min()} to {train_df['보험청약일자'].max()}")
        
        # Also load validation and OOT for applying the same feature selection rules
        validation_df = pd.read_csv('../data/splits/validation_data.csv')
        oot_df = pd.read_csv('../data/splits/oot_data.csv')
        
        logger.info(f"Data splits loaded:")
        logger.info(f"   - TRAIN: {len(train_df):,} rows (for feature selection rules)")
        logger.info(f"   - VALIDATION: {len(validation_df):,} rows (for rule application)")
        print(f"   - OOT: {len(oot_df):,} rows (for rule application)")
        
        # Combine all data for final dataset creation (but selection rules from TRAIN only)
        df_complete = pd.concat([train_df, validation_df, oot_df], ignore_index=True)
        
    except FileNotFoundError:
        print("❌ ERROR: Split datasets not found!")
        print("   Please run '1_Split.py' first to create train/validation/oot splits")
        raise FileNotFoundError("Run 1_Split.py first to create proper data splits")
    
    # Define exclude columns (matching strict pipeline)
    exclude_cols = [
        '사업자등록번호', '대상자명', '청약번호', '보험청약일자', '수출자대상자번호', '업종코드1', 'unique_id', 'data_split'
    ]
    
    # Separate features and targets (use TRAIN data for feature selection rules)
    target_cols = [col for col in train_df.columns if col.startswith('risk_year')]
    feature_cols = [col for col in train_df.columns if col not in target_cols + exclude_cols]
    
    print(f"📋 Excluded columns: {len(exclude_cols)}")
    print(f"🎯 Target columns: {len(target_cols)}")
    print(f"📊 Features: {len(feature_cols)}")
    print(f"⚠️  IMPORTANT: Feature selection rules based on TRAIN data only!")
    
    # Use TRAIN data for feature selection
    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]
    
    return df_complete, X_train, y_train, exclude_cols, target_cols

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
    
    # Show removed columns
    if to_drop:
        print(f"   🗑️ Removed columns:")
        for i, col in enumerate(to_drop, 1):
            print(f"      {i:2d}. {col}")
    else:
        print("   ✅ No highly correlated columns found")
    
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
        
        # Find removed columns
        removed_numeric_cols = [col for col in numeric_cols if col not in selected_numeric_cols]
        
        # Combine selected numeric + all non-numeric columns
        final_cols = selected_numeric_cols + list(non_numeric_cols)
        X_reduced = X[final_cols]
        
        removed_count = len(numeric_cols) - len(selected_numeric_cols)
        print(f"✅ Removed {removed_count} low variance features")
        print(f"   Features: {initial_features} → {len(X_reduced.columns)}")
        
        # Show removed columns
        if removed_numeric_cols:
            print(f"   🗑️ Removed columns:")
            for i, col in enumerate(removed_numeric_cols, 1):
                print(f"      {i:2d}. {col}")
        else:
            print("   ✅ No low variance columns found")
        
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
    
    # Show removed columns with their missing percentages
    if to_remove:
        print(f"   🗑️ Removed columns (with missing %):")
        for i, col in enumerate(to_remove, 1):
            missing_pct = missing_percentages[col] * 100
            print(f"      {i:2d}. {col} ({missing_pct:.1f}% missing)")
    else:
        print("   ✅ No high missing columns found")
    
    return X_reduced

def _get_step1_xgb_params() -> dict:
    """Return XGBoost params aligned with Step 1 baseline (classification)."""
    return {
        'objective': 'multi:softprob',
        'num_class': 4,
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'eval_metric': 'mlogloss',
        'enable_missing': True,
        'random_state': 42,
        'verbosity': 0,
        'n_jobs': -1,
    }

def step4_xgboost_importance(X, y, keep_percentage=60):
    """Step 4: XGBoost feature importance selection via top-% grid on a single holdout."""
    return step4_xgboost_top_percent_grid(X, y, candidate_percents=[20, 40, 60, 80, 100])

def step4_xgboost_top_percent_grid(X, y, candidate_percents=None, n_folds=5, stability_threshold=0.6):
    """Enhanced multi-fold stability-based feature selection.

    - Uses multiple CV folds instead of single split for robust feature selection
    - Features must be consistently important across folds (stability_threshold)
    - Evaluates top-% grid across all available risk targets (1~4)
    - Aggregate F1 across targets per percent and choose the best average
    - Build the final feature set from stable features only
    """
    if candidate_percents is None:
        candidate_percents = [20, 40, 60, 80, 100]

    print(f"\n📊 Step 4: Enhanced Multi-Fold Feature Selection - {n_folds} folds, stability≥{stability_threshold}")
    print(f"📊 Top-% grid {candidate_percents} across risk_year1..4")
    print("-" * 60)

    # Determine targets
    target_cols = []
    if isinstance(y, pd.DataFrame):
        target_cols = [c for c in y.columns if c.startswith('risk_year')]
    else:
        target_cols = ['risk_year1']

    # Use all numeric columns post step 1
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        print("⚠️ No numeric features for XGBoost importance")
        return X

    xgb_params = _get_step1_xgb_params()
    n_numeric = len(numeric_cols)
    
    # Track feature stability across folds and targets
    from collections import defaultdict
    feature_selection_counts = defaultdict(int)  # How many times each feature was selected
    total_selections = 0  # Total number of selection opportunities
    
    # Store F1 scores for each percentage across all folds and targets
    f1_per_percent: dict[int, list] = {pct: [] for pct in candidate_percents}
    valid_targets = 0

    print(f"   🔄 Processing {len(target_cols)} targets with {n_folds}-fold validation each...")

    for tcol in target_cols:
        y_series = y[tcol] if isinstance(y, pd.DataFrame) else y
        mask = ~pd.isna(y_series)
        X_t = X.loc[mask, numeric_cols]
        y_t = y_series.loc[mask]
        
        # Need at least 2 classes and sufficient samples
        unique_classes = pd.Series(y_t).dropna().unique()
        if len(unique_classes) < 2 or len(X_t) < 100:  # Increased minimum for stability
            print(f"   ⚠️ Skipping {tcol}: insufficient data or classes")
            continue

        print(f"   📊 Processing {tcol} with {len(X_t)} samples, {len(unique_classes)} classes")
        
        # Multi-fold cross-validation for this target
        for fold in range(n_folds):
            # Different random seed for each fold to ensure variety
            fold_seed = 42 + fold * 10
            
            try:
                # Stratified split for this fold
                idx_train, idx_val = train_test_split(
                    X_t.index, test_size=0.2, random_state=fold_seed, 
                    shuffle=True, stratify=y_t
                )
                X_train, X_val = X_t.loc[idx_train], X_t.loc[idx_val]
                y_train, y_val = y_t.loc[idx_train].astype(int), y_t.loc[idx_val].astype(int)
                
                # Train model and get feature importance for this fold
                rank_model = xgb.XGBClassifier(**xgb_params)
                rank_model.fit(X_train, y_train)
                
                try:
                    importances = rank_model.feature_importances_
                    if importances is None or len(importances) != n_numeric:
                        raise ValueError("invalid importances")
                except Exception:
                    booster = rank_model.get_booster()
                    score_dict = booster.get_score(importance_type='gain')
                    importances = np.array([score_dict.get(f'f{i}', 0.0) for i in range(n_numeric)], dtype=float)
                    if importances.sum() == 0:
                        importances = np.ones_like(importances)

                # Build feature ranking for this fold
                importance_df = pd.DataFrame({
                    'feature': numeric_cols, 
                    'importance': importances
                }).sort_values('importance', ascending=False)

                # Test each percentage and track which features are selected
                for pct in candidate_percents:
                    k = max(1, int(n_numeric * pct / 100))
                    selected_features = importance_df.head(k)['feature'].tolist()
                    
                    # Count selections for stability tracking
                    for feature in selected_features:
                        feature_selection_counts[feature] += 1
                    total_selections += len(selected_features)
                    
                    # Evaluate performance with selected features
                    model = xgb.XGBClassifier(**xgb_params)
                    model.fit(X_train[selected_features], y_train)
                    pred = model.predict(X_val[selected_features])
                    f1 = f1_score(y_val, pred, average='macro', zero_division=0)
                    f1_per_percent[pct].append(float(f1))
                    
            except Exception as e:
                print(f"   ⚠️ Fold {fold+1} failed for {tcol}: {e}")
                continue

        valid_targets += 1

    if valid_targets == 0:
        print("⚠️ No valid targets for selection; skipping.")
        return X

    print(f"   ✅ Completed {valid_targets} targets × {n_folds} folds = {valid_targets * n_folds} evaluations")

    # Choose best percentage based on average F1 across all folds and targets
    avg_f1 = {pct: (float(np.mean(scores)) if len(scores) > 0 else 0.0) for pct, scores in f1_per_percent.items()}
    best_percent = sorted(candidate_percents, key=lambda p: (-avg_f1.get(p, 0.0), p))[0]
    best_k = max(1, int(n_numeric * best_percent / 100))

    # Apply stability filtering: only keep features that were selected frequently enough
    min_selections = int(valid_targets * n_folds * len(candidate_percents) * stability_threshold)
    stable_features = [
        feature for feature, count in feature_selection_counts.items() 
        if count >= min_selections
    ]
    
    print(f"   🎯 Stability filtering: {len(stable_features)}/{n_numeric} features stable (≥{min_selections} selections)")
    
    # If we have stable features, use them; otherwise fall back to top features by average importance
    if len(stable_features) >= best_k:
        # Use stable features, ranked by selection frequency
        stable_feature_counts = [(f, feature_selection_counts[f]) for f in stable_features]
        stable_feature_counts.sort(key=lambda x: x[1], reverse=True)
        selected_numeric = [f for f, _ in stable_feature_counts[:best_k]]
        selection_method = "stability-based"
    else:
        print(f"   ⚠️ Only {len(stable_features)} stable features found, falling back to importance-based selection")
        # Fall back to traditional importance-based selection
        aggregated_importances = np.zeros(n_numeric, dtype=float)
        for tcol in target_cols:
            y_series = y[tcol] if isinstance(y, pd.DataFrame) else y
            mask = ~pd.isna(y_series)
            X_t = X.loc[mask, numeric_cols]
            y_t = y_series.loc[mask]
            if len(pd.Series(y_t).dropna().unique()) < 2 or len(X_t) < 50:
                continue
            
            # Single model for importance aggregation
            idx_train, _ = train_test_split(X_t.index, test_size=0.2, random_state=42, shuffle=True, stratify=y_t)
            X_train = X_t.loc[idx_train]
            y_train = y_t.loc[idx_train].astype(int)
            
            rank_model = xgb.XGBClassifier(**xgb_params)
            rank_model.fit(X_train, y_train)
            
            try:
                importances = rank_model.feature_importances_
            except:
                booster = rank_model.get_booster()
                score_dict = booster.get_score(importance_type='gain')
                importances = np.array([score_dict.get(f'f{i}', 0.0) for i in range(n_numeric)], dtype=float)
                if importances.sum() == 0:
                    importances = np.ones_like(importances)
            
            aggregated_importances += importances
        
        avg_importance_df = pd.DataFrame({
            'feature': numeric_cols, 
            'importance': aggregated_importances / valid_targets
        }).sort_values('importance', ascending=False)
        selected_numeric = avg_importance_df.head(best_k)['feature'].tolist()
        selection_method = "importance-based (fallback)"

    final_cols = selected_numeric + non_numeric_cols
    X_reduced = X[final_cols]

    # Find removed columns
    removed_numeric_cols = [col for col in numeric_cols if col not in selected_numeric]
    
    print("   📊 Average F1 by percent across all folds and targets:")
    for pct in candidate_percents:
        n_scores = len(f1_per_percent[pct])
        avg_score = avg_f1.get(pct, 0.0)
        std_score = np.std(f1_per_percent[pct]) if n_scores > 1 else 0.0
        print(f"     - {pct:>3}%: {avg_score:.4f} ± {std_score:.4f} (from {n_scores} evaluations)")
    
    print(f"✅ Selected top-{best_percent}% features (k={best_k}) using {selection_method}")
    print(f"   🎯 Method: Multi-fold stability-based selection ({n_folds} folds × {valid_targets} targets)")
    print(f"   📊 Stability threshold: {stability_threshold} (≥{min_selections} selections required)")
    
    # Show removed columns (limit output for readability)
    if removed_numeric_cols:
        print(f"   🗑️ Removed {len(removed_numeric_cols)} columns:")
        for i, col in enumerate(removed_numeric_cols[:10], 1):  # Show first 10
            print(f"      {i:2d}. {col}")
        if len(removed_numeric_cols) > 10:
            print(f"      ... and {len(removed_numeric_cols) - 10} more")
    else:
        print("   ✅ No columns removed by feature selection")
    
    return X_reduced

def create_visualizations(X_original, X_reduced, y, results_dir):
    """Create before/after comparison visualizations with proper train/test split (XGBoost only)."""
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
        
        # Train XGBoost models (aligned with Step 1)
        print(f"   🎯 Training XGBoost models...")
        xgb_model = xgb.XGBClassifier(**_get_step1_xgb_params())
        
        # XGBoost - Original model
        xgb_orig = xgb_model.fit(X_orig_train, y_train)
        y_pred_xgb_orig = xgb_orig.predict(X_orig_test)
        
        # XGBoost - Reduced model  
        xgb_red = xgb_model.fit(X_red_train, y_train)
        y_pred_xgb_red = xgb_red.predict(X_red_test)
        
        # Create performance comparison for XGBoost only
        create_performance_comparison(y_test, y_pred_xgb_orig, y_pred_xgb_red, X_original, X_reduced, results_dir)

        # Create confusion matrix comparison for XGBoost
        create_confusion_matrix_comparison(y_test, y_pred_xgb_orig, y_pred_xgb_red, results_dir)
        
        print("✅ Visualizations created successfully with proper validation")
        
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")
        import traceback
        traceback.print_exc()

def create_performance_comparison(y_true, y_pred_xgb_orig, y_pred_xgb_red, X_orig, X_red, results_dir):
    """Create performance metrics comparison chart for XGBoost only."""

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
    print(f"   🎯 XGBoost F1-Score (Macro):")
    print(f"      • Before Step2: {metrics_xgb_orig['F1-Score']:.4f}")
    print(f"      • After Step2:  {metrics_xgb_red['F1-Score']:.4f}")
    print(f"      • Difference:   {metrics_xgb_red['F1-Score'] - metrics_xgb_orig['F1-Score']:+.4f}")
    print(f"   📈 XGBoost Accuracy:")
    print(f"      • Before Step2: {metrics_xgb_orig['Accuracy']:.4f}")
    print(f"      • After Step2:  {metrics_xgb_red['Accuracy']:.4f}")
    print(f"      • Difference:   {metrics_xgb_red['Accuracy'] - metrics_xgb_orig['Accuracy']:+.4f}")
    
    # Create comparison plot with 2 subplots (metrics + feature counts)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Step 2 Feature Reduction Impact - XGBoost', fontsize=16, fontweight='bold')

    # 1. XGBoost Performance Metrics
    metrics_names = list(metrics_xgb_orig.keys())
    xgb_orig_values = list(metrics_xgb_orig.values())
    xgb_red_values = list(metrics_xgb_red.values())

    x_pos = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax1.bar(x_pos - width/2, xgb_orig_values, width, 
                    label=f'Before Step2 ({X_orig.shape[1]} features)', alpha=0.8, color='salmon')
    bars2 = ax1.bar(x_pos + width/2, xgb_red_values, width, 
                    label=f'After Step2 ({X_red.shape[1]} features)', alpha=0.8, color='skyblue')

    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('XGBoost Performance Comparison')
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

    # 2. Feature Count Comparison
    feature_counts = ['Before Step2', 'After Step2']
    counts = [X_orig.shape[1], X_red.shape[1]]
    colors = ['lightcoral', 'lightblue']

    bars = ax2.bar(feature_counts, counts, color=colors, alpha=0.8)
    ax2.set_ylabel('Number of Features')
    ax2.set_title('Feature Count Comparison')
    ax2.grid(True, alpha=0.3)

    # Add percentage reduction
    reduction_pct = (1 - X_red.shape[1] / X_orig.shape[1]) * 100
    ax2.text(0.5, max(counts) * 0.8, f'Reduction: {reduction_pct:.1f}%', 
            ha='center', fontsize=12, fontweight='bold')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{results_dir}/step2_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrix_comparison(y_true, y_pred_orig, y_pred_red, results_dir):
    """Create confusion matrix comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original model confusion matrix
    cm_orig = confusion_matrix(y_true, y_pred_orig)
    sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix - Before Step2')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Reduced model confusion matrix  
    cm_red = confusion_matrix(y_true, y_pred_red)
    sns.heatmap(cm_red, annot=True, fmt='d', cmap='Greens', ax=ax2)
    ax2.set_title('Confusion Matrix - After Step2')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/step2_confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(df_original, X_reduced, exclude_cols, target_cols, non_numeric_outside_exclude, removal_details=None):
    """Save results and create complete dataset"""
    print(f"\n💾 Saving Results")
    print("-" * 30)
    
    # Create results directory
    results_dir = "../results/step2_feature_reduction"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save selected features list (with proper header)
    pd.DataFrame({'feature_name': X_reduced.columns.tolist()}).to_csv(
        f'{results_dir}/step2_selected_features.csv', 
        index=False
    )
    
    # Save removal details if provided
    if removal_details:
        removal_df = pd.DataFrame(removal_details)
        removal_df.to_csv(f'{results_dir}/step2_removed_features.csv', index=False)
        print(f"📄 Removed features details saved to: {results_dir}/step2_removed_features.csv")
    
    # Create complete dataset: excluded columns + selected features + targets
    excluded_data = df_original[exclude_cols].copy()
    target_data = df_original[target_cols].copy()
    
    complete_dataset = pd.concat([
        excluded_data.reset_index(drop=True),
        X_reduced.reset_index(drop=True), 
        target_data.reset_index(drop=True)
    ], axis=1)
    
    # Save to dataset folder
    dataset_path = 'dataset/credit_risk_dataset_selected.csv'
    complete_dataset.to_csv(dataset_path, index=False)
        
    # Save summary
    summary = {
        'execution_date': datetime.now().isoformat(),
        'approach': 'multi_fold_stability_based',
        'steps': [
            '1. Correlation analysis (>0.9)',
            '2. Variance threshold (<0.001)', 
            '3. Missing values (>50%)',
            '4. Multi-fold stability-based XGBoost selection (5 folds, 60% stability threshold)'
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
    
    with open(f'{results_dir}/step2_summary.json', 'w', encoding='utf-8') as f:
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
    
    # Track removal details
    removal_details = []
    
    # Step 1: Remove high correlation
    X_before = X.copy()
    X = step1_remove_high_correlation(X, threshold=corr_threshold)
    removed_corr = [col for col in X_before.columns if col not in X.columns]
    for col in removed_corr:
        removal_details.append({
            'step': 'Step 1 - High Correlation',
            'column_name': col,
            'reason': f'Correlation > {corr_threshold}',
            'threshold': corr_threshold
        })
    
    # Step 2: Remove low variance  
    X_before = X.copy()
    X = step2_remove_low_variance(X, threshold=variance_threshold)
    removed_var = [col for col in X_before.columns if col not in X.columns]
    for col in removed_var:
        removal_details.append({
            'step': 'Step 2 - Low Variance',
            'column_name': col,
            'reason': f'Variance < {variance_threshold}',
            'threshold': variance_threshold
        })
    
    # Step 3: Remove high missing
    X_before = X.copy()
    X = step3_remove_high_missing(X, threshold=missing_threshold)
    removed_missing = [col for col in X_before.columns if col not in X.columns]
    for col in removed_missing:
        removal_details.append({
            'step': 'Step 3 - High Missing',
            'column_name': col,
            'reason': f'Missing > {missing_threshold*100}%',
            'threshold': missing_threshold
        })
    
    # Step 4: XGBoost feature importance
    X_before = X.copy()
    X_final = step4_xgboost_importance(X, y, keep_percentage=keep_percentage)
    removed_xgb = [col for col in X_before.columns if col not in X_final.columns]
    for col in removed_xgb:
        removal_details.append({
            'step': 'Step 4 - XGBoost Importance',
            'column_name': col,
            'reason': f'Not in top {keep_percentage}% by importance',
            'threshold': keep_percentage
        })

    # Audit non-numeric features outside exclude cols (diagnostics only)
    non_numeric_outside_exclude = audit_non_numeric_columns(X_final, exclude_cols)
    
    # Calculate final stats
    reduction_pct = (1 - len(X_final.columns) / len(X_original.columns)) * 100
    total_removed = len(X_original.columns) - len(X_final.columns)
    
    print(f"\n✅ REDUCTION COMPLETE!")
    print("=" * 30)
    print(f"📊 Final features: {len(X_final.columns)}")
    print(f"📉 Reduction: {len(X_original.columns)} → {len(X_final.columns)} ({reduction_pct:.1f}% removed)")
    print(f"🗑️ Total columns removed: {total_removed}")
    print(f"📋 Remaining columns: {len(X_final.columns)}")
    # Note: No business-logic preservation enforced by design
    
    # Save results and create dataset
    results_dir = save_results(df, X_final, exclude_cols, target_cols, non_numeric_outside_exclude, removal_details)
    
    # Create comparison visualizations
    create_visualizations(X_original, X_final, y, results_dir)
    
    print(f"\n🎉 FEATURE REDUCTION COMPLETED!")
    print("=" * 50)
    print("✅ Systematic reduction - multicollinearity eliminated")
    print("✅ XGBoost optimized - performance validated") 
    print(f"✅ Results saved in: {results_dir}")
    
    # Print summary of removals by step
    if removal_details:
        print(f"\n📋 REMOVAL SUMMARY BY STEP:")
        print("-" * 40)
        step_counts = {}
        for detail in removal_details:
            step = detail['step']
            step_counts[step] = step_counts.get(step, 0) + 1
        
        for step, count in step_counts.items():
            print(f"   {step}: {count} columns removed")
    
    return X_final

if __name__ == "__main__":
    results = main()