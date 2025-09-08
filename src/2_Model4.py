"""
Step 4: Fair Model Family Comparison (Temporal - Rolling Window)
==================================================================

Goal:
- FAIR comparison of XGBoost, RandomForest, MLP, and Logistic Regression
- ALL models use IDENTICAL preprocessing (no advantage to XGBoost native NaN handling)
- Use Step 2 dataset (feature-refined) and a unified Rolling Window CV
- Train individual models per target (risk_year1..risk_year4), same conditions
- Shared preprocessing pipeline ensures scientific validity of comparison
- Save metrics in a consistent structure for downstream analysis

Key Improvement:
- XGBoost now uses same preprocessed data as other models (fair comparison)
- Previous versions gave XGBoost unfair advantage with native missing value handling

Outputs:
- Per model/target metrics JSON: ../results/step4_model_comparison_temporal/metrics/step4_<model>_<target>.json
- Summary JSON: ../results/step4_model_comparison_temporal/step4_summary.json
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import platform
import warnings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.base import clone

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# Visualization font setup (keep Korean font logic consistent)
# ------------------------------------------------------------
def setup_korean_font():
    system = platform.system()
    if system == "Windows":
        korean_fonts = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'Gulim', 'Dotum']
    elif system == "Darwin":
        korean_fonts = ['AppleGothic', 'NanumGothic']
    else:
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


setup_korean_font()
plt.style.use('default')
sns.set_palette("husl")


# ------------------------------------------------------------
# Data loading (Step 2 dataset) and preparation
# ------------------------------------------------------------
def load_and_prepare_selected_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    print("\n🚀 MODEL FAMILY COMPARISON (Rolling Window)\n" + "=" * 60)
    print("📂 Loading Step 2 Optimized Dataset")
    print("-" * 40)

    df = pd.read_csv('../data/processed/credit_risk_dataset_selected.csv')
    print(f"✅ Step 2 dataset loaded: {df.shape}")

    # Sort by 보험청약일자 to preserve temporal order
    if '보험청약일자' in df.columns:
        df = df.sort_values('보험청약일자').reset_index(drop=True)
        print("✅ Data sorted by 보험청약일자 for temporal validation")
    else:
        print("⚠️ 보험청약일자 not found - using index order for temporal validation")

    exclude_cols = [
        '사업자등록번호', '대상자명', '청약번호', '보험청약일자', '수출자대상자번호', '업종코드1'
    ]

    target_cols = [c for c in df.columns if c.startswith('risk_year')]
    feature_cols = [c for c in df.columns if c not in target_cols + exclude_cols]

    print(f"📋 Excluded columns: {len(exclude_cols)}")
    print(f"🎯 Target columns: {len(target_cols)} → {target_cols}")
    print(f"📊 Features (pre-preprocessing): {len(feature_cols)}")

    X = df[feature_cols]
    y = df[target_cols]
    return df, X, y, exclude_cols, target_cols


# ------------------------------------------------------------
# Preprocessing builder (shared for all models)
# ------------------------------------------------------------
def _make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder compatible across sklearn versions.

    sklearn >= 1.2 removes 'sparse' and uses 'sparse_output'.
    This helper tries new API first, then falls back.
    """
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


def build_preprocessor(
    X: pd.DataFrame,
    max_unique_categories: int = 40,
) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    """Build a ColumnTransformer with KNN imputation (numeric) and OHE (categorical).

    Returns: (preprocessor, numeric_cols_used, categorical_cols_used, dropped_high_cardinality_cols)
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_all = X.select_dtypes(exclude=[np.number]).columns.tolist()

    dropped_high_card_cols: List[str] = []
    categorical_cols: List[str] = []
    for c in cat_all:
        nunique = X[c].nunique(dropna=True)
        if nunique <= max_unique_categories:
            categorical_cols.append(c)
        else:
            dropped_high_card_cols.append(c)

    # Pipelines
    numeric_pipeline = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5, weights='distance')),
        ('scaler', StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', _make_one_hot_encoder()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('cat', categorical_pipeline, categorical_cols),
        ],
        remainder='drop',
        n_jobs=None,
    )

    return preprocessor, numeric_cols, categorical_cols, dropped_high_card_cols


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
def get_models(random_state: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {
        # All models use identical preprocessing - FAIR COMPARISON
        'xgboost': xgb.XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            verbosity=0,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='mlogloss'
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        'mlp': MLPClassifier(
            random_state=random_state,
        ),
        'logistic_regression': LogisticRegression(
            multi_class='auto',
            random_state=random_state,
        ),
    }
    return models


# ------------------------------------------------------------
# Rolling Window CV split utility
# ------------------------------------------------------------
def make_rolling_window_splits(n_samples: int, window_size: float, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    window_samples = int(n_samples * window_size)
    step_size = max(1, (n_samples - window_samples) // n_splits)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        start_idx = fold * step_size
        end_idx = start_idx + window_samples
        if end_idx >= n_samples:
            break
        test_end_idx = min(end_idx + step_size, n_samples)
        test_idx = np.arange(end_idx, test_end_idx)
        if len(test_idx) == 0:
            break
        train_idx = np.arange(start_idx, end_idx)
        splits.append((train_idx, test_idx))
    return splits


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
def precompute_preprocessed_folds(
    X: pd.DataFrame,
    y: pd.Series,
    window_size: float,
    n_splits: int,
    max_unique_categories: int,
) -> Tuple[List[Dict], Dict, Dict]:
    """Fit preprocessing once per fold and return transformed arrays for reuse across models."""
    preprocessor, num_cols, cat_cols, dropped_high_card = build_preprocessor(
        X, max_unique_categories=max_unique_categories
    )
    n_samples = len(X)
    splits = make_rolling_window_splits(n_samples, window_size, n_splits)

    fold_data: List[Dict] = []
    used_folds = 0

    print(f"   • Total samples: {n_samples:,}")
    print(f"   • Window size: {int(n_samples * window_size):,} samples")
    print(f"   • Step size: {max(1, (n_samples - int(n_samples * window_size)) // n_splits):,} samples")
    print(f"   • Features: numeric={len(num_cols)}, categorical={len(cat_cols)} (dropped_high_card={len(dropped_high_card)})")

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if pd.Series(y_train).nunique() < 2:
            print(f"   ⚠️ Fold {fold_idx}: skipped (train has <2 classes)")
            continue

        preprocessor.fit(X_train, y_train.astype(int))
        X_train_t = preprocessor.transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        # Downcast to float32 to speed up model training and reduce memory
        if isinstance(X_train_t, np.ndarray):
            X_train_t = X_train_t.astype(np.float32, copy=False)
            X_test_t = X_test_t.astype(np.float32, copy=False)

        fold_data.append({
            'X_train_t': X_train_t,
            'X_test_t': X_test_t,
            'y_train': y_train.astype(int).values,
            'y_test': y_test.values,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'fold_index': fold_idx,
        })
        used_folds += 1

    features_info = {
        'numeric': len(num_cols),
        'categorical': len(cat_cols),
        'dropped_high_cardinality': dropped_high_card,
    }
    window_info = {
        'window_size_fraction': window_size,
        'total_samples': n_samples,
        'n_splits_attempted': n_splits,
        'n_splits_used': used_folds,
    }
    return fold_data, features_info, window_info


def evaluate_with_preprocessed_folds(
    fold_data: List[Dict],
    model_name: str,
    estimator,
    features_info: Dict,
    window_info: Dict,
) -> Dict:
    fold_metrics: List[Dict] = []
    train_sizes: List[int] = []
    used_folds = 0

    for fd in fold_data:
        model = clone(estimator)
        model.fit(fd['X_train_t'], fd['y_train'])
        y_pred = model.predict(fd['X_test_t'])

        metrics = {
            'f1_macro': float(f1_score(fd['y_test'], y_pred, average='macro', zero_division=0)),
            'accuracy': float(accuracy_score(fd['y_test'], y_pred)),
            'precision_macro': float(precision_score(fd['y_test'], y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(fd['y_test'], y_pred, average='macro', zero_division=0)),
        }
        fold_metrics.append(metrics)
        train_sizes.append(fd['train_size'])
        used_folds += 1
        print(f"   Fold {fd['fold_index']}: Train={fd['train_size']:,}, Test={fd['test_size']:,}, F1={metrics['f1_macro']:.4f}")

    if used_folds == 0:
        print("   ⚠️ No valid folds generated")
        return {
            'model': model_name,
            'avg_metrics': None,
            'fold_metrics': [],
            'train_sizes': [],
            'n_splits': 0,
            'features': features_info,
            'window': window_info,
        }

    avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()}
    print(f"   🎯 Average F1-Score (Macro): {avg_metrics['f1_macro']:.4f}")
    print(f"   📈 Average Accuracy: {avg_metrics['accuracy']:.4f}")

    return {
        'model': model_name,
        'avg_metrics': avg_metrics,
        'fold_metrics': fold_metrics,
        'train_sizes': train_sizes,
        'n_splits': used_folds,
        'features': features_info,
        'window': window_info,
    }


# ------------------------------------------------------------
# Orchestration per target and saving
# ------------------------------------------------------------
def run_models_for_target(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    results_dir: str,
    window_size: float = 0.6,
    n_splits: int = 5,
    max_unique_categories: int = 40,
) -> Dict[str, Dict]:
    print(f"\n🎯 MODEL COMPARISON FOR {target_name}\n" + "=" * 70)
    models = get_models(random_state=42)

    # Filter valid samples (non-NaN target)
    mask = ~pd.isna(y)
    X_t = X.loc[mask]
    y_t = y.loc[mask].astype(int)
    print(f"📊 Available data for {target_name}: {len(X_t):,} samples")

    # Precompute preprocessing once per fold (reused by all models)
    fold_data, features_info, window_info = precompute_preprocessed_folds(
        X_t, y_t, window_size=window_size, n_splits=n_splits, max_unique_categories=max_unique_categories
    )

    # Run each model using preprocessed folds
    model_results: Dict[str, Dict] = {}
    for mname, model in models.items():
        print(f"\n🧪 Evaluating {mname} (Rolling Window CV: window={window_size}, n_splits={n_splits})")
        res = evaluate_with_preprocessed_folds(
            fold_data=fold_data,
            model_name=mname,
            estimator=model,
            features_info=features_info,
            window_info=window_info,
        )
        model_results[mname] = res

        # Save per-model-per-target metrics
        metrics_dir = os.path.join(results_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        out_path = os.path.join(metrics_dir, f"step4_{mname}_{target_name}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved: {out_path}")

    # Print quick ranking by F1
    ranking = sorted(
        ((m, r['avg_metrics']['f1_macro']) for m, r in model_results.items() if r['avg_metrics'] is not None),
        key=lambda x: x[1], reverse=True
    )
    if ranking:
        print("\n🏆 Ranking by F1-macro:")
        for i, (m, score) in enumerate(ranking, 1):
            print(f"   {i}. {m}: {score:.4f}")

    return model_results


def summarize_overall(results_by_target: Dict[str, Dict[str, Dict]], results_dir: str):
    # Aggregate average F1 across targets per model
    model_to_scores: Dict[str, List[float]] = {}
    for target, mres in results_by_target.items():
        for mname, res in mres.items():
            if res['avg_metrics'] is None:
                continue
            model_to_scores.setdefault(mname, []).append(res['avg_metrics']['f1_macro'])

    model_summary = {
        m: {
            'mean_f1_macro': float(np.mean(scores)) if scores else None,
            'std_f1_macro': float(np.std(scores)) if scores else None,
            'count_targets': len(scores),
        }
        for m, scores in model_to_scores.items()
    }

    best_model = None
    if model_summary:
        best_model = sorted(
            ((m, s['mean_f1_macro']) for m, s in model_summary.items() if s['mean_f1_macro'] is not None),
            key=lambda x: x[1], reverse=True
        )[0][0]

    summary = {
        'execution_note': 'Step 4 model comparison with rolling window',
        'targets_compared': list(results_by_target.keys()),
        'model_summary': model_summary,
        'best_model_by_avg_f1_macro': best_model,
    }

    out_path = os.path.join(results_dir, 'step4_summary.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Summary saved: {out_path}")


# ------------------------------------------------------------
# Visualization and XGBoost integration from Step 4
# ------------------------------------------------------------
def load_xgb_from_step4(step4_results_path: str) -> Dict[str, Dict]:
    """Load XGBoost rolling-window results from Step 4 comprehensive results file.

    Returns: { target_name: { 'avg_metrics': {...}, 'model': 'xgboost_from_step4' } }
    """
    if not os.path.exists(step4_results_path):
        print(f"⚠️ Step 4 results not found at: {step4_results_path}")
        return {}
    with open(step4_results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out: Dict[str, Dict] = {}
    for target_name, sections in data.items():
        if 'rolling_window' in sections:
            metrics = sections['rolling_window'].get('metrics', {})
            out[target_name] = {
                'model': 'xgboost',
                'avg_metrics': {
                    'f1_macro': float(metrics.get('f1_macro', np.nan)),
                    'accuracy': float(metrics.get('accuracy', np.nan)),
                    'precision_macro': float(metrics.get('precision', np.nan)),
                    'recall_macro': float(metrics.get('recall', np.nan)),
                }
            }
    return out


def create_model_comparison_visualizations(
    results_by_target: Dict[str, Dict[str, Dict]],
    xgb_results: Dict[str, Dict],  # Kept for compatibility but not used
    results_dir: str,
):
    print("\n📊 Creating Model Comparison Visualizations (Fair Preprocessing for All Models)")
    vis_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    targets = list(results_by_target.keys())
    model_order = ['xgboost', 'random_forest', 'mlp', 'logistic_regression']
    colors = ['gold', 'lightgreen', 'plum', 'lightblue']

    # 1) F1-macro comparison across targets
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Step 4: Fair Model Comparison (F1-macro) - Identical Preprocessing', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for idx, target in enumerate(targets):
        ax = axes[idx]
        names, scores, bar_colors = [], [], []

        # All models from current run (including XGBoost with fair preprocessing)
        for mname in model_order:
            if mname in results_by_target[target] and results_by_target[target][mname]['avg_metrics'] is not None:
                names.append(mname)
                scores.append(results_by_target[target][mname]['avg_metrics']['f1_macro'])
                color_idx = model_order.index(mname)
                bar_colors.append(colors[color_idx])

        bars = ax.bar(names, scores, color=bar_colors, alpha=0.85)
        ax.set_ylabel('F1-Score (Macro)')
        ax.set_title(f'{target} - F1 by Model (Fair Comparison)')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=20)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

    plt.tight_layout()
    out_f1 = os.path.join(vis_dir, 'step4_fair_model_comparison_f1.png')
    plt.savefig(out_f1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out_f1}")

    # 2) Accuracy comparison across targets
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Step 4: Fair Model Comparison (Accuracy) - Identical Preprocessing', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for idx, target in enumerate(targets):
        ax = axes[idx]
        names, scores, bar_colors = [], [], []

        # All models from current run (including XGBoost with fair preprocessing)
        for mname in model_order:
            if mname in results_by_target[target] and results_by_target[target][mname]['avg_metrics'] is not None:
                names.append(mname)
                scores.append(results_by_target[target][mname]['avg_metrics']['accuracy'])
                color_idx = model_order.index(mname)
                bar_colors.append(colors[color_idx])

        bars = ax.bar(names, scores, color=bar_colors, alpha=0.85)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{target} - Accuracy by Model (Fair Comparison)')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=20)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

    plt.tight_layout()
    out_acc = os.path.join(vis_dir, 'step4_fair_model_comparison_accuracy.png')
    plt.savefig(out_acc, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out_acc}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    df, X, y, exclude_cols, target_cols = load_and_prepare_selected_data()

    results_dir = '../results/step4_model_comparison_temporal'
    os.makedirs(results_dir, exist_ok=True)

    print("\n🔬 TEMPORAL COMPARISON ENGINE (Rolling Window)")
    print("=" * 60)
    print(f"🎯 Targets: {target_cols}")
    print(f"📈 Features (columns before preprocessing): {X.shape[1]}")

    # Compare models per target
    results_by_target: Dict[str, Dict[str, Dict]] = {}
    for target in target_cols:
        model_results = run_models_for_target(
            X=X,
            y=y[target],
            target_name=target,
            results_dir=results_dir,
            window_size=0.6,
            n_splits=5,
            max_unique_categories=40,
        )
        results_by_target[target] = model_results

    # Save overall summary (now includes XGBoost)
    summarize_overall(results_by_target, results_dir)

    # Create visualizations comparing all models with fair preprocessing
    create_model_comparison_visualizations(results_by_target, {}, results_dir)

    print("\n🎉 MODEL FAMILY COMPARISON COMPLETED!")
    print("=" * 60)
    print(f"✅ Results saved in: {results_dir}")
    return results_by_target


if __name__ == "__main__":
    main()


