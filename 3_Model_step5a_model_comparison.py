"""
Step 5A: Temporal Model Comparison (XGBoost vs RandomForest vs MLP)
===================================================================

Purpose:
- Compare multiple algorithms under temporal validation to avoid optimistic bias
- Use dataset produced by Step 1 data pipeline (dataset/credit_risk_dataset.csv)
- Splits:
  1) Temporal Holdout (past → train, future → test)
  2) Rolling TimeSeriesSplit (expanding window)

Notes:
- Focus on per-target multi-class classification (0..3) for risk_year1..4
- XGBoost handles missing values natively; RF/MLP use SimpleImputer
- Results saved under result/step5a_model_comparison_temporal
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


def get_config() -> Dict:
    return {
        'data_path': 'dataset/credit_risk_dataset.csv',
        'date_column': '보험청약일자',
        'target_columns': ['risk_year1', 'risk_year2', 'risk_year3', 'risk_year4'],
        'exclude_columns': [
            '사업자등록번호', '대상자명', '대상자등록이력일시', '대상자기본주소',
            '청약번호', '보험청약일자', '청약상태코드', '수출자대상자번호', '특별출연협약코드', '업종코드1'
        ],
        'temporal_holdout_ratio': 0.2,
        'rolling_splits': 5,
        'results_dir': 'result/step5a_model_comparison_temporal',
        'random_state': 42,
    }


def prepare_data(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    df = pd.read_csv(cfg['data_path'])
    if cfg['date_column'] in df.columns:
        df[cfg['date_column']] = pd.to_datetime(df[cfg['date_column']], errors='coerce')
        df = df.sort_values(cfg['date_column']).reset_index(drop=True)
        print(f"✅ Data sorted by {cfg['date_column']} for temporal validation")
    else:
        print(f"⚠️ {cfg['date_column']} not found. Using current order for temporal validation")

    feature_cols = [c for c in df.columns if not c.startswith('risk_year')]
    feature_cols = [c for c in feature_cols if c not in set(cfg['exclude_columns'])]
    X = df[feature_cols]

    y_dict = {}
    for target in cfg['target_columns']:
        if target in df.columns:
            y_dict[target] = df[target].values
    return df, X, y_dict


def build_models(random_state: int = 42) -> Dict[str, object]:
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=4, enable_missing=True, random_state=random_state, verbosity=0, n_jobs=-1
    )
    rf_model = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('rf', RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=random_state))
    ])
    mlp_model = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(256, 64), activation='relu', alpha=1e-4,
                              batch_size=512, learning_rate_init=1e-3, max_iter=100,
                              early_stopping=True, n_iter_no_change=5, random_state=random_state))
    ])
    return {
        'xgboost': xgb_model,
        'random_forest': rf_model,
        'mlp': mlp_model,
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro')),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro')),
    }


def temporal_holdout_indices(n_samples: int, test_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    split_idx = int(n_samples * (1 - test_ratio))
    train_idx = np.arange(0, split_idx)
    test_idx = np.arange(split_idx, n_samples)
    return train_idx, test_idx


def run_temporal_holdout(X: pd.DataFrame, y: np.ndarray, model, mask_valid: np.ndarray, test_ratio: float) -> Dict:
    idx = np.where(mask_valid)[0]
    if len(idx) == 0:
        return {'error': 'no_valid_samples'}
    Xv = X.iloc[idx]
    yv = y[idx].astype(int)
    tr_idx, te_idx = temporal_holdout_indices(len(Xv), test_ratio)
    X_tr, X_te = Xv.iloc[tr_idx], Xv.iloc[te_idx]
    y_tr, y_te = yv[tr_idx], yv[te_idx]
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    metrics = evaluate_predictions(y_te, preds)
    return {'split': 'temporal_holdout', 'metrics': metrics}


def run_rolling_cv(X: pd.DataFrame, y: np.ndarray, model, mask_valid: np.ndarray, n_splits: int) -> Dict:
    idx = np.where(mask_valid)[0]
    if len(idx) == 0:
        return {'error': 'no_valid_samples'}
    Xv = X.iloc[idx]
    yv = y[idx].astype(int)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: List[Dict[str, float]] = []
    for fold, (tr, te) in enumerate(tscv.split(Xv), 1):
        model_fold = clone(model)
        model_fold.fit(Xv.iloc[tr], yv[tr])
        preds = model_fold.predict(Xv.iloc[te])
        fold_metrics.append(evaluate_predictions(yv[te], preds))
    avg_metrics = {
        'accuracy': float(np.mean([m['accuracy'] for m in fold_metrics])),
        'f1_macro': float(np.mean([m['f1_macro'] for m in fold_metrics])),
        'precision_macro': float(np.mean([m['precision_macro'] for m in fold_metrics])),
        'recall_macro': float(np.mean([m['recall_macro'] for m in fold_metrics])),
    }
    return {'split': 'rolling_tscv', 'folds': fold_metrics, 'metrics': avg_metrics}


def ensure_dirs(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'metrics'), exist_ok=True)


def save_results(base_dir: str, algorithm: str, target: str, results: Dict):
    path = os.path.join(base_dir, 'metrics', f'step5a_{algorithm}_{target}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    # Also write a compact CSV row for the top-level metrics of each split
    rows = []
    for split_name in results.keys():
        entry = results[split_name]
        metrics = entry.get('metrics', {})
        if metrics:
            rows.append({'algorithm': algorithm, 'target': target, 'split': split_name, **metrics})
    if rows:
        csv_df = pd.DataFrame(rows)
        csv_path = os.path.join(base_dir, 'metrics', f'step5a_{algorithm}_{target}.csv')
        csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')


def run_comparison(config: Dict):
    ensure_dirs(config['results_dir'])
    df, X, y_dict = prepare_data(config)
    models = build_models(config['random_state'])

    summary_rows: List[Dict] = []

    for algo_name, model in models.items():
        print(f"\n==============================\n🚀 Evaluating algorithm: {algo_name}")
        for target, y in y_dict.items():
            print(f"   🎯 Target: {target}")
            mask_valid = ~pd.isna(y)
            results_for_target = {}
            try:
                holdout_res = run_temporal_holdout(X, y, clone(model), mask_valid, config['temporal_holdout_ratio'])
                results_for_target['temporal_holdout'] = holdout_res
            except Exception as e:
                results_for_target['temporal_holdout'] = {'error': str(e)}
            try:
                rolling_res = run_rolling_cv(X, y, clone(model), mask_valid, config['rolling_splits'])
                results_for_target['rolling_tscv'] = rolling_res
            except Exception as e:
                results_for_target['rolling_tscv'] = {'error': str(e)}

            save_results(config['results_dir'], algo_name, target, results_for_target)

            # Aggregate for summary
            for split_key, entry in results_for_target.items():
                metrics = entry.get('metrics')
                if metrics:
                    summary_rows.append({
                        'algorithm': algo_name,
                        'target': target,
                        'split': split_key,
                        **metrics
                    })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(config['results_dir'], 'step5a_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 Summary saved: {summary_path}")
    print("\n✅ Step 5A temporal model comparison completed")


if __name__ == "__main__":
    config = get_config()
    run_comparison(config)


