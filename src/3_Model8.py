"""
Step 8: Final Model Training with Strict Data Separation
=======================================================

Purpose
- Train final model(s) using TRAIN + VALIDATION data only
- Export predictions for all data (train+validation+oot)
- Respect strict data separation rules from 1_Split.py

Key Changes for Strict Validation:
- Model training: TRAIN + VALIDATION data only
- Feature selection: Already done in Step 2 with TRAIN data only
- Hyperparameter tuning: Should be done with TRAIN + VALIDATION
- Predictions: Export for all data but maintain split awareness

Outputs
- ../results/predictions/yearly_multiclass_proba.csv (all data with split labels)
- ../results/step8_post/quick_report.md
- ../results/step8_post/plots/feature_importance_*.png
- ../results/step8_post/models/* (JSON boosters)
- ../results/step8_post/manifest.json

Data Usage Rules:
- Training: TRAIN + VALIDATION (never OOT)
- Evaluation plots: DEV subset only (never OOT for model development)
- OOT data: Only for final unbiased evaluation in Step 5
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.utils.class_weight import compute_sample_weight

# Visualization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------


@dataclass
class Step8Config:
    dataset_path: str = "../data/processed/credit_risk_dataset_selected.csv"
    predictions_path: str = "../results/predictions/yearly_multiclass_proba.csv"
    output_dir: str = "../results/step8_post"
    best_params_dir: str = "../results/step7_optuna"
    arch: str = "unified"  # individual|unified
    shap_sample_size: int = 1000  # 0 disables; else sample up to N rows per target for contribs
    date_column: str = "보험청약일자"
    key_column: str = "청약번호"
    random_state: int = 42
    # Optional: run grading after predictions
    run_grade: bool = False
    grade_script: str = "4_Model9.py"
    grade_output_dir: str = "../results/grading"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_selected_data(dataset_path: str, date_column: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    if date_column in df.columns:
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception:
            pass
    return df

def load_split_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the strictly split datasets from 1_Split.py
    
    Returns:
        Tuple of (train_data, validation_data, oot_data)
    """
    split_dir = "../data/splits"
    
    train_data = pd.read_csv(os.path.join(split_dir, "train_data.csv"))
    validation_data = pd.read_csv(os.path.join(split_dir, "validation_data.csv"))
    oot_data = pd.read_csv(os.path.join(split_dir, "oot_data.csv"))
    
    logger.info(f"Loaded split datasets:")
    logger.info(f"  - Train: {len(train_data):,} rows")
    logger.info(f"  - Validation: {len(validation_data):,} rows") 
    logger.info(f"  - OOT: {len(oot_data):,} rows")
    logger.info(f"  - Total: {len(train_data) + len(validation_data) + len(oot_data):,} rows")
    
    return train_data, validation_data, oot_data

def get_training_data(train_data: pd.DataFrame, validation_data: pd.DataFrame) -> pd.DataFrame:
    """
    Combine train and validation data for final model training
    This follows our strict validation rules: final model uses train+validation
    """
    training_data = pd.concat([train_data, validation_data], ignore_index=True)
    training_data = training_data.sort_values('보험청약일자').reset_index(drop=True)
    
    logger.info(f"Combined training data: {len(training_data):,} rows")
    logger.info(f"  - Period: {training_data['보험청약일자'].min()} to {training_data['보험청약일자'].max()}")
    
    return training_data


def get_feature_and_target_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    exclude_cols = [
        "사업자등록번호",
        "대상자명",
        "청약번호",
        "보험청약일자",
        "수출자대상자번호",
        "업종코드1",
        "data_split",  # Exclude split identifier from features
    ]
    target_cols = [c for c in df.columns if c.startswith("risk_year")]
    feature_cols = [c for c in df.columns if c not in exclude_cols + target_cols]
    return feature_cols, target_cols, exclude_cols


def detect_best_params(cfg: Step8Config) -> Tuple[str, Dict[str, dict], Dict[str, any]]:
    """
    Returns (approach, best_params_by_target_or_unified, meta)
    - approach: 'individual' or 'unified'
    - best_params_by_target_or_unified:
        individual -> {"risk_year1": {..}, ..., "risk_year4": {..}}
        unified    -> {"unified": {..}}
    - meta: additional info from the best-params file
    """
    candidates_ordered: List[Tuple[str, str]] = []
    if cfg.arch == "individual":
        candidates_ordered = [
            ("individual", os.path.join(cfg.best_params_dir, "step7_best_params_individual_cpu.json")),
            ("individual", os.path.join(cfg.best_params_dir, "step7_best_params_individual_gpu.json")),
        ]
    elif cfg.arch == "unified":
        candidates_ordered = [
            ("unified", os.path.join(cfg.best_params_dir, "step7_best_params_unified_cpu.json")),
            ("unified", os.path.join(cfg.best_params_dir, "step7_best_params_unified_gpu.json")),
        ]
    else:
        candidates_ordered = [
            ("individual", os.path.join(cfg.best_params_dir, "step7_best_params_individual_cpu.json")),
            ("unified", os.path.join(cfg.best_params_dir, "step7_best_params_unified_cpu.json")),
            ("individual", os.path.join(cfg.best_params_dir, "step7_best_params_individual_gpu.json")),
            ("unified", os.path.join(cfg.best_params_dir, "step7_best_params_unified_gpu.json")),
        ]

    for approach, path in candidates_ordered:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = {k: v for k, v in data.items() if k not in {"best_params", "best_params_by_target"}}
            if approach == "individual" and "best_params_by_target" in data:
                return "individual", data["best_params_by_target"], meta
            if approach == "unified" and "best_params" in data:
                return "unified", {"unified": data["best_params"]}, meta
            # Fallback if schema differs
            if "best_params_by_target" in data:
                return "individual", data["best_params_by_target"], meta
            if "best_params" in data:
                return "unified", {"unified": data["best_params"]}, meta

    raise FileNotFoundError(
        "Could not find Step 7 best-params file. Expected one of: "
        "step7_best_params_individual(_cpu.json|_gpu.json) or step7_best_params_unified(_cpu.json|_gpu.json) in ../results/step7_optuna"
    )






def expand_proba_to_4_classes(model_classes: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """
    Ensures probability matrix has 4 columns for classes {0,1,2,3}.
    """
    num_rows = proba.shape[0]
    full = np.zeros((num_rows, 4), dtype=float)
    for idx, cls in enumerate(model_classes):
        if 0 <= int(cls) <= 3:
            full[:, int(cls)] = proba[:, idx]
    # Normalize safety (should already sum to 1 across present classes)
    row_sum = full.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        full = np.where(row_sum > 0, full / row_sum, full)
    return full


def xgb_classifier_from_params(params: Dict[str, any], random_state: int) -> xgb.XGBClassifier:
    base = dict(
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss",
    )
    # Remove None-valued keys
    base = {k: v for k, v in base.items() if v is not None}
    model = xgb.XGBClassifier(**base, **params)
    return model


def save_model_booster(model: xgb.XGBClassifier, path: str) -> None:
    try:
        booster = model.get_booster()
        booster.save_model(path)
    except Exception:
        # Fallback to sklearn pickle if booster not available
        import joblib

        joblib.dump(model, path + ".pkl")


def feature_importance_gain(model: xgb.XGBClassifier, feature_names: List[str]) -> pd.DataFrame:
    try:
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
    except Exception:
        score = {}
    # Normalize keys to feature names when possible
    # XGBoost returns feature keys like 'f0', 'f1' if not trained with named columns.
    # Here we trained with pandas DataFrame, so names should be preserved.
    rows = [(k, float(v)) for k, v in score.items() if k in feature_names]
    df = pd.DataFrame(rows, columns=["feature", "gain"]).sort_values("gain", ascending=False)
    return df


def try_configure_korean_font() -> None:
    try:
        # Prefer common Korean fonts on Windows and Linux
        for f in ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "AppleGothic"]:
            plt.rcParams["font.family"] = f
            break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def compute_individual_contributions(model: xgb.XGBClassifier, X_features: pd.DataFrame, feature_names: List[str], key_values: pd.Series, batch_size: int = 5000) -> pd.DataFrame:
    """
    Compute individual feature contributions for each prediction using XGBoost's pred_contribs.
    Fixed version with proper feature alignment and batch processing.
    
    Args:
        model: Trained XGBoost classifier
        X_features: Feature data WITHOUT key column (only features used in training)
        feature_names: List of feature names used in training
        key_values: Series of key values (청약번호) corresponding to X_features rows
        batch_size: Process in batches to manage memory
    
    Returns:
        DataFrame with key_column and top_var_1, top_var_2, top_var_3
    """
    try:
        # Ensure feature alignment
        if len(feature_names) != X_features.shape[1]:
            logger.warning(f"Feature count mismatch: expected {len(feature_names)}, got {X_features.shape[1]}")
            # Try to align features
            available_features = [f for f in feature_names if f in X_features.columns]
            X_features = X_features[available_features]
            feature_names = available_features
            logger.info(f"Aligned to {len(feature_names)} common features")
        
        results = []
        n_samples = len(X_features)
        
        # Process in batches to manage memory
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_X = X_features.iloc[start_idx:end_idx]
            batch_keys = key_values.iloc[start_idx:end_idx]
            
            # Use XGBoost's built-in contribution calculation
            dmatrix = xgb.DMatrix(batch_X, feature_names=feature_names)
            contributions = model.get_booster().predict(dmatrix, pred_contribs=True)
            
            # contributions shape: (batch_size, n_features + 1) - last column is bias
            feature_contribs = contributions[:, :-1]  # Remove bias column
            
            for i, row_contribs in enumerate(feature_contribs):
                # Get absolute contributions and find top 3
                abs_contribs = np.abs(row_contribs)
                top_3_idx = np.argsort(abs_contribs)[-3:][::-1]  # Top 3 in descending order
                
                # Create result row
                result_row = {'unique_id': batch_keys.iloc[i]}
                
                for j, idx in enumerate(top_3_idx, 1):
                    if idx < len(feature_names):
                        result_row[f'top_var_{j}'] = feature_names[idx]
                        result_row[f'top_var_{j}_contrib'] = float(abs_contribs[idx])
                    else:
                        result_row[f'top_var_{j}'] = ''
                        result_row[f'top_var_{j}_contrib'] = 0.0
                
                results.append(result_row)
            
            if start_idx % (batch_size * 10) == 0:  # Progress indicator
                logger.info(f"Processed {min(end_idx, n_samples):,} / {n_samples:,} contributions")
        
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.warning(f"Individual contributions calculation failed: {e}")
        # Return empty DataFrame with expected columns
        empty_data = {'unique_id': key_values}
        for j in range(1, 4):
            empty_data[f'top_var_{j}'] = ''
            empty_data[f'top_var_{j}_contrib'] = 0.0
        return pd.DataFrame(empty_data)




def plot_feature_importance(fi_df: pd.DataFrame, path: str, title: str, top_n: int = 20) -> None:
    if fi_df is None or fi_df.empty:
        return
    df = fi_df.head(top_n)
    plt.figure(figsize=(7.5, 5.0))
    sns.barplot(data=df, x="gain", y="feature", color="#4C78A8")
    plt.xlabel("Gain (importance)")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_shap_importance(shap_df: pd.DataFrame, path: str, title: str, top_n: int = 20) -> None:
    """
    Plot SHAP-based feature importance with comparison to gain-based importance if available.
    """
    if shap_df is None or shap_df.empty:
        return
    
    df = shap_df.head(top_n)
    plt.figure(figsize=(8.0, 5.5))
    
    # Create horizontal bar plot
    bars = plt.barh(range(len(df)), df["shap_importance"], color="#E74C3C", alpha=0.7)
    
    # Customize plot
    plt.yticks(range(len(df)), df["feature"])
    plt.xlabel("SHAP Importance (Mean |SHAP value|)")
    plt.ylabel("Feature")
    plt.title(title)
    plt.gca().invert_yaxis()  # Top features at the top
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(df.iterrows()):
        plt.text(row["shap_importance"], i, f'{row["shap_importance"]:.3f}', 
                va='center', ha='left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def create_business_variable_mapping() -> Dict[str, str]:
    """
    Create mapping from technical variable names to business-friendly Korean names.
    """
    return {
        # Add mappings as needed - this is a template
        "업력": "기업 업력 (년)",
        "자본금": "자본금 규모",
        "매출액": "연간 매출액",
        "업종코드1": "주요 업종",
        "보험료": "보험료 수준",
        "보험가입금액": "보험 가입금액",
        "조기경보선정기준변화": "조기경보 기준 변화",
        "다이렉트보증보험유무": "다이렉트 보증보험 가입여부",
        # Add more mappings based on your actual features
    }

def generate_readme(output_dir: str, feature_cols: List[str], model_manifest: Dict[str, any]) -> None:
    """
    Generate comprehensive README.md for model usage and maintenance.
    """
    approach = model_manifest.get('approach', 'Unknown')
    execution_date = model_manifest.get('execution_date', 'Unknown')
    
    readme_content = f"""# KSURE 수출신용보험 리스크 예측 모델

## 🎯 모델 개요
- **실행일시**: {execution_date}
- **모델 아키텍처**: {approach.upper()} ({"통합 모델" if approach == "unified" else "개별 모델"})
- **예측 대상**: 1-4년 리스크 등급 (0: 안전 → 3: 위험)
- **총 특성 수**: {len(feature_cols)}개

## 📁 주요 파일 구조
```
../results/step8_post/
├── unified_model.json          # 학습된 XGBoost 모델
├── plots/
│   └── feature_importance_overall.png  # 특성 중요도 시각화
├── variable_mapping.json       # 변수명 한글 매핑
└── README.md                   # 본 파일

../results/predictions/
└── yearly_multiclass_proba.csv # 예측 결과 + 기여도 분석
```

## 🚀 모델 사용법

### 기본 예측
```python
import xgboost as xgb
import pandas as pd
import numpy as np

# 1. 모델 로드
model = xgb.Booster()
model.load_model('../results/step8_post/unified_model.json')

# 2. 새로운 데이터 준비 (task_id 추가 필요)
new_data = pd.read_csv('new_companies.csv')
new_data['task_id'] = 0  # 1년 예측의 경우

# 3. 예측 실행
dmatrix = xgb.DMatrix(new_data)
predictions = model.predict(dmatrix)  # 4개 클래스 확률

# 4. 최종 등급 결정
predicted_classes = np.argmax(predictions, axis=1)
```

### 연도별 예측
```python
# 1-4년 모든 연도 예측
all_predictions = {{}}
for year in range(1, 5):
    data_with_task = new_data.copy()
    data_with_task['task_id'] = year - 1
    dmatrix = xgb.DMatrix(data_with_task)
    all_predictions[f'year_{{year}}'] = model.predict(dmatrix)
```

## 🔧 재학습 가이드

### 정기 재학습 (권장: 분기별)
1. **데이터 업데이트**
   ```bash
   python 1_Dataset.py --update_date 2024-01-01
   ```

2. **하이퍼파라미터 최적화**
   ```bash
   python 2_Model7.py --arch unified --trials 100
   ```

3. **최종 모델 학습**
   ```bash
   python 3_Model8.py --arch unified
   ```

### 성능 모니터링
- **월별 검증**: `5_Model10.py` 실행으로 성능 추이 확인
- **임계값**: AR < 0.3 또는 KS < 0.25 시 재학습 필요

## 📊 주요 특성 Top 10
{chr(10).join([f"{i+1:2d}. {feat}" for i, feat in enumerate(feature_cols[:10])])}
{'    ...' if len(feature_cols) > 10 else ''}

## ⚠️ 주의사항
- **task_id**: 반드시 0-3 범위 (0=1년, 1=2년, 2=3년, 3=4년)
- **특성 순서**: 학습 시와 동일한 순서 유지 필요
- **결측치**: 예측 전 적절한 처리 필요

## 📞 문의사항
- **기술 문의**: 데이터분석팀
- **비즈니스 문의**: 리스크관리팀
- **긴급 장애**: [내부 연락처]

---
*Generated by KSURE Risk Model Pipeline v1.0*
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    logger.info(f"Comprehensive README generated: {readme_path}")


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Step 8: Streamlined Model Training and Prediction")
    parser.add_argument("--dataset_path", type=str, default="../data/processed/credit_risk_dataset_selected.csv")
    parser.add_argument("--predictions_path", type=str, default="../results/predictions/yearly_multiclass_proba.csv")
    parser.add_argument("--output_dir", type=str, default="../results/step8_post")
    parser.add_argument("--best_params_dir", type=str, default="../results/step7_optuna")
    parser.add_argument("--arch", type=str, default="unified", choices=["individual", "unified"])
    parser.add_argument("--enable_contributions", action="store_true", default=True, help="Enable individual feature contributions")
    parser.add_argument("--date_column", type=str, default="보험청약일자")
    parser.add_argument("--key_column", type=str, default="청약번호")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    cfg = Step8Config(
        dataset_path=args.dataset_path,
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
        best_params_dir=args.best_params_dir,
        arch=args.arch,
        shap_sample_size=0,  # Disabled
        date_column=args.date_column,
        key_column=args.key_column,
        random_state=int(args.random_state),
        run_grade=False,  # Removed auto-grade execution
        grade_script="",
        grade_output_dir="",
    )

    ensure_dir(os.path.dirname(cfg.predictions_path))
    ensure_dir(cfg.output_dir)
    # Simplified directory structure
    plots_dir = os.path.join(cfg.output_dir, "plots")
    ensure_dir(plots_dir)
    try_configure_korean_font()

    # Load strictly split data following our validation rules
    logger.info("Loading split datasets from 1_Split.py...")
    train_data, validation_data, oot_data = load_split_datasets()
    
    # Combine train+validation for final model training (strict rule)
    training_data = get_training_data(train_data, validation_data)
    
    # All data for predictions (but maintain split awareness)
    df = pd.concat([train_data, validation_data, oot_data], ignore_index=True)
    df = df.sort_values(cfg.date_column).reset_index(drop=True)
    
    logger.info(f"Final dataset for predictions: {len(df):,} rows")
    logger.info(f"Training dataset: {len(training_data):,} rows (TRAIN + VALIDATION only)")
    
    feature_cols, target_cols, exclude_cols = get_feature_and_target_columns(df)
    X_all = df[feature_cols].copy()
    X_training = training_data[feature_cols].copy()

    # Detect best params and potentially architecture
    approach, best_params_map, meta = detect_best_params(cfg)


    # 🔑 SIMPLIFIED: Use unique_id as primary key (no more complex composite keys!)
    if 'unique_id' not in df.columns:
        raise ValueError("unique_id column not found. Please run 1_Split.py first to generate unique IDs.")
    
    # Prepare predictions frame with unique_id and essential metadata
    preds_df = df[['unique_id']].copy()
    
    # Include essential metadata for downstream processing
    if cfg.date_column in df.columns:
        preds_df[cfg.date_column] = df[cfg.date_column]
    if 'data_split' in df.columns:
        preds_df['data_split'] = df['data_split']
    
    logger.info(f"Using unique_id as primary key - eliminates complex merge logic!")
    logger.info(f"Predictions will be linked via unique_id: {preds_df['unique_id'].nunique():,} unique records")

    model_manifest: Dict[str, any] = {
        "execution_date": datetime.now().isoformat(),
        "approach": approach,
        "use_gpu": False,
        "dataset_path": cfg.dataset_path,
        "predictions_path": cfg.predictions_path,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "exclude_cols": exclude_cols,
        "meta_from_step7": meta,
    }

    if approach == "individual":
        # Train one model per target and predict all rows
        for _, target in enumerate(sorted(target_cols)):
            # Guard to only accept expected targets risk_year1..4
            if not target.startswith("risk_year"):
                continue
            params = best_params_map.get(target, best_params_map.get(str(target), {}))
            model = xgb_classifier_from_params(params, cfg.random_state)

            # Use only training data (TRAIN + VALIDATION) for model fitting
            mask = ~pd.isna(training_data[target])
            X_train = X_training.loc[mask]
            y_train = training_data[target].loc[mask].astype(int).values
            if len(y_train) == 0:
                # No labels; skip training and fill uniform probabilities
                uniform = np.full((len(df), 4), 0.25, dtype=float)
                try:
                    year_idx = int(str(target).replace("risk_year", ""))
                except Exception:
                    year_idx = 1
                for cls in range(4):
                    preds_df[f"proba_y{year_idx}_{cls}"] = uniform[:, cls]
                continue

            sample_weights = compute_sample_weight("balanced", y_train)
            model.fit(X_train, y_train, sample_weight=sample_weights)

            # Persist booster
            save_model_booster(model, os.path.join(cfg.output_dir, f"individual_{target}.json"))

            # Predict for all rows (production predictions)
            proba_all = model.predict_proba(X_all)
            proba_full = expand_proba_to_4_classes(getattr(model, "classes_", np.array([0, 1, 2, 3])), proba_all)
            # Map target name to year index explicitly
            try:
                year_idx = int(str(target).replace("risk_year", ""))
            except Exception:
                # Fallback to 1-based index if parsing fails (should not happen)
                year_idx = 1
            for cls in range(4):
                preds_df[f"proba_y{year_idx}_{cls}"] = proba_full[:, cls]


            # Feature importance (gain) plot - only for first target to avoid duplicates
            if target == sorted(target_cols)[0]:
                fi_df = feature_importance_gain(model, feature_cols)
                plot_feature_importance(
                    fi_df,
                    os.path.join(plots_dir, f"feature_importance_individual.png"),
                    title=f"특성 중요도 (Gain) - Individual Models",
                    top_n=15,
                )

    else:
        # Unified model with task_id
        frames: List[pd.DataFrame] = []
        labels: List[np.ndarray] = []
        task_id_map = {f"risk_year{i}": (i - 1) for i in range(1, 5)}
        for t in range(1, 5):
            target = f"risk_year{t}"
            if target not in training_data.columns:
                continue
            # Use only training data (TRAIN + VALIDATION) for model fitting
            mask = ~pd.isna(training_data[target])
            if mask.sum() == 0:
                continue
            X_t = X_training.loc[mask].copy()
            X_t["task_id"] = task_id_map[target]
            frames.append(X_t)
            labels.append(training_data[target].loc[mask].astype(int).values)
        if not frames:
            raise ValueError("No labeled rows found for any target. Cannot train unified model.")

        X_stacked = pd.concat(frames, axis=0).reset_index(drop=True)
        y_stacked = np.concatenate(labels, axis=0)

        params = best_params_map.get("unified", {})
        model = xgb_classifier_from_params(params, cfg.random_state)
        sample_weights = compute_sample_weight("balanced", y_stacked)
        model.fit(X_stacked, y_stacked, sample_weight=sample_weights)

        # Persist booster
        save_model_booster(model, os.path.join(cfg.output_dir, "unified_model.json"))

        # Predict for each target by setting task_id (production predictions)
        for t in range(1, 5):
            X_pred = X_all.copy()
            X_pred["task_id"] = t - 1
            proba_all = model.predict_proba(X_pred)
            proba_full = expand_proba_to_4_classes(getattr(model, "classes_", np.array([0, 1, 2, 3])), proba_all)
            for cls in range(4):
                preds_df[f"proba_y{t}_{cls}"] = proba_full[:, cls]


        # Unified feature importance (overall)
        fi_df = feature_importance_gain(model, feature_cols + ["task_id"])
        if not fi_df.empty:
            fi_df = fi_df[fi_df["feature"] != "task_id"]
        plot_feature_importance(
            fi_df,
            os.path.join(plots_dir, "feature_importance_overall.png"),
            title="특성 중요도 (Gain) - Unified",
            top_n=15,
        )

        # Compute individual contributions for all predictions
        logger.info("Computing individual feature contributions...")
        X_for_contrib = X_all.copy()
        X_for_contrib["task_id"] = 0  # Use task_id=0 for contribution calculation
        # Get feature names used in training (including task_id)
        training_features = feature_cols + ["task_id"]
        key_values = df['unique_id']  # Use unique_id instead of complex key
        contributions_df = compute_individual_contributions(model, X_for_contrib, training_features, key_values)

    # Merge contributions with predictions if available
    if approach == "unified" and 'contributions_df' in locals() and not contributions_df.empty:
        preds_df = pd.merge(preds_df, contributions_df, on='unique_id', how='left')  # Use unique_id
        logger.info(f"Added individual contributions for {len(contributions_df)} predictions")
    
    # Save enhanced predictions with contributions (for Model 9 to use)
    preds_df.to_csv(cfg.predictions_path, index=False)
    
    logger.info(f"Predictions with contributions saved: {cfg.predictions_path}")
    logger.info(f"Total predictions: {len(preds_df):,} rows")
    if 'top_var_1' in preds_df.columns:
        logger.info(f"Individual contributions included: ✅")
    else:
        logger.info(f"Individual contributions: ❌ (will use fallback in Model 9)")

    # Generate README and simplified metadata
    generate_readme(cfg.output_dir, feature_cols, model_manifest)
    
    # Save business variable mapping
    var_mapping = create_business_variable_mapping()
    mapping_path = os.path.join(cfg.output_dir, "variable_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(var_mapping, f, indent=2, ensure_ascii=False)
    
    # Simplified metadata (only essential info)
    essential_metadata = {
        "execution_date": model_manifest["execution_date"],
        "approach": approach,
        "total_predictions": len(preds_df),
        "feature_count": len(feature_cols),
        "predictions_file": cfg.predictions_path
    }
    
    with open(os.path.join(cfg.output_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(essential_metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Step 8 completed successfully!")
    logger.info(f"  Predictions: {cfg.predictions_path}")
    logger.info(f"  Model README: {os.path.join(cfg.output_dir, 'README.md')}")
    logger.info(f"  Variable mapping: {mapping_path}")


if __name__ == "__main__":
    main()


