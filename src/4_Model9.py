"""
Final Grading Engine with Strict Data Separation (v1)
====================================================

Purpose
- Convert four annual ordinal predictions into discrete grades
- Respect strict data separation rules from 1_Split.py

Key Changes for Strict Validation:
- Grade rule creation: Use TRAIN + VALIDATION data only
- Grade application: Apply rules to all data (train+validation+oot)
- Calibration tables: Built from TRAIN + VALIDATION only
- Never use OOT data for any rule creation or calibration

Design Highlights
- Per-year ordinal-to-event calibration using empirical event rates r_{t,k} with isotonic regression
- Hazard transform h_t = -ln(1 - p_any_t) with epsilon clipping
- Monotonic supervised binning: quantile pre-bins + Pool-Adjacent-Violators merging
- Grade rules created from TRAIN+VALIDATION, applied to all data

Inputs (defaults)
- predictions_path: ../results/predictions/yearly_multiclass_proba.csv (from Step 3 with split labels)
- Uses split datasets from ../data/splits/ directory

Outputs
- ../results/grading/grade_assignments.csv (all data with grades)
- ../results/grading/grade_bins.json (rules created from train+validation)
- ../results/grading/calibration_tables.csv
- ../results/grading/config_used.json

Data Usage Rules:
- Rule Creation: TRAIN + VALIDATION only
- Rule Application: All data (train+validation+oot)
- OOT data: Gets grades but never used for rule creation
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

# Suppress pandas FutureWarnings about fillna downcasting
pd.set_option('future.no_silent_downcasting', True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define consistent grade order (safe to risky)
GRADE_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
# Add Excel writing capability
try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("Warning: openpyxl not available. Excel output will be disabled.")


# -----------------------------
# Configuration dataclasses
# -----------------------------


@dataclass
class GradingConfig:
    dataset_path: str = "../data/processed/credit_risk_dataset_selected.csv"
    predictions_path: str = "../results/predictions/yearly_multiclass_proba.csv"
    output_dir: str = "../results/grading"
    weights: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)
    epsilon: float = 1e-6
    target_num_grades: int = 7
    grade_labels: Optional[List[str]] = None  # default derived if None
    min_prebin_count: int = 50  # minimum samples per pre-bin after quantile split safeguard
    max_prebins: int = 100
    min_final_bin_frac: float = 0.05  # 5% of dev sample per final bin minimum
    min_final_bin_events: int = 30  # minimum events per final bin
    reason_merge_threshold: float = 0.03  # 3 percentage points
    calibration_start: Optional[str] = None  # e.g., '2018-01-01'
    calibration_end: Optional[str] = None  # e.g., '2022-12-31'
    date_column: str = "보험청약일자"
    key_column: str = "청약번호"
    auto_weight_selection: bool = False
    # Simple constrained candidate sets for auto-weight selection (kept lean)
    candidate_weights: Optional[List[Tuple[float, float, float, float]]] = None


# -----------------------------
# Utilities
# -----------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args() -> GradingConfig:
    parser = argparse.ArgumentParser(description="KSURE Final Grading Engine (v1)")
    parser.add_argument("--dataset_path", type=str, default="../data/processed/credit_risk_dataset_selected.csv")
    parser.add_argument("--predictions_path", type=str, default="../results/predictions/yearly_multiclass_proba.csv")
    parser.add_argument("--output_dir", type=str, default="../results/grading")
    parser.add_argument("--weights", type=float, nargs=4, default=[0.4, 0.3, 0.2, 0.1], help="Year weights w1 w2 w3 w4 (sum=1)")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--grades", type=str, nargs="*", default=None, help="Grade labels low-risk→high-risk (e.g., AAA AA A BBB BB B CCC)")
    parser.add_argument("--num_grades", type=int, default=7)
    parser.add_argument("--min_prebin_count", type=int, default=50)
    parser.add_argument("--max_prebins", type=int, default=100)
    parser.add_argument("--min_final_bin_frac", type=float, default=0.05)
    parser.add_argument("--min_final_bin_events", type=int, default=30)
    parser.add_argument("--reason_merge_threshold", type=float, default=0.03)
    parser.add_argument("--calibration_start", type=str, default=None)
    parser.add_argument("--calibration_end", type=str, default=None)
    parser.add_argument("--date_column", type=str, default="보험청약일자")
    parser.add_argument("--key_column", type=str, default="청약번호")
    parser.add_argument("--auto_weight_selection", action="store_true")
    args = parser.parse_args()

    # Normalize weights
    w = tuple(float(x) for x in args.weights)
    w_sum = sum(w)
    if w_sum <= 0:
        raise ValueError("Weights must sum to a positive number")
    weights = tuple(float(x / w_sum) for x in w)

    return GradingConfig(
        dataset_path=args.dataset_path,
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
        weights=weights,
        epsilon=args.epsilon,
        target_num_grades=args.num_grades,
        grade_labels=args.grades if args.grades else None,
        min_prebin_count=args.min_prebin_count,
        max_prebins=args.max_prebins,
        min_final_bin_frac=args.min_final_bin_frac,
        min_final_bin_events=args.min_final_bin_events,
        reason_merge_threshold=args.reason_merge_threshold,
        calibration_start=args.calibration_start,
        calibration_end=args.calibration_end,
        date_column=args.date_column,
        key_column=args.key_column,
        auto_weight_selection=args.auto_weight_selection,
        candidate_weights=None,
    )


def read_data(cfg: GradingConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(cfg.dataset_path)
    preds = pd.read_csv(cfg.predictions_path)
    if cfg.date_column in df.columns:
        try:
            df[cfg.date_column] = pd.to_datetime(df[cfg.date_column])
        except Exception:
            pass
    return df, preds

def load_split_datasets_for_grading() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load split datasets for grading with strict data separation
    
    Returns:
        Tuple of (train_data, validation_data, oot_data)
    """
    split_dir = "data_splits"
    
    train_data = pd.read_csv(os.path.join(split_dir, "train_data.csv"))
    validation_data = pd.read_csv(os.path.join(split_dir, "validation_data.csv"))
    oot_data = pd.read_csv(os.path.join(split_dir, "oot_data.csv"))
    
    # Ensure date columns are datetime
    for data in [train_data, validation_data, oot_data]:
        if '보험청약일자' in data.columns:
            data['보험청약일자'] = pd.to_datetime(data['보험청약일자'])
    
    logger.info(f"Loaded split datasets for grading:")
    logger.info(f"  - Train: {len(train_data):,} rows")
    logger.info(f"  - Validation: {len(validation_data):,} rows") 
    logger.info(f"  - OOT: {len(oot_data):,} rows")
    
    return train_data, validation_data, oot_data

def get_grade_rule_data(train_data: pd.DataFrame, validation_data: pd.DataFrame) -> pd.DataFrame:
    """
    Combine train and validation data for grade rule creation
    This follows strict validation: grade rules use TRAIN+VALIDATION only
    """
    rule_data = pd.concat([train_data, validation_data], ignore_index=True)
    rule_data = rule_data.sort_values('보험청약일자').reset_index(drop=True)
    
    logger.info(f"Grade rule creation data: {len(rule_data):,} rows")
    logger.info(f"  - Period: {rule_data['보험청약일자'].min()} to {rule_data['보험청약일자'].max()}")
    
    return rule_data


def required_proba_columns() -> List[str]:
    cols = []
    for t in range(1, 5):
        for k in range(4):
            cols.append(f"proba_y{t}_{k}")
    return cols


def validate_predictions(preds: pd.DataFrame, cfg: GradingConfig) -> None:
    missing = [c for c in required_proba_columns() if c not in preds.columns]
    if missing:
        raise ValueError(f"Predictions file missing required columns: {missing}")
    
    # 🔑 SIMPLIFIED: Validate unique_id instead of complex composite key
    if 'unique_id' not in preds.columns:
        raise ValueError("Predictions file missing unique_id column. Please run updated pipeline.")
    
    # Check for unique_id duplicates (should not happen by design)
    unique_id_duplicates = preds['unique_id'].duplicated(keep=False)
    if unique_id_duplicates.any():
        logger.error(f"ERROR: {unique_id_duplicates.sum()} duplicate unique_id values found in predictions!")
        raise ValueError("unique_id duplicates detected - this should not happen")
    else:
        logger.info(f"unique_id validation passed: {len(preds):,} unique records")


def validate_data_integrity(df: pd.DataFrame, preds: pd.DataFrame, cfg: GradingConfig) -> bool:
    """🔑 SIMPLIFIED: Validate data integrity for unique_id approach"""
    # Check for unique_id columns
    if 'unique_id' not in df.columns or 'unique_id' not in preds.columns:
        return False
    
    # Check for required prediction columns
    pred_cols = [f'proba_y{t}_{k}' for t in range(1,5) for k in range(4)]
    if not all(col in preds.columns for col in pred_cols):
        return False
    
    return True


def join_dataset_predictions(df: pd.DataFrame, preds: pd.DataFrame, cfg: GradingConfig) -> pd.DataFrame:
    """
    🔑 SIMPLIFIED: Join using unique_id as primary key (eliminates complex composite key logic!)
    """
    # Check for unique_id in both dataframes
    if 'unique_id' not in df.columns:
        raise ValueError("unique_id column not found in dataset. Please run 1_Split.py first to generate unique IDs.")
    if 'unique_id' not in preds.columns:
        raise ValueError("unique_id column not found in predictions. Please run 3_Model8.py with updated data.")
    
    logger.info(f"Using unique_id as primary key - no more complex composite key logic!")
    
    # Simple, reliable merge using unique_id
    merged = pd.merge(df, preds, on='unique_id', how="inner", suffixes=("", "_pred"))
    
    logger.info(f"Merged {len(df):,} dataset rows with {len(preds):,} prediction rows -> {len(merged):,} final rows")
    
    # With unique_id, row multiplication is impossible by design
    if len(merged) != min(len(df), len(preds)):
        logger.info(f"Note: Merged row count differs from input - this is expected with inner join")
        logger.info(f"  Dataset: {len(df):,} rows, Predictions: {len(preds):,} rows, Merged: {len(merged):,} rows")
    else:
        logger.info(f"Perfect 1:1 merge - no data multiplication issues!")
    
    return merged


def filter_calibration_window(df: pd.DataFrame, cfg: GradingConfig) -> pd.DataFrame:
    if cfg.calibration_start is None and cfg.calibration_end is None:
        return df.copy()
    d = df.copy()
    if cfg.calibration_start is not None:
        try:
            d = d[d[cfg.date_column] >= pd.to_datetime(cfg.calibration_start)]
        except Exception:
            pass
    if cfg.calibration_end is not None:
        try:
            d = d[d[cfg.date_column] <= pd.to_datetime(cfg.calibration_end)]
        except Exception:
            pass
    return d


def compute_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for t in range(1, 5):
        col = f"risk_year{t}"
        out[f"event_{t}y"] = np.where(~pd.isna(out[col]) & (out[col] >= 1), 1, np.where(~pd.isna(out[col]), 0, np.nan))
    return out


def argmax_class_row(row: pd.Series, t: int) -> int:
    vals = [row[f"proba_y{t}_{k}"] for k in range(4)]
    return int(np.argmax(vals))


def compute_predicted_classes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for t in range(1, 5):
        out[f"pred_class_y{t}"] = out.apply(lambda r: argmax_class_row(r, t), axis=1)
    return out


def isotonic_monotone_fit_over_classes(rates_by_class: Dict[int, float]) -> Dict[int, float]:
    xs = np.array(sorted(rates_by_class.keys()), dtype=float)
    ys = np.array([rates_by_class[int(k)] for k in xs], dtype=float)
    iso = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0)
    yhat = iso.fit_transform(xs, ys)
    return {int(k): float(v) for k, v in zip(xs, yhat)}


def compute_yearly_calibration_tables(merged: pd.DataFrame, cfg: GradingConfig) -> Tuple[pd.DataFrame, Dict[int, Dict[int, float]]]:
    """
    Returns:
        calib_table_df: long table with columns [year, cls, raw_event_rate, monotone_event_rate, count]
        r_tk: dict year -> dict class -> monotone event rate
    """
    dev = filter_calibration_window(merged, cfg)
    calib_rows = []
    r_tk: Dict[int, Dict[int, float]] = {}
    for t in range(1, 5):
        event_col = f"event_{t}y"
        ymask = ~pd.isna(dev[event_col])
        if ymask.sum() == 0:
            # fallback: use global baseline 1% per year if no labels
            base = {0: 0.005, 1: 0.01, 2: 0.03, 3: 0.10}
            r_tk[t] = isotonic_monotone_fit_over_classes(base)
            for k in range(4):
                calib_rows.append({
                    "year": t, "cls": k, "raw_event_rate": base[k],
                    "monotone_event_rate": r_tk[t][k], "count": 0
                })
            continue
        sub = dev.loc[ymask, [event_col, f"pred_class_y{t}"]].copy()
        # Aggregate raw event rates by predicted class
        raw = (
            sub.groupby(f"pred_class_y{t}")[event_col]
            .agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "raw_event_rate", "count": "count", f"pred_class_y{t}": "cls"})
        )
        # Ensure all classes present
        present = set(raw["cls"].astype(int).tolist())
        for k in range(4):
            if k not in present:
                raw = pd.concat([
                    raw,
                    pd.DataFrame({"cls": [k], "raw_event_rate": [sub[event_col].mean()], "count": [0]})
                ], ignore_index=True)
        raw = raw.sort_values("cls")
        monotone = isotonic_monotone_fit_over_classes(dict(zip(raw["cls"].astype(int), raw["raw_event_rate"].astype(float))))
        r_tk[t] = monotone
        for _, r in raw.iterrows():
            k = int(r["cls"]) 
            calib_rows.append({
                "year": t,
                "cls": k,
                "raw_event_rate": float(r["raw_event_rate"]),
                "monotone_event_rate": float(monotone[k]),
                "count": int(r["count"]) 
            })
    calib_table_df = pd.DataFrame(calib_rows)
    return calib_table_df, r_tk


def compute_p_any(merged: pd.DataFrame, r_tk: Dict[int, Dict[int, float]], epsilon: float) -> pd.DataFrame:
    out = merged.copy()
    for t in range(1, 5):
        p = np.zeros(len(out), dtype=float)
        for k in range(4):
            p += out[f"proba_y{t}_{k}"].astype(float) * float(r_tk[t][k])
        p = np.clip(p, epsilon, 1.0 - epsilon)
        out[f"p_any_{t}y"] = p
    return out


def compute_hazard_and_score(df: pd.DataFrame, weights: Tuple[float, float, float, float], epsilon: float) -> pd.DataFrame:
    out = df.copy()
    h_cols = []
    contrib_cols = []
    for idx, t in enumerate(range(1, 5)):
        p = out[f"p_any_{t}y"].astype(float).clip(epsilon, 1.0 - epsilon)
        h = -np.log(1.0 - p)
        out[f"h_{t}y"] = h
        h_cols.append(f"h_{t}y")
        contrib = float(weights[idx]) * h
        out[f"contrib_{t}y"] = contrib
        contrib_cols.append(f"contrib_{t}y")
    H = sum(float(weights[idx]) * out[c] for idx, c in enumerate(h_cols))
    out["final_score"] = 1.0 - np.exp(-H)
    total_contrib = np.maximum(sum(out[c] for c in contrib_cols), 1e-12)
    for t in range(1, 5):
        out[f"share_{t}y"] = (out[f"contrib_{t}y"] / total_contrib).astype(float)
    return out


def reason_code_from_shares(row: pd.Series, threshold: float) -> str:
    labels = {1: "단기(1년)", 2: "중기(2년)", 3: "중장기(3년)", 4: "장기(4년)"}
    shares = [(t, float(row[f"share_{t}y"])) for t in range(1, 5)]
    shares.sort(key=lambda x: x[1], reverse=True)
    (t1, s1), (t2, s2) = shares[0], shares[1]
    if abs(s1 - s2) <= threshold:
        return f"{labels[t1]} 및 {labels[t2]} 리스크 복합"
    return f"{labels[t1]} 리스크 기여도 높음"


# SHAP functionality removed - using feature importance instead


def create_excel_output(assignments: pd.DataFrame, grade_stats: pd.DataFrame, 
                       output_path: str, var_mapping: Dict[str, str] = None) -> None:
    """
    Create Excel file with multiple sheets for business users.
    """
    if not EXCEL_AVAILABLE:
        logger.warning("Excel output skipped - openpyxl not available")
        return
    
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Main predictions
            main_cols = ['unique_id', '보험청약일자', 'final_score', 'grade', 'reason_code']
            main_df = assignments[main_cols].copy()
            
            main_df.to_excel(writer, sheet_name='예측결과', index=False)
            
            # Sheet 2: Grade statistics
            grade_stats.to_excel(writer, sheet_name='등급별통계', index=False)
            
            # Sheet 3: Usage guide
            guide_data = {
                '항목': ['final_score', 'grade', 'reason_code'],
                '설명': [
                    '0-1 스케일의 리스크 점수 (높을수록 위험)',
                    '최종 등급 (AAA가 가장 안전, CCC가 가장 위험)',
                    '등급 판정의 주요 근거 (시기별 기여도)'
                ]
            }
            pd.DataFrame(guide_data).to_excel(writer, sheet_name='사용가이드', index=False)
        
        logger.info(f"Excel output created: {output_path}")
        
    except Exception as e:
        logger.error(f"Excel creation failed: {e}")
        # Fallback to CSV
        assignments.to_csv(output_path.replace('.xlsx', '_fallback.csv'), index=False)


# -----------------------------
# Monotonic supervised binning
# -----------------------------


def _prebin_quantiles(scores: np.ndarray, y: np.ndarray, max_prebins: int, min_prebin_count: int) -> pd.DataFrame:
    n = len(scores)
    if n == 0:
        return pd.DataFrame(columns=["bin_id", "left", "right", "count", "events", "bad_rate"]) 
    # target number of prebins based on size
    est_bins = min(max_prebins, max(10, int(math.sqrt(n))))
    # quantile edges (unique, bounded)
    quantiles = np.linspace(0, 1, est_bins + 1)
    edges = np.unique(np.quantile(scores, quantiles))
    # Assign bins
    bin_ids = np.digitize(scores, bins=edges[1:-1], right=False)
    # Build dataframe
    df = pd.DataFrame({"bin_id": bin_ids, "score": scores, "y": y})
    agg = df.groupby("bin_id").agg(count=("y", "size"), events=("y", "sum"), left=("score", "min"), right=("score", "max")).reset_index()
    agg["bad_rate"] = np.where(agg["count"] > 0, agg["events"] / agg["count"], 0.0)
    agg = agg.sort_values("bin_id").reset_index(drop=True)
    # Merge undersized prebins with neighbors
    merged = []
    for _, r in agg.iterrows():
        if not merged:
            merged.append(r.to_dict())
            continue
        if merged[-1]["count"] < min_prebin_count or int(r["count"]) < min_prebin_count:
            # merge with previous
            last = merged.pop()
            merged.append({
                "bin_id": last["bin_id"],
                "left": min(last["left"], r["left"]),
                "right": max(last["right"], r["right"]),
                "count": int(last["count"]) + int(r["count"]),
                "events": float(last["events"]) + float(r["events"]),
                "bad_rate": 0.0,  # to be recomputed
            })
        else:
            merged.append(r.to_dict())
    if not merged:
        return agg
    # recompute bad rates
    for m in merged:
        cnt = max(int(m["count"]), 1)
        m["bad_rate"] = float(m["events"]) / float(cnt)
    return pd.DataFrame(merged)


def _pav_monotone_merge(bins_df: pd.DataFrame) -> pd.DataFrame:
    # Pool Adjacent Violators to enforce non-decreasing bad_rate
    bins = [{
        "left": float(r["left"]),
        "right": float(r["right"]),
        "count": int(r["count"]),
        "events": float(r["events"]),
        "bad_rate": float(r["bad_rate"]),
    } for _, r in bins_df.sort_values("left").iterrows()]

    i = 1
    while i < len(bins):
        if bins[i]["bad_rate"] + 1e-12 < bins[i - 1]["bad_rate"]:
            # merge i-1 and i
            merged = {
                "left": bins[i - 1]["left"],
                "right": bins[i]["right"],
                "count": bins[i - 1]["count"] + bins[i]["count"],
                "events": bins[i - 1]["events"] + bins[i]["events"],
            }
            merged["bad_rate"] = merged["events"] / max(merged["count"], 1)
            bins[i - 1] = merged
            del bins[i]
            if i > 1:
                i -= 1
        else:
            i += 1
    out = pd.DataFrame(bins)
    return out


def _reduce_to_target_bins(monotone_df: pd.DataFrame, target_bins: int, min_frac: float, min_events: int) -> pd.DataFrame:
    df = monotone_df.sort_values("left").reset_index(drop=True)
    total = int(df["count"].sum())
    # Enforce minimums first
    i = 0
    while i < len(df) - 1:
        need_merge = (df.loc[i, "count"] < max(1, int(min_frac * total))) or (df.loc[i, "events"] < min_events)
        if need_merge:
            # merge with neighbor that yields smallest bad_rate jump
            j = i + 1
            merged = {
                "left": float(df.loc[i, "left"]),
                "right": float(df.loc[j, "right"]),
                "count": int(df.loc[i, "count"]) + int(df.loc[j, "count"]),
                "events": float(df.loc[i, "events"]) + float(df.loc[j, "events"]),
            }
            merged["bad_rate"] = merged["events"] / max(merged["count"], 1)
            df.loc[i, ["right", "count", "events", "bad_rate"]] = [merged["right"], merged["count"], merged["events"], merged["bad_rate"]]
            df = df.drop(index=i + 1).reset_index(drop=True)
            i = max(i - 1, 0)
            continue
        i += 1
    # Then reduce to target_bins by merging adjacent pairs with smallest bad_rate difference
    while len(df) > target_bins:
        diffs = [(idx, abs(df.loc[idx + 1, "bad_rate"] - df.loc[idx, "bad_rate"])) for idx in range(len(df) - 1)]
        if not diffs:
            break
        best_idx = min(diffs, key=lambda x: x[1])[0]
        merged = {
            "left": float(df.loc[best_idx, "left"]),
            "right": float(df.loc[best_idx + 1, "right"]),
            "count": int(df.loc[best_idx, "count"]) + int(df.loc[best_idx + 1, "count"]),
            "events": float(df.loc[best_idx, "events"]) + float(df.loc[best_idx + 1, "events"]),
        }
        merged["bad_rate"] = merged["events"] / max(merged["count"], 1)
        df.loc[best_idx, ["right", "count", "events", "bad_rate"]] = [merged["right"], merged["count"], merged["events"], merged["bad_rate"]]
        df = df.drop(index=best_idx + 1).reset_index(drop=True)
    return df


def monotone_supervised_binning(scores: np.ndarray, y: np.ndarray, cfg: GradingConfig) -> Tuple[pd.DataFrame, List[float]]:
    """
    Returns
        final_bins_df: columns [left, right, count, events, bad_rate, bin_id]
        cutpoints: list of right edges excluding +inf; these define bins in ascending risk
    """
    # Initial pre-binning by quantiles
    pre = _prebin_quantiles(scores, y, cfg.max_prebins, cfg.min_prebin_count)
    if len(pre) == 0:
        raise ValueError("No data to bin.")
    # Enforce monotonic bad_rate via PAV
    mono = _pav_monotone_merge(pre)
    # Reduce to target grades while keeping monotonicity
    final_df = _reduce_to_target_bins(mono, cfg.target_num_grades, cfg.min_final_bin_frac, cfg.min_final_bin_events)
    final_df = final_df.sort_values("left").reset_index(drop=True)
    final_df["bin_id"] = list(range(len(final_df)))
    # Cutpoints are right edges except last
    cutpoints = [float(x) for x in final_df["right"].values[:-1]]
    return final_df, cutpoints


def assign_grade_from_cutpoints(scores: np.ndarray, cutpoints: List[float], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    # Digitize assigns 0..len(cutpoints) based on right edges
    bin_idx = np.digitize(scores, bins=np.array(cutpoints), right=True)
    # Map ascending risk to grade labels
    if len(labels) != len(set(bin_idx)) and len(labels) != (len(cutpoints) + 1):
        # fallback: generate generic labels
        labels = [f"G{i}" for i in range(len(cutpoints) + 1)]
    grades = np.array([labels[i] for i in bin_idx], dtype=object)
    return grades, bin_idx


def default_grade_labels(n: int) -> List[str]:
    presets = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
    if n <= len(presets):
        return presets[:n]
    # extend if needed
    return [f"G{i+1}" for i in range(n)]


def try_configure_korean_font() -> None:
    try:
        for f in ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "AppleGothic"]:
            plt.rcParams["font.family"] = f
            break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def plot_grade_distribution(assignments: pd.DataFrame, labels: List[str], output_dir: str) -> None:
    if "grade" not in assignments.columns or len(assignments) == 0:
        return
    try_configure_korean_font()
    # Order grades by consistent GRADE_ORDER (safe to risky)
    available_grades = [g for g in GRADE_ORDER if g in assignments["grade"].unique().tolist()]
    dist = assignments["grade"].value_counts().reindex(available_grades, fill_value=0)
    total = max(int(dist.sum()), 1)
    prop = (dist / total).astype(float)
    plt.figure(figsize=(7.5, 4.5))
    sns.barplot(x=dist.index, y=dist.values, color="#4C78A8")
    for idx, v in enumerate(dist.values):
        plt.text(idx, v, f"{v:,}\n({prop.values[idx]*100:.1f}%)", ha="center", va="bottom", fontsize=9)
    plt.xlabel("등급 (Grade)")
    plt.ylabel("표본 수 (Count)")
    plt.title("등급 분포 (Grade Distribution)")
    plt.tight_layout()
    path = os.path.join(output_dir, "grade_distribution.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def select_weights_by_cv(dev_df: pd.DataFrame, cfg: GradingConfig) -> Tuple[float, float, float, float]:
    """
    Lean, constrained search for weights that maximize mean AUC (AR equivalent) on development rows.
    🔄 CHANGED: Now optimizes for 1-year event prediction instead of 4-year.
    """
    # Candidate set if not provided: a small, ordered family
    candidates = cfg.candidate_weights or [
        (0.4, 0.3, 0.2, 0.1),
        (0.5, 0.25, 0.15, 0.1),
        (0.35, 0.3, 0.2, 0.15),
        (0.3, 0.3, 0.25, 0.15),
    ]
    # Normalize
    candidates = [tuple(np.array(c) / (sum(c) if sum(c) > 0 else 1.0)) for c in candidates]

    # Build simple temporal folds: 5 slices by date if available, else uniform index slices
    if cfg.date_column in dev_df.columns and np.issubdtype(dev_df[cfg.date_column].dtype, np.datetime64):
        dev_sorted = dev_df.sort_values(cfg.date_column).reset_index(drop=True)
    else:
        dev_sorted = dev_df.reset_index(drop=True)
    n = len(dev_sorted)
    k = 5
    fold_edges = [int(i * n / k) for i in range(k + 1)]
    folds = [(fold_edges[i], fold_edges[i + 1]) for i in range(k)]

    def auc_for_weights(w: Tuple[float, float, float, float]) -> Tuple[float, float]:
        aucs = []
        for a, b in folds:
            sl = dev_sorted.iloc[a:b]
            # Already has p_any_t and hazards computed for base weights; recompute here
            h_stack = np.column_stack([sl[f"h_{t}y"].values for t in range(1, 5)])
            H = np.dot(h_stack, np.array(w, dtype=float))
            score = 1.0 - np.exp(-H)
            y = (sl["risk_year1"].astype(float) >= 1).astype(int).values  # 🔄 CHANGED: 1-year events
            mask = ~np.isnan(y)
            if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
                continue
            try:
                aucs.append(roc_auc_score(y[mask], score[mask]))
            except Exception:
                continue
        if not aucs:
            return 0.0, 0.0
        return float(np.mean(aucs)), float(np.std(aucs))

    best_w = cfg.weights
    best_key = (-1.0, 1e9)  # maximize mean, minimize std
    for w in candidates:
        mean_auc, std_auc = auc_for_weights(w)
        key = (mean_auc, -std_auc)
        if key > best_key:
            best_key = key
            best_w = w
    return tuple(float(x) for x in best_w)


def main():
    cfg = parse_args()
    ensure_dir(cfg.output_dir)
    log_lines: List[str] = []

    # Load split datasets following strict validation rules
    logger.info("Loading split datasets from 1_Split.py...")
    train_data, validation_data, oot_data = load_split_datasets_for_grading()
    
    # Get grade rule creation data (TRAIN + VALIDATION only)
    rule_data = get_grade_rule_data(train_data, validation_data)
    
    # All data for grade application
    all_data = pd.concat([train_data, validation_data, oot_data], ignore_index=True)
    all_data = all_data.sort_values(cfg.date_column).reset_index(drop=True)
    
    # Load predictions (should contain all data from Step 3)
    preds = pd.read_csv(cfg.predictions_path)
    validate_predictions(preds, cfg)
    
    log_lines.append(f"Rule creation data (Train+Validation): {len(rule_data):,} rows")
    log_lines.append(f"All data for grade application: {len(all_data):,} rows")
    log_lines.append(f"Predictions: {len(preds):,} rows")

    # Join rule data with predictions for grade rule creation
    rule_merged = join_dataset_predictions(rule_data, preds, cfg)
    
    # Join all data with predictions for grade application
    all_merged = join_dataset_predictions(all_data, preds, cfg)
    
    log_lines.append(f"Rule creation merged: {len(rule_merged):,} rows")
    log_lines.append(f"All data merged: {len(all_merged):,} rows")
    
    # Log data integrity information
    if cfg.key_column in rule_merged.columns:
        duplicates = rule_merged[cfg.key_column].duplicated(keep=False)
        if duplicates.any():
            log_lines.append(f"Note: {duplicates.sum():,} duplicate {cfg.key_column} values handled with composite keys")
    
    # Process rule data for grade rule creation
    rule_merged = compute_event_flags(rule_merged)
    rule_merged = compute_predicted_classes(rule_merged)
    
    # Process all data for final grade application
    all_merged = compute_event_flags(all_merged)
    all_merged = compute_predicted_classes(all_merged)

    # Calibration tables - use rule data only (TRAIN + VALIDATION)
    calib_df, r_tk = compute_yearly_calibration_tables(rule_merged, cfg)
    log_lines.append("Per-year class-to-event calibration completed with isotonic monotonicity (TRAIN+VALIDATION only).")

    # Apply calibration to rule data for grade rule creation
    rule_merged = compute_p_any(rule_merged, r_tk, cfg.epsilon)
    rule_merged = compute_hazard_and_score(rule_merged, cfg.weights, cfg.epsilon)
    
    # Apply same calibration to all data for grade application
    all_merged = compute_p_any(all_merged, r_tk, cfg.epsilon)
    all_merged = compute_hazard_and_score(all_merged, cfg.weights, cfg.epsilon)

    # Optional: simple CV-based weight selection on rule data only (TRAIN + VALIDATION)
    if cfg.auto_weight_selection:
        dev_mask = ~pd.isna(rule_merged["risk_year1"])  # 🔄 CHANGED: evaluate vs 1-year event
        dev_df = rule_merged.loc[dev_mask].copy()
        if len(dev_df) > 0:
            new_w = select_weights_by_cv(dev_df, cfg)
            log_lines.append(f"Auto-selected weights from candidates: old={cfg.weights} → new={new_w}")
            cfg.weights = new_w
            # recompute final_score and shares with new weights
            rule_merged = compute_hazard_and_score(rule_merged, cfg.weights, cfg.epsilon)
            all_merged = compute_hazard_and_score(all_merged, cfg.weights, cfg.epsilon)
        else:
            log_lines.append("Auto weight selection skipped (no labeled development rows).")

    # Build supervised monotonic bins ONLY on rule data (TRAIN + VALIDATION)
    dev_mask = ~pd.isna(rule_merged["risk_year1"])  # 🔄 CHANGED: only where 1-year label matured
    dev_rows = rule_merged.loc[dev_mask].copy()
    y_dev = (dev_rows["risk_year1"].astype(float) >= 1).astype(int).values  # 🔄 CHANGED: 1-year events
    scores_dev = dev_rows["final_score"].astype(float).values
    final_bins_df, cutpoints = monotone_supervised_binning(scores_dev, y_dev, cfg)
    log_lines.append(f"Monotonic supervised binning on TRAIN+VALIDATION produced {len(cutpoints)+1} bins (target={cfg.target_num_grades}).")

    # Apply grade rules to ALL data (using cutpoints from TRAIN+VALIDATION)
    labels = cfg.grade_labels or default_grade_labels(len(cutpoints) + 1)
    grades, bin_idx = assign_grade_from_cutpoints(all_merged["final_score"].astype(float).values, cutpoints, labels)
    all_merged["bin_id"] = bin_idx.astype(int)
    all_merged["grade"] = grades

    # Reason codes for all data
    all_merged["reason_code"] = all_merged.apply(lambda r: reason_code_from_shares(r, cfg.reason_merge_threshold), axis=1)
    
    logger.info(f"Applied grades to all data:")
    logger.info(f"  - Grade rules created from: {len(rule_merged):,} rows (TRAIN+VALIDATION)")
    logger.info(f"  - Grades applied to: {len(all_merged):,} rows (ALL DATA)")
    
    # Use all_merged for final outputs (contains all data with grades)
    merged = all_merged

    # SHAP contributions removed - using feature importance for interpretability
    logger.info(f"Using feature importance for model interpretability")

    # Prepare assignments data - 🔑 SIMPLIFIED: Use unique_id as primary identifier
    assign_cols = ['unique_id']  # Primary key
    if cfg.date_column in merged.columns:
        assign_cols.append(cfg.date_column)
    if 'data_split' in merged.columns:
        assign_cols.append('data_split')
    assign_cols += ["final_score", "bin_id", "grade", "reason_code"]
    
    # SHAP top variables removed
    
    assignments = merged[assign_cols].copy()
    
    # Save CSV for compatibility
    assignments_path = os.path.join(cfg.output_dir, "grade_assignments.csv")
    assignments.to_csv(assignments_path, index=False)

    # Generate grade-level statistics for Excel
    grade_stats_rows = []
    total_count = len(assignments)
    # Sort by proper grade order (safe to risky)
    available_grades = [g for g in GRADE_ORDER if g in labels]
    for grade in available_grades:
        grade_data = assignments[assignments['grade'] == grade]
        if len(grade_data) > 0:
            grade_stats_rows.append({
                '등급': grade,
                '건수': len(grade_data),
                '비율': f"{len(grade_data)/total_count*100:.1f}%",
                '평균점수': f"{grade_data['final_score'].mean():.3f}",
                '주요사유': grade_data['reason_code'].mode().iloc[0] if not grade_data['reason_code'].mode().empty else 'N/A'
            })
    
    grade_stats_df = pd.DataFrame(grade_stats_rows)
    
    # Load variable mapping from Model 8
    var_mapping = {}
    mapping_path = "../results/step8_post/variable_mapping.json"
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                var_mapping = json.load(f)
        except Exception:
            pass
    
    # Create Excel output (main deliverable)
    excel_path = os.path.join(cfg.output_dir, "final_predictions.xlsx")
    create_excel_output(assignments, grade_stats_df, excel_path, var_mapping)
    
    # Create business insights guide
    insights_content = f"""# 리스크 등급 분석 결과 해석 가이드

## 등급 체계
- **AAA, AA, A**: 우량 기업 (낮은 리스크)
- **BBB, BB**: 보통 기업 (중간 리스크) 
- **B, CCC**: 주의 기업 (높은 리스크)

## 주요 통계
- 전체 분석 대상: {total_count:,}개 기업
- 우량 등급 (A 이상): {len(assignments[assignments['grade'].isin(['AAA', 'AA', 'A'])]):,}개 ({len(assignments[assignments['grade'].isin(['AAA', 'AA', 'A'])])/total_count*100:.1f}%)
- 주의 등급 (B 이하): {len(assignments[assignments['grade'].isin(['B', 'CCC'])]):,}개 ({len(assignments[assignments['grade'].isin(['B', 'CCC'])])/total_count*100:.1f}%)

## 활용 방법
1. **등급**: 기본적인 리스크 수준 판단
2. **이유코드**: 어느 시기의 리스크가 주요 원인인지 파악
3. **상위 변수**: 해당 등급에 가장 큰 영향을 준 요인들

## 문의사항
상세한 분석이나 해석이 필요한 경우 데이터분석팀으로 연락 바랍니다.
"""
    
    with open(os.path.join(cfg.output_dir, "business_insights.md"), "w", encoding="utf-8") as f:
        f.write(insights_content)
    
    # Simplified metadata (essential only)
    essential_meta = {
        "execution_date": datetime.now().isoformat(),
        "total_predictions": total_count,
        "grade_distribution": {row['등급']: row['건수'] for row in grade_stats_rows},
        "cutpoints": cutpoints,
        "labels": available_grades,  # Use ordered grades
    }
    
    with open(os.path.join(cfg.output_dir, "grade_summary.json"), "w", encoding="utf-8") as f:
        json.dump(essential_meta, f, indent=2, ensure_ascii=False)

    logger.info(f"Grading completed successfully!")
    logger.info(f"  Excel 결과물: {excel_path}")
    logger.info(f"  CSV 호환성: {assignments_path}")
    logger.info(f"  해석 가이드: business_insights.md")
    logger.info(f"  등급 요약: grade_summary.json")


if __name__ == "__main__":
    main()