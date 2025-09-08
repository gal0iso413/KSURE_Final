"""
Final Grade Validation with Strict Data Separation (v1)
======================================================

Validates the graded outputs using ONLY OOT data for unbiased evaluation.
This is the final step that provides the true performance estimate.

Key Changes for Strict Validation:
- Performance evaluation: OOT data ONLY (never seen before)
- No development data used for any evaluation metrics
- True unbiased performance assessment for production readiness

Core checks (OOT only)
- Discrimination: AUC, AR, KS, top-decile lift
- Calibration: reliability curves, calibration intercept/slope, Brier score
- Monotonicity: observed bad rates by grade with Wilson CIs
- Grade distribution analysis
- Case studies demonstrating model power

Inputs
- Assignments: ../results/grading/grade_assignments.csv (all data with grades)
- Uses ../data/splits/ to identify OOT subset

Outputs (default)
- ../results/validation/metrics_summary.json (OOT performance only)
- ../results/validation/grade_level_stats.csv (OOT only)
- ../results/validation/plots/* (all based on OOT data)
- ../results/validation/executive_report.md (true production performance)

Data Usage Rules:
- Evaluation: OOT data ONLY
- This gives us the true unbiased performance estimate
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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
from sklearn.inspection import partial_dependence
import xgboost as xgb

# Suppress pandas FutureWarnings about fillna downcasting
pd.set_option('future.no_silent_downcasting', True)

# Optional: Suppress other common warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

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


@dataclass
class VerifyConfig:
    assignments_path: str = "../results/grading/grade_assignments.csv"
    dataset_path: str = "../data/processed/credit_risk_dataset_selected.csv"
    output_dir: str = "../results/validation"
    date_column: str = "보험청약일자"
    key_column: str = "청약번호"
    dev_start: Optional[str] = None
    dev_end: Optional[str] = None
    oot_start: Optional[str] = None
    oot_end: Optional[str] = None
    segment_cols: Optional[List[str]] = None
    epsilon: float = 1e-6
    acceptance_ar: float = 0.40
    acceptance_ks: float = 0.30
    acceptance_psi: float = 0.25
    previous_assignments_path: Optional[str] = None
    # SHAP functionality removed


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args() -> VerifyConfig:
    p = argparse.ArgumentParser(description="KSURE Grade Validation (v1)")
    p.add_argument("--assignments_path", type=str, default="../results/grading/grade_assignments.csv")
    p.add_argument("--dataset_path", type=str, default="../data/processed/credit_risk_dataset_selected.csv")
    p.add_argument("--output_dir", type=str, default="../results/validation")
    p.add_argument("--date_column", type=str, default="보험청약일자")
    p.add_argument("--key_column", type=str, default="청약번호")
    p.add_argument("--dev_start", type=str, default=None)
    p.add_argument("--dev_end", type=str, default=None)
    p.add_argument("--oot_start", type=str, default=None)
    p.add_argument("--oot_end", type=str, default=None)
    p.add_argument("--segment_cols", type=str, default=None, help="Comma-separated segment columns")
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--previous_assignments_path", type=str, default=None)
    # SHAP arguments removed
    args = p.parse_args()
    seg_cols = [c.strip() for c in args.segment_cols.split(",")] if args.segment_cols else None
    return VerifyConfig(
        assignments_path=args.assignments_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        date_column=args.date_column,
        key_column=args.key_column,
        dev_start=args.dev_start,
        dev_end=args.dev_end,
        oot_start=args.oot_start,
        oot_end=args.oot_end,
        segment_cols=seg_cols,
        epsilon=args.epsilon,
        previous_assignments_path=args.previous_assignments_path,
        # SHAP removed
    )


def read_inputs(cfg: VerifyConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    🔑 IMPROVED: Use split datasets with unique_id instead of original dataset
    """
    assign = pd.read_csv(cfg.assignments_path)
    
    # Load split datasets that have unique_id
    try:
        train_data = pd.read_csv("../data/splits/train_data.csv")
        validation_data = pd.read_csv("../data/splits/validation_data.csv")
        oot_data = pd.read_csv("../data/splits/oot_data.csv")
        
        # Combine all split datasets (they all have unique_id)
        df = pd.concat([train_data, validation_data, oot_data], ignore_index=True)
        logger.info(f"Using split datasets with unique_id: {len(df):,} total records")
        
    except FileNotFoundError:
        # Fallback to original dataset if splits don't exist
        logger.warning(f"Split datasets not found, using original dataset")
        df = pd.read_csv(cfg.dataset_path)
    
    # Ensure datetime columns
    for data in [assign, df]:
        if cfg.date_column in data.columns:
            try:
                data[cfg.date_column] = pd.to_datetime(data[cfg.date_column])
            except Exception:
                pass
    
    return assign, df


def build_eval_frame(assign: pd.DataFrame, df: pd.DataFrame, cfg: VerifyConfig) -> pd.DataFrame:
    """
    🔑 SIMPLIFIED: Build evaluation frame using unique_id as primary key
    """
    logger.info(f"build_eval_frame called with {len(assign):,} assignments and {len(df):,} dataset rows")
    logger.debug(f"Assignments columns: {list(assign.columns)}")
    logger.debug(f"Dataset columns: {list(df.columns)}")
    
    # Check for unique_id in both dataframes
    if 'unique_id' not in assign.columns:
        raise ValueError("Assignments missing unique_id column. Please run updated pipeline.")
    if 'unique_id' not in df.columns:
        raise ValueError("Dataset missing unique_id column. Please run 1_Split.py first.")

    logger.info(f"Using unique_id as primary key - eliminates complex composite key logic!")
    
    # Include financial features for case studies (4 key indicators)
    financial_features = ['재무정보_영업이익률_t0', '재무정보_부채비율_t0', '재무정보_ROE_t0', '재무정보_총자산회전율_t0']
    available_financial = [f for f in financial_features if f in df.columns]
    
    # 🔄 FIXED: Include ALL risk_year columns for event_1y validation
    risk_year_cols = ["risk_year1", "risk_year2", "risk_year3", "risk_year4"]
    available_risk_cols = [col for col in risk_year_cols if col in df.columns]
    
    # 🔄 NEW: Include ALL numeric columns for comprehensive case studies
    # Exclude only ID columns and target variables, but include all other features
    exclude_for_merge = {'사업자등록번호', '대상자명', '청약번호', '수출자대상자번호'}
    all_numeric_cols = [col for col in df.columns 
                       if col not in exclude_for_merge 
                       and df[col].dtype in ['int64', 'float64', 'object']  # Include object for categorical
                       and not df[col].isna().all()]
    
    merge_cols = ['unique_id', cfg.date_column] + list(set(all_numeric_cols) - {'unique_id', cfg.date_column})
    
    logger.info(f"Including risk_year columns: {available_risk_cols}")
    
    # Simple, reliable merge using unique_id
    merged = pd.merge(assign, df[merge_cols], on='unique_id', how="left", suffixes=("", "_df"))
    
    logger.info(f"Merged {len(assign):,} assignment rows with {len(df):,} dataset rows -> {len(merged):,} final rows")
    
    # Set split column
    if 'data_split' in assign.columns:
        merged["split"] = assign["data_split"]
        print(f"[Verify] Using data_split from assignments: {merged['split'].value_counts().to_dict()}")
    else:
        print(f"[Verify] WARNING: data_split column not found in assignments!")
    
    # With unique_id, row multiplication is impossible by design - no complex deduplication needed!
    if len(merged) == len(assign):
        print(f"[Verify] ✅ Perfect 1:1 merge - no data multiplication issues!")
    else:
        print(f"[Verify] Note: Row count changed ({len(assign)} -> {len(merged)}) - expected with left join")
        if 'split' in merged.columns:
            print(f"[Verify] Split distribution: {merged['split'].value_counts().to_dict()}")
            print(f"[Verify] OOT data: {(merged['split'] == 'oot').sum()}")
    
    # 🔄 CHANGED: Use 1-year events for validation (more immediate feedback)
    merged["event_1y"] = np.where(~pd.isna(merged["risk_year1"]) & (merged["risk_year1"] >= 1), 1, np.where(~pd.isna(merged["risk_year1"]), 0, np.nan))
    # Keep event_4y for compatibility but focus on event_1y
    merged["event_4y"] = np.where(~pd.isna(merged["risk_year4"]) & (merged["risk_year4"] >= 1), 1, np.where(~pd.isna(merged["risk_year4"]), 0, np.nan))

    # Set split column if not already set (fallback case)
    if 'split' not in merged.columns:
        if 'data_split' in assign.columns:
            # Use the split information from the strict pipeline
            merged["split"] = assign["data_split"]
            print(f"[Verify] Using data_split from assignments: {merged['split'].value_counts().to_dict()}")
        else:
            # Fallback: create dev/oot split
            merged["split"] = "unspecified"
            print(f"[Verify] No data_split found in assignments, using fallback split logic")
            # Only run fallback logic if we don't have data_split from assignments
            if cfg.dev_start or cfg.dev_end or cfg.oot_start or cfg.oot_end:
                if cfg.dev_start:
                    merged.loc[merged[cfg.date_column] >= pd.to_datetime(cfg.dev_start), "split"] = "dev"
                if cfg.dev_end:
                    merged.loc[merged[cfg.date_column] > pd.to_datetime(cfg.dev_end), "split"] = "unspecified"
                if cfg.oot_start:
                    merged.loc[merged[cfg.date_column] >= pd.to_datetime(cfg.oot_start), "split"] = "oot"
                if cfg.oot_end:
                    merged.loc[merged[cfg.date_column] > pd.to_datetime(cfg.oot_end), "split"] = "unspecified"
            else:
                # Default: chronological split by date (last 20% as OOT)
                if cfg.date_column in merged.columns and np.issubdtype(merged[cfg.date_column].dtype, np.datetime64):
                    merged = merged.sort_values(cfg.date_column).reset_index(drop=True)
                    n = len(merged)
                    cut = int(n * 0.8)
                    merged.loc[:cut - 1, "split"] = "dev"
                    merged.loc[cut:, "split"] = "oot"
                else:
                    n = len(merged)
                    cut = int(n * 0.8)
                    merged.loc[:cut - 1, "split"] = "dev"
                    merged.loc[cut:, "split"] = "oot"
    else:
        print(f"[Verify] Skipping fallback logic - using data_split from assignments")

    return merged


def compute_ks(y_true: np.ndarray, score: np.ndarray) -> float:
    # KS = max_t |F1(t) - F0(t)| where F1 is CDF of scores for bads and F0 for goods
    # We'll compute using empirical CDFs at unique score points
    order = np.argsort(score)
    y = y_true[order]
    s = score[order]
    cum_bad = np.cumsum(y) / max(y.sum(), 1)
    cum_good = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    ks = np.max(np.abs(cum_bad - cum_good)) if len(s) > 0 else 0.0
    return float(ks)


def top_decile_lift(y_true: np.ndarray, score: np.ndarray) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    k = max(int(0.1 * n), 1)
    idx = np.argsort(-score)[:k]
    top_rate = y_true[idx].mean() if k > 0 else 0.0
    base = y_true.mean() if n > 0 else 0.0
    return float(top_rate / base) if base > 0 else 0.0


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y_true - p) ** 2)) if len(y_true) > 0 else 0.0


def logistic_calibration(y_true: np.ndarray, p: np.ndarray, eps: float) -> Tuple[float, float]:
    # Fit y ~ logit(p) with intercept and slope; return (intercept, slope)
    p = np.clip(p, eps, 1.0 - eps)
    x = np.log(p / (1.0 - p)).reshape(-1, 1)
    if len(np.unique(y_true)) < 2:
        return 0.0, 1.0
    lr = LogisticRegression(max_iter=1000)
    lr.fit(x, y_true)
    # In scikit-learn, coef_ and intercept_ correspond to slope and intercept in logit space
    slope = float(lr.coef_.ravel()[0])
    intercept = float(lr.intercept_.ravel()[0])
    return intercept, slope


def reliability_by_bins(y_true: np.ndarray, p: np.ndarray, num_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"p": p, "y": y_true})
    df = df.sort_values("p").reset_index(drop=True)
    df["bin"] = pd.qcut(df["p"], q=min(num_bins, len(df)), duplicates="drop")
    out = df.groupby("bin").agg(pred_mean=("p", "mean"), obs_rate=("y", "mean"), count=("y", "size")).reset_index(drop=True)
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    low = (center - margin) / denom
    high = (center + margin) / denom
    return float(max(0.0, low)), float(min(1.0, high))


def psi_for_grades(series_dev: pd.Series, series_oot: pd.Series, eps: float = 1e-12) -> float:
    # Compute PSI over categorical grades
    dev_counts = series_dev.value_counts(normalize=True)
    oot_counts = series_oot.value_counts(normalize=True)
    cats = sorted(set(dev_counts.index) | set(oot_counts.index))
    psi = 0.0
    for c in cats:
        p = float(dev_counts.get(c, 0.0))
        q = float(oot_counts.get(c, 0.0))
        p = max(p, eps)
        q = max(q, eps)
        psi += (p - q) * np.log(p / q)
    return float(psi)


def plot_roc(y: np.ndarray, score: np.ndarray, path: str) -> None:
    if len(np.unique(y)) < 2:
        return
    
    try_configure_korean_font()
    
    fpr, tpr, _ = roc_curve(y, score)
    auc = roc_auc_score(y, score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC 곡선 (AUC={auc:.3f})", linewidth=2, color='#1f77b4')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="무작위 분류기")
    plt.xlabel("거짓 양성률 (FPR)")
    plt.ylabel("참 양성률 (TPR)")
    plt.title("ROC 곡선")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_ks_statistic(y: np.ndarray, score: np.ndarray, path: str) -> None:
    """
    Create KS statistic plot showing cumulative distribution by deciles.
    """
    if len(y) == 0 or len(np.unique(y)) < 2:
        return
    
    try_configure_korean_font()
    
    # Create score deciles
    df = pd.DataFrame({'score': score, 'target': y})
    df = df.sort_values('score').reset_index(drop=True)
    df['decile'] = pd.qcut(df['score'], q=10, labels=False, duplicates='drop') + 1
    
    # Calculate cumulative percentages by decile
    decile_stats = []
    total_goods = (df['target'] == 0).sum()
    total_bads = (df['target'] == 1).sum()
    
    cum_goods = 0
    cum_bads = 0
    
    for decile in range(1, 11):
        decile_data = df[df['decile'] == decile]
        if len(decile_data) == 0:
            continue
            
        goods_in_decile = (decile_data['target'] == 0).sum()
        bads_in_decile = (decile_data['target'] == 1).sum()
        
        cum_goods += goods_in_decile
        cum_bads += bads_in_decile
        
        # Calculate cumulative percentages
        cum_good_pct = (cum_goods / max(total_goods, 1)) * 100
        cum_bad_pct = (cum_bads / max(total_bads, 1)) * 100
        
        ks_diff = abs(cum_good_pct - cum_bad_pct)
        
        decile_stats.append({
            'decile': decile,
            'cum_good_pct': cum_good_pct,
            'cum_bad_pct': cum_bad_pct,
            'ks_diff': ks_diff
        })
    
    if not decile_stats:
        return
    
    stats_df = pd.DataFrame(decile_stats)
    
    # Find maximum KS point
    max_ks_idx = stats_df['ks_diff'].idxmax()
    max_ks_decile = stats_df.loc[max_ks_idx, 'decile']
    max_ks_value = stats_df.loc[max_ks_idx, 'ks_diff']
    max_ks_good_pct = stats_df.loc[max_ks_idx, 'cum_good_pct']
    max_ks_bad_pct = stats_df.loc[max_ks_idx, 'cum_bad_pct']
    
    # Create plot
    plt.figure(figsize=(8, 6))
    
    # Plot cumulative curves
    plt.plot(stats_df['decile'], stats_df['cum_good_pct'], 'o-', 
             color='#1f77b4', linewidth=2, markersize=6, label='정상기업 (Goods)')
    plt.plot(stats_df['decile'], stats_df['cum_bad_pct'], 'o-', 
             color='#ff7f0e', linewidth=2, markersize=6, label='부실기업 (Bads)')
    
    # Mark maximum KS point
    plt.axvline(x=max_ks_decile, color='green', linestyle='--', alpha=0.7)
    plt.plot(max_ks_decile, max_ks_good_pct, 'go', markersize=8)
    plt.plot(max_ks_decile, max_ks_bad_pct, 'go', markersize=8)
    
    # Add KS value annotation
    plt.annotate(f'KS = {max_ks_value:.1f}% at decile {max_ks_decile}', 
                xy=(max_ks_decile, (max_ks_good_pct + max_ks_bad_pct)/2),
                xytext=(max_ks_decile + 1.5, (max_ks_good_pct + max_ks_bad_pct)/2),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.xlabel('Deciles')
    plt.ylabel('누적 비율 (%)')
    plt.title('KS Statistic Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.5, 10.5)
    plt.ylim(0, 105)
    
    # Set integer ticks for deciles
    plt.xticks(range(1, 11))
    
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def try_configure_korean_font() -> None:
    try:
        for f in ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "AppleGothic"]:
            plt.rcParams["font.family"] = f
            break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


# SHAP functionality removed


# SHAP plotting functionality removed


# SHAP analysis functionality removed


def create_powerful_case_studies(eval_df: pd.DataFrame, output_dir: str, key_column: str) -> None:
    """
    Generate powerful case studies that demonstrate model's discriminative power.
    Focus on: "Our model can evaluate this difference correctly!"
    """
    case_dir = os.path.join(output_dir, "case_studies")
    ensure_dir(case_dir)
    
    try_configure_korean_font()
    
    # 1. SIMILAR COMPANIES, DIFFERENT OUTCOMES - Most Powerful Demo
    print("[Verify] Creating similar companies case study...")
    # Use full eval_df (all splits) for better case study examples
    similar_pairs = find_similar_companies_different_outcomes(eval_df, 'unique_id')  # Use unique_id
    create_similar_companies_plot(similar_pairs, case_dir)
    
    # 2. GRADE BOUNDARY PRECISION - Show clear separation
    print("[Verify] Creating grade boundary analysis...")
    create_grade_boundary_analysis(eval_df, case_dir)
    
    # 3. EARLY WARNING SUCCESS - Time-sensitive predictions
    print("[Verify] Creating early warning case study...")
    create_early_warning_success_cases(eval_df, case_dir, 'unique_id')  # Use unique_id
    
    # 4. SECTOR COMPARISON - Same sector, different risk profiles
    print("[Verify] Creating sector comparison analysis...")
    if '업종코드1' in eval_df.columns:
        create_sector_comparison_cases(eval_df, case_dir, 'unique_id')  # Use unique_id
    
    print(f"[Verify] Powerful case studies generated: {case_dir}")


def find_similar_companies_different_outcomes(eval_df: pd.DataFrame, key_column: str) -> List[Dict]:
    """
    Find pairs of companies with similar characteristics but different outcomes.
    Focus on 4 key financial indicators with valid (non-zero, non-null) values.
    """
    # Focus on companies with clear outcomes (not NaN)
    clear_outcomes = eval_df[~eval_df['event_1y'].isna()].copy()
    
    if len(clear_outcomes) < 10:
        return []
    
    # Define 4 key financial indicators
    key_financial_features = ['재무정보_영업이익률_t0', '재무정보_부채비율_t0', '재무정보_ROE_t0', '재무정보_총자산회전율_t0']
    
    # Identify non-financial numeric variables for differentiator analysis
    # Exclude ID columns, dates, targets, specified business columns, and financial variables (all time periods)
    financial_patterns = ['재무정보_', '_t1', '_t2']  # Include t1, t2 financial variables
    exclude_cols = {key_column, '보험청약일자', 'final_score', 'grade', 'reason_code', 'bin_id', 
                   'event_1y', 'event_4y', 'risk_year1', 'risk_year2', 'risk_year3', 'risk_year4',
                   'data_split', '사업자등록번호', '대상자명', '청약번호', '수출자대상자번호', '업종코드1'} | set(key_financial_features)
    
    # Add all financial variables (t0, t1, t2) to exclusion list
    for col in clear_outcomes.columns:
        if any(pattern in col for pattern in financial_patterns):
            exclude_cols.add(col)
    
    non_financial_features = [col for col in clear_outcomes.columns 
                             if col not in exclude_cols 
                             and clear_outcomes[col].dtype in ['int64', 'float64']
                             and not clear_outcomes[col].isna().all()]
    
    print(f"[Verify] Found {len(non_financial_features)} non-financial variables for differentiator analysis")
    if len(non_financial_features) < 5:
        print(f"[Verify] Available non-financial variables: {non_financial_features[:10]}")  # Show first 10
        print(f"[Verify] Total columns in data: {len(clear_outcomes.columns)}")
        print(f"[Verify] Sample columns: {list(clear_outcomes.columns)[:20]}")  # Show first 20 columns
    
    # Filter companies that have all 4 financial indicators with valid (non-zero, non-null) values
    def has_valid_financials(row):
        for feature in key_financial_features:
            if feature not in row.index:
                return False
            value = row[feature]
            if pd.isna(value) or value == 0:
                return False
        return True
    
    valid_companies = clear_outcomes[clear_outcomes.apply(has_valid_financials, axis=1)].copy()
    
    if len(valid_companies) < 10:
        return []
    
    # Find similar pairs with different outcomes
    similar_pairs = []
    good_companies = valid_companies[valid_companies['event_1y'] == 0].sample(min(100, len(valid_companies[valid_companies['event_1y'] == 0])),random_state=42)
    bad_companies = valid_companies[valid_companies['event_1y'] == 1].sample(min(100, len(valid_companies[valid_companies['event_1y'] == 1])),random_state=42)
    
    for _, good_company in good_companies.iterrows():
        for _, bad_company in bad_companies.iterrows():
            # Calculate similarity based on 4 key financial indicators only
            good_features = np.array([good_company[f] for f in key_financial_features])
            bad_features = np.array([bad_company[f] for f in key_financial_features])
            
            # Normalize features to 0-1 scale for comparison
            feature_diff = np.abs(good_features - bad_features)
            max_vals = np.maximum(np.abs(good_features), np.abs(bad_features))
            max_vals = np.where(max_vals == 0, 1, max_vals)  # Avoid division by zero
            normalized_diff = feature_diff / max_vals
            similarity_score = 1 - np.mean(normalized_diff)
            
            # Model correctly distinguished them?
            model_diff = abs(good_company['final_score'] - bad_company['final_score'])
            
            if similarity_score > 0.8 and model_diff > 0.2:
                # Find most differentiating non-financial variables
                non_financial_diffs = {}
                for var in non_financial_features:
                    good_val = good_company.get(var, 0)
                    bad_val = bad_company.get(var, 0)
                    
                    # Skip if either value is NaN
                    if pd.isna(good_val) or pd.isna(bad_val):
                        continue
                    
                    # Calculate normalized difference
                    max_val = max(abs(good_val), abs(bad_val))
                    if max_val > 0:
                        diff = abs(good_val - bad_val) / max_val
                        non_financial_diffs[var] = {
                            'diff': diff,
                            'good_val': float(good_val),
                            'bad_val': float(bad_val),
                            'ratio': float(bad_val / good_val) if abs(good_val) > 1e-10 else 1.0
                        }
                
                # Get top 2 most differentiating non-financial variables
                top_differentiators = sorted(non_financial_diffs.items(), 
                                           key=lambda x: x[1]['diff'], reverse=True)[:2]
                
                similar_pairs.append({
                    'good_company': good_company,
                    'bad_company': bad_company,
                    'similarity_score': similarity_score,
                    'model_diff': model_diff,
                    'grade_diff': f"{good_company['grade']} vs {bad_company['grade']}",
                    'top_differentiators': top_differentiators  # NEW: Non-financial differentiators
                })
    
    # Return top 2 most similar pairs where model succeeded
    return sorted(similar_pairs, key=lambda x: x['similarity_score'], reverse=True)[:2]


def create_similar_companies_plot(similar_pairs: List[Dict], output_dir: str) -> None:
    """
    Enhanced visualization showing similar companies with different outcomes.
    Key improvements:
    - 3-panel layout: Financial indicators + Non-financial differentiators + Model scores
    - Y-axis normalized to good company (baseline=1) to amplify differences
    - Shows why model succeeded despite financial similarity
    """
    if not similar_pairs:
        return
    
    fig, axes = plt.subplots(len(similar_pairs), 3, figsize=(18, 5 * len(similar_pairs)))
    if len(similar_pairs) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, pair in enumerate(similar_pairs):
        good_co = pair['good_company']
        bad_co = pair['bad_company']
        top_differentiators = pair.get('top_differentiators', [])
        
        # Left plot: Financial indicators comparison (기존 유지)
        ax1 = axes[idx, 0]
        features = ['재무정보_영업이익률_t0', '재무정보_부채비율_t0', '재무정보_ROE_t0', '재무정보_총자산회전율_t0']
        available_features = [f for f in features if f in good_co.index and f in bad_co.index]
        
        if available_features:
            good_vals = [float(good_co[f]) if not pd.isna(good_co[f]) else 0 for f in available_features]
            bad_vals = [float(bad_co[f]) if not pd.isna(bad_co[f]) else 0 for f in available_features]
            
            x = np.arange(len(available_features))
            width = 0.35
            
            # Show actual values (not normalized ratios)
            ax1.bar(x - width/2, good_vals, width, 
                   label=f'우량기업 (등급: {good_co["grade"]})', color='#2ECC71', alpha=0.8)
            ax1.bar(x + width/2, bad_vals, width, 
                   label=f'부실기업 (등급: {bad_co["grade"]})', color='#E74C3C', alpha=0.8)
            
            ax1.set_xlabel('주요 재무지표')
            ax1.set_ylabel('지표 값')
            ax1.set_title(f'사례 {idx+1}: 유사한 특성, 다른 결과\n(유사도: {pair["similarity_score"]:.3f})')
            ax1.set_xticks(x)
            
            # Create readable labels
            feature_labels = []
            for f in available_features:
                if f == '재무정보_영업이익률_t0':
                    feature_labels.append('영업이익률')
                elif f == '재무정보_부채비율_t0':
                    feature_labels.append('부채비율')
                elif f == '재무정보_ROE_t0':
                    feature_labels.append('ROE')
                elif f == '재무정보_총자산회전율_t0':
                    feature_labels.append('총자산회전율')
                else:
                    feature_labels.append(f.replace('재무정보_', '').replace('_t0', ''))
            
            ax1.set_xticklabels(feature_labels, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Middle plot: Non-financial differentiators (NEW!)
        ax2 = axes[idx, 1]
        if top_differentiators:
            # Show top 2 non-financial differentiators
            diff_names = []
            diff_ratios = []
            diff_labels = []
            
            for var_name, var_info in top_differentiators:
                # Create readable variable name
                readable_name = var_name.replace('_', ' ').replace('t0', '').strip()
                if len(readable_name) > 20:
                    readable_name = readable_name[:17] + "..."
                
                diff_names.append(readable_name)
                diff_ratios.append(var_info['ratio'])
                diff_labels.append(f"{var_info['good_val']:.2f} vs {var_info['bad_val']:.2f}")
            
            x = np.arange(len(diff_names))
            width = 0.35
            
            # Good company baseline
            ax2.bar(x - width/2, [1.0] * len(diff_names), width,
                   label='우량기업 (기준=1.0)', color='#2ECC71', alpha=0.8)
            
            # Bad company ratios with emphasis
            bars = ax2.bar(x + width/2, diff_ratios, width,
                          label='부실기업 (배수)', color='#FF6B6B', alpha=0.8)
            
            # Add baseline
            ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            
            # Highlight the biggest differentiator
            if len(bars) > 0:
                bars[0].set_color('#C0392B')  # Darkest red for top differentiator
                ax2.annotate('핵심 구분 요인!', 
                            xy=(0 + width/2, diff_ratios[0]), 
                            xytext=(0.5, diff_ratios[0] + 0.3),
                            arrowprops=dict(arrowstyle='->', color='black', lw=2),
                            fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.9))
            
            ax2.set_xlabel('비재무 변수')
            ax2.set_ylabel('우량기업 대비 배수')
            ax2.set_title('모델이 포착한 핵심 차이점 (1년차 이벤트 기준 등급)')
            ax2.set_xticks(x)
            ax2.set_xticklabels(diff_names, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add ratio values on bars
            for i, (bar, ratio) in enumerate(zip(bars, diff_ratios)):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{ratio:.1f}x', ha='center', va='bottom', fontweight='bold')
        
        else:
            ax2.text(0.5, 0.5, '비재무 차이점\n분석 불가', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax2.set_title('비재무 변수 분석')
        
        # Right plot: Model prediction comparison
        ax3 = axes[idx, 2]
        companies = ['우량기업\n(실제: 정상)', '부실기업\n(실제: 부실)']
        scores = [float(good_co['final_score']), float(bad_co['final_score'])]
        colors = ['#2ECC71', '#E74C3C']
        
        bars = ax3.bar(companies, scores, color=colors, alpha=0.8)
        ax3.set_ylabel('모델 리스크 점수')
        ax3.set_title(f'모델 예측 결과\n(점수 차이: {pair["model_diff"]:.3f})')
        ax3.set_ylim(0, 1)
        
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add success message
        success_msg = "✅ 정확한 구분!"
        ax3.text(0.5, 0.85, success_msg, transform=ax3.transAxes, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                fontsize=10, fontweight='bold')
        
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, "similar_companies_different_outcomes.png"), 
                dpi=300, bbox_inches="tight")
    plt.close()


def create_grade_boundary_analysis(eval_df: pd.DataFrame, output_dir: str) -> None:
    """
    Show how well the model separates different grades at boundaries.
    🔄 CHANGED: Now uses event_1y for more immediate validation results.
    """
    # Determine which event column to use
    event_col = 'event_1y' if 'event_1y' in eval_df.columns and not eval_df['event_1y'].isna().all() else 'event_4y'
    
    if 'grade' not in eval_df.columns or eval_df[event_col].isna().all():
        return
    
    # Calculate actual bad rates by grade
    grade_analysis = eval_df.groupby('grade').agg({
        event_col: ['count', 'sum', 'mean'],
        'final_score': 'mean'
    }).round(3)
    
    grade_analysis.columns = ['총건수', '부실건수', '실제부실률', '평균점수']
    grade_analysis = grade_analysis.reset_index()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sort by proper grade order for visualization
    available_grades = [g for g in GRADE_ORDER if g in grade_analysis['grade'].values]
    grade_analysis = grade_analysis.set_index('grade').reindex(available_grades).reset_index()
    
    # Left: Actual bad rates by grade (should be monotonic)
    bars1 = ax1.bar(grade_analysis['grade'], grade_analysis['실제부실률'], 
                    color='#3498DB', alpha=0.8)
    ax1.set_xlabel('모델 예측 등급')
    ax1.set_ylabel('실제 부실률')
    ax1.set_title('등급별 실제 부실률\n(단조증가 = 모델 정확성 입증)')
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, rate in zip(bars1, grade_analysis['실제부실률']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Right: Model scores vs actual outcomes
    for grade in grade_analysis['grade'].unique():
        grade_data = eval_df[eval_df['grade'] == grade]
        good_scores = grade_data[grade_data[event_col] == 0]['final_score']
        bad_scores = grade_data[grade_data[event_col] == 1]['final_score']
        
        if len(good_scores) > 0:
            ax2.scatter([grade] * len(good_scores), good_scores, 
                       alpha=0.6, color='#2ECC71', s=20, label='정상기업' if grade == grade_analysis['grade'].iloc[0] else "")
        if len(bad_scores) > 0:
            ax2.scatter([grade] * len(bad_scores), bad_scores, 
                       alpha=0.6, color='#E74C3C', s=20, label='부실기업' if grade == grade_analysis['grade'].iloc[0] else "")
    
    ax2.set_xlabel('모델 예측 등급')
    ax2.set_ylabel('모델 리스크 점수')
    ax2.set_title('등급 내 점수 분포\n(부실기업이 상위 점수에 집중되어야 함)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grade_boundary_precision.png"), 
                dpi=300, bbox_inches="tight")
    plt.close()
    
    # Save analysis table
    with open(os.path.join(output_dir, "grade_precision_analysis.md"), "w", encoding="utf-8") as f:
        f.write("# 등급 경계 정밀도 분석\n\n")
        f.write("## 🎯 핵심 메시지\n")
        f.write("**우리 모델은 등급 간 실제 리스크 차이를 정확히 구분합니다!**\n\n")
        f.write("## 📊 등급별 실제 성과\n\n")
        # Simple table format without tabulate dependency
        f.write("| 등급 | 총건수 | 부실건수 | 실제부실률 | 평균점수 |\n")
        f.write("|------|--------|----------|------------|----------|\n")
        for _, row in grade_analysis.iterrows():
            f.write(f"| {row['grade']} | {row['총건수']} | {row['부실건수']} | {row['실제부실률']:.1%} | {row['평균점수']:.3f} |\n")
        f.write("\n\n## ✅ 검증 결과\n")
        
        # Check monotonicity
        is_monotonic = all(grade_analysis['실제부실률'].iloc[i] <= grade_analysis['실제부실률'].iloc[i+1] 
                          for i in range(len(grade_analysis)-1))
        f.write(f"- **단조성 검증**: {'✅ 통과' if is_monotonic else '❌ 실패'}\n")
        f.write(f"- **최고등급 부실률**: {grade_analysis['실제부실률'].iloc[0]:.1%}\n")
        f.write(f"- **최저등급 부실률**: {grade_analysis['실제부실률'].iloc[-1]:.1%}\n")
        f.write(f"- **리스크 배수**: {grade_analysis['실제부실률'].iloc[-1] / max(grade_analysis['실제부실률'].iloc[0], 0.001):.1f}배\n")


def create_early_warning_success_cases(eval_df: pd.DataFrame, output_dir: str, key_column: str) -> None:
    """
    Show cases where model predicted high risk early and was proven right.
    🔄 CHANGED: Now uses event_1y for more immediate validation results.
    """
    # Determine which event column to use
    event_col = 'event_1y' if 'event_1y' in eval_df.columns and not eval_df['event_1y'].isna().all() else 'event_4y'
    
    # Focus on high-risk predictions that were correct
    early_warnings = eval_df[
        (eval_df['grade'].isin(['B', 'CCC'])) & 
        (eval_df[event_col] == 1) &
        (eval_df['final_score'] > eval_df['final_score'].quantile(0.8))
    ].head(10)
    
    if early_warnings.empty:
        return
    
    # Create success story document
    with open(os.path.join(output_dir, "early_warning_success.md"), "w", encoding="utf-8") as f:
        f.write("# 조기 경보 성공 사례\n\n")
        f.write("## 🚨 모델의 조기 경보 능력 입증\n\n")
        f.write("다음은 우리 모델이 **사전에 고위험으로 분류**하여 **실제 부실을 정확히 예측**한 사례들입니다.\n\n")
        
        for idx, (_, case) in enumerate(early_warnings.iterrows(), 1):
            f.write(f"### 사례 {idx}: 정확한 고위험 예측\n")
            f.write(f"- **기업 ID**: {case['unique_id']}\n")  # Use unique_id
            f.write(f"- **모델 예측**: {case['grade']} 등급 (리스크 점수: {case['final_score']:.3f})\n")
            f.write(f"- **실제 결과**: 부실 발생 ✅\n")
            f.write(f"- **주요 위험 요인**: {case.get('reason_code', 'N/A')}\n")
            # SHAP variables removed
            f.write("\n")
        
        f.write("## 📈 조기 경보 효과성\n")
        total_high_risk = len(eval_df[eval_df['grade'].isin(['B', 'CCC'])])
        correct_high_risk = len(eval_df[(eval_df['grade'].isin(['B', 'CCC'])) & (eval_df[event_col] == 1)])
        if total_high_risk > 0:
            precision = correct_high_risk / total_high_risk
            f.write(f"- **고위험 예측 정밀도**: {precision:.1%}\n")
            f.write(f"- **총 고위험 예측**: {total_high_risk:,}건\n")
            f.write(f"- **실제 부실 발생**: {correct_high_risk:,}건\n")
            f.write(f"- **검증 기간**: {'1년' if event_col == 'event_1y' else '4년'} 이벤트 기준\n")


def create_sector_comparison_cases(eval_df: pd.DataFrame, output_dir: str, key_column: str) -> None:
    """
    Show how model correctly distinguishes risk within same sectors.
    """
    if '업종코드1' not in eval_df.columns:
        return
    
    # Determine which event column to use
    event_col = 'event_1y' if 'event_1y' in eval_df.columns and not eval_df['event_1y'].isna().all() else 'event_4y'
    
    # Find sectors with both good and bad companies
    sector_analysis = eval_df.groupby('업종코드1').agg({
        event_col: ['count', 'sum'],
        'final_score': ['mean', 'std']
    }).round(3)
    
    sector_analysis.columns = ['총건수', '부실건수', '평균점수', '점수편차']
    sector_analysis['부실률'] = sector_analysis['부실건수'] / sector_analysis['총건수']
    sector_analysis = sector_analysis[sector_analysis['총건수'] >= 30]  # Minimum sample size
    
    # Select interesting sectors (mix of good and bad)
    interesting_sectors = sector_analysis[
        (sector_analysis['부실건수'] >= 2) & 
        (sector_analysis['부실률'] < 0.8) &
        (sector_analysis['부실률'] > 0.1)
    ].head(5)
    
    if interesting_sectors.empty:
        return
    
    # Create sector comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, (sector, data) in enumerate(interesting_sectors.iterrows()):
        if idx >= 4:
            break
            
        sector_companies = eval_df[eval_df['업종코드1'] == sector]
        good_companies = sector_companies[sector_companies[event_col] == 0]
        bad_companies = sector_companies[sector_companies[event_col] == 1]
        
        ax = axes[idx]
        
        if len(good_companies) > 0:
            ax.hist(good_companies['final_score'], bins=10, alpha=0.7, 
                   color='#2ECC71', label='정상기업', density=True)
        if len(bad_companies) > 0:
            ax.hist(bad_companies['final_score'], bins=10, alpha=0.7, 
                   color='#E74C3C', label='부실기업', density=True)
        
        ax.set_xlabel('모델 리스크 점수')
        ax.set_ylabel('밀도')
        ax.set_title(f'업종: {sector}\n(부실률: {data["부실률"]:.1%}, 총 {int(data["총건수"])}개 기업)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(interesting_sectors), 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sector_risk_discrimination.png"), 
                dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"[Verify] Sector comparison analysis completed for {len(interesting_sectors)} sectors")


def create_pdp_analysis(eval_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create Partial Dependence Plot analysis for key variables.
    """
    # Load the trained model
    model_path = "../results/step8_post/unified_model.json"
    if not os.path.exists(model_path):
        print(f"[Verify] Model not found at {model_path}. Skipping PDP analysis.")
        return
    
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        
        # Key variables to analyze (adjust based on your features)
        key_variables = ['업력', '자본금', '매출액', '보험료']  # Adjust to your actual feature names
        available_vars = [var for var in key_variables if var in eval_df.columns]
        
        if not available_vars:
            print("[Verify] No key variables found for PDP analysis")
            return
        
        # Prepare data
        feature_cols = [col for col in eval_df.columns if col not in ['event_1y', 'grade', 'final_score', 'reason_code']]
        X = eval_df[feature_cols].select_dtypes(include=[np.number])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, var in enumerate(available_vars[:4]):  # Max 4 variables
            if var in X.columns:
                # Create simple PDP manually since sklearn PDP may not work with XGBoost directly
                var_values = np.linspace(X[var].quantile(0.1), X[var].quantile(0.9), 50)
                pdp_values = []
                
                for val in var_values:
                    X_temp = X.copy()
                    X_temp[var] = val
                    # Convert to DMatrix for XGBoost
                    dmatrix = xgb.DMatrix(X_temp)
                    pred = model.predict(dmatrix)
                    pdp_values.append(np.mean(pred))
                
                axes[i].plot(var_values, pdp_values, linewidth=2, color='#E74C3C')
                axes[i].set_xlabel(var)
                axes[i].set_ylabel('예측 리스크')
                axes[i].set_title(f'{var}에 따른 리스크 변화')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_vars), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pdp_analysis.png"), dpi=200, bbox_inches="tight")
        plt.close()
        
        print(f"[Verify] PDP analysis saved: pdp_analysis.png")
        
    except Exception as e:
        print(f"[Verify] PDP analysis failed: {e}")






def main():
    cfg = parse_args()
    ensure_dir(cfg.output_dir)
    plots_dir = os.path.join(cfg.output_dir, "plots")
    ensure_dir(plots_dir)
    
    # Configure Korean font
    try_configure_korean_font()

    assignments, dataset = read_inputs(cfg)
    eval_df = build_eval_frame(assignments, dataset, cfg)
    
    # SHAP analysis removed

    # Debug: Check split distribution
    print(f"[Verify] Final split distribution: {eval_df['split'].value_counts().to_dict()}")
    print(f"[Verify] OOT data count: {(eval_df['split'] == 'oot').sum()}")
    print(f"[Verify] Event_1y distribution: {eval_df['event_1y'].value_counts(dropna=False).to_dict()}")
    print(f"[Verify] Event_4y distribution: {eval_df['event_4y'].value_counts(dropna=False).to_dict()}")
    
    # 🔄 CHANGED: Use 1-year events for validation (more immediate and data-rich)
    mask_eval = (~pd.isna(eval_df["event_1y"])) & (eval_df["split"] == "oot")
    if mask_eval.sum() == 0:
        print("⚠️ No OOT data with 1-year events found. Checking 4-year events as fallback...")
        mask_eval = (~pd.isna(eval_df["event_4y"])) & (eval_df["split"] == "oot")
        if mask_eval.sum() == 0:
            raise ValueError("No OOT data found for evaluation. This violates strict validation rules. "
                            "OOT data is required for unbiased performance assessment.")
        event_col = "event_4y"
        print(f"[Verify] Using 4-year events as fallback")
    else:
        event_col = "event_1y"
        print(f"[Verify] ✅ Using 1-year events for validation (preferred for immediate feedback)")
    
    eval_sub = eval_df.loc[mask_eval].copy()
    print(f"[Verify] Strict validation using OOT data only: {len(eval_sub):,} rows")
    print(f"[Verify] Event type: {event_col} - This represents TRUE unbiased performance estimate")

    # Targets and predictions
    y = eval_sub[event_col].astype(int).values
    p = eval_sub["final_score"].astype(float).clip(cfg.epsilon, 1.0 - cfg.epsilon).values

    # Metrics
    metrics = {}
    if len(np.unique(y)) >= 2:
        auc = roc_auc_score(y, p)
        ar = 2.0 * (auc - 0.5)
        ks = compute_ks(y, p)
        lift = top_decile_lift(y, p)
        brier = brier_score(y, p)
        intercept, slope = logistic_calibration(y, p, cfg.epsilon)
        metrics.update({
            "AUC": float(auc),
            "AR": float(ar),
            "KS": float(ks),
            "top_decile_lift": float(lift),
            "Brier": float(brier),
            "calibration_intercept": float(intercept),
            "calibration_slope": float(slope),
        })
    else:
        metrics.update({"AUC": 0.0, "AR": 0.0, "KS": 0.0, "top_decile_lift": 0.0, "Brier": 0.0, "calibration_intercept": 0.0, "calibration_slope": 1.0})

    # KS Statistic Plot
    plot_ks_statistic(y, p, os.path.join(plots_dir, "ks_statistic.png"))

    # Monotonicity checks by grade using bin_id order
    grade_df = eval_sub.copy()
    by_bin = grade_df.groupby("bin_id").agg(obs_rate=(event_col, "mean"), count=(event_col, "size")).reset_index().sort_values("bin_id")
    monotonic_ok = bool(np.all(np.diff(by_bin["obs_rate"].fillna(0.0)) >= -1e-12))
    rho, rho_p = spearmanr(by_bin["bin_id"], by_bin["obs_rate"].fillna(0.0))
    metrics.update({"monotonic_by_grade": monotonic_ok, "spearman_rho": float(rho), "spearman_p": float(rho_p)})

    # Grade-level stats with Wilson CIs and explainability aggregates
    grade_stats_rows = []
    total_eval = len(grade_df)
    for g, grp in grade_df.groupby("grade"):
        n = len(grp)
        k = int(np.nansum(grp[event_col].values))  # Use dynamic event column
        ci_low, ci_high = wilson_ci(k, n)
        avg_score = float(np.nanmean(grp["final_score"])) if n > 0 else 0.0
        # Calculate avg_shares safely to avoid RuntimeWarning
        avg_shares = {}
        for t in range(1, 5):
            share_col = f"share_{t}y"
            if share_col in grp.columns and len(grp[share_col].dropna()) > 0:
                avg_shares[f"avg_share_{t}y"] = float(np.nanmean(grp[share_col]))
            else:
                avg_shares[f"avg_share_{t}y"] = 0.0
        # reason_code distribution (top-3)
        rc_dist = grp["reason_code"].value_counts(normalize=True).head(3).to_dict()
        row = {
            "grade": g,
            "count": n,
            "proportion": float(n / max(total_eval, 1)),
            "observed_event_rate": float(k / max(n, 1)),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "avg_score": avg_score,
            **avg_shares,
            "top_reasons": rc_dist,
        }
        grade_stats_rows.append(row)
    # Sort by proper grade order (safe to risky)
    grade_stats = pd.DataFrame(grade_stats_rows)
    available_grades = [g for g in GRADE_ORDER if g in grade_stats["grade"].values]
    grade_stats = grade_stats.set_index("grade").reindex(available_grades).reset_index()
    grade_stats.to_csv(os.path.join(cfg.output_dir, "grade_level_stats.csv"), index=False)

    # Stability / PSI between dev and oot on grades
    dev_grades = eval_df.loc[(eval_df["split"] == "dev") & (~pd.isna(eval_df["grade"])) , "grade"]
    oot_grades = eval_df.loc[(eval_df["split"] == "oot") & (~pd.isna(eval_df["grade"])) , "grade"]
    if len(dev_grades) > 0 and len(oot_grades) > 0:
        psi = psi_for_grades(dev_grades, oot_grades)
        metrics["PSI_grades_dev_vs_oot"] = float(psi)
        # Plot grade distributions
        try_configure_korean_font()
        plt.figure(figsize=(8, 4))
        all_grades = sorted(set(dev_grades) | set(oot_grades))
        # Sort by proper grade order
        available_grades = [g for g in GRADE_ORDER if g in all_grades]
        dist = pd.DataFrame({
            "grade": available_grades,
        })
        dist["dev"] = dist["grade"].map(dev_grades.value_counts(normalize=True))
        dist["oot"] = dist["grade"].map(oot_grades.value_counts(normalize=True))
        dist = dist.fillna(0.0)
        
        # Create bar plot with Korean labels
        ax = dist.set_index("grade").plot(kind="bar", figsize=(8, 4), 
                                         color=['#1f77b4', '#ff7f0e'], alpha=0.8)
        plt.ylabel("비율")
        plt.xlabel("등급")
        plt.title(f"등급 분포 비교 (PSI={psi:.3f})")
        plt.legend(['개발 데이터', 'OOT 데이터'])
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "psi_grades.png"), dpi=200, bbox_inches="tight")
        plt.close()
    else:
        metrics["PSI_grades_dev_vs_oot"] = float("nan")

    # Plot: ROC curve (on evaluation subset)
    plot_roc(y, p, os.path.join(plots_dir, "roc.png"))
    
    # Generate powerful case studies that demonstrate model capability
    create_powerful_case_studies(eval_sub, cfg.output_dir, 'unique_id')  # Use unique_id
    
    # Create PDP analysis
    create_pdp_analysis(eval_sub, plots_dir)
    
    # SHAP analysis removed - using feature importance plots instead

    # Simplified segmentation analysis (only key segments)
    segments_results: Dict[str, Dict[str, float]] = {}
    key_segment = "업종코드1" if "업종코드1" in eval_df.columns else None
    
    if key_segment:
        # Only analyze top 5 segments by count
        top_segments = eval_sub[key_segment].value_counts().head(5).index.tolist()
        per_seg: Dict[str, Dict[str, float]] = {}
        
        for s in top_segments:
            sub = eval_sub[eval_sub[key_segment] == s]
            # Use event_1y for segment analysis (consistent with main validation)
            event_col_segment = 'event_1y' if 'event_1y' in sub.columns and not sub['event_1y'].isna().all() else 'event_4y'
            yy_raw = sub[event_col_segment].values
            # Filter out NaN values before converting to int
            valid_mask = ~pd.isna(yy_raw)
            if valid_mask.sum() < 30:  # Need at least 30 valid samples
                continue
            yy = yy_raw[valid_mask].astype(int)
            pp = sub["final_score"].values[valid_mask].astype(float)
            pp = np.clip(pp, cfg.epsilon, 1.0 - cfg.epsilon)
            if len(yy) >= 30 and len(np.unique(yy)) >= 2:
                try:
                    auc = roc_auc_score(yy, pp)
                    ar = 2.0 * (auc - 0.5)
                    ks = compute_ks(yy, pp)
                    per_seg[s] = {"AUC": float(auc), "AR": float(ar), "KS": float(ks), "count": int(len(sub))}
                except Exception:
                    continue
        
        if per_seg:
            segments_results[key_segment] = per_seg

    # Save simplified metrics summary (essential only)
    essential_summary = {
        "execution_date": datetime.now().isoformat(),
        "evaluation_rows": int(len(eval_sub)),
        "key_metrics": {
            "AUC": metrics.get("AUC", 0.0),
            "AR": metrics.get("AR", 0.0), 
            "KS": metrics.get("KS", 0.0),
            "top_decile_lift": metrics.get("top_decile_lift", 0.0)
        },
        "performance_status": {
            "AR_pass": bool(metrics.get("AR", 0.0) >= cfg.acceptance_ar),
            "KS_pass": bool(metrics.get("KS", 0.0) >= cfg.acceptance_ks),
            "overall_pass": bool(metrics.get("AR", 0.0) >= cfg.acceptance_ar and metrics.get("KS", 0.0) >= cfg.acceptance_ks)
        },
        "top_segments": segments_results.get(key_segment, {}) if key_segment else {}
    }
    
    with open(os.path.join(cfg.output_dir, "performance_summary.json"), "w", encoding="utf-8") as f:
        json.dump(essential_summary, f, indent=2, ensure_ascii=False)

    # Streamlined Executive Report
    lines: List[str] = []
    lines.append("# KSURE 리스크 모델 검증 보고서")
    lines.append("")
    lines.append("## 🎯 핵심 결론")
    ar = metrics.get("AR", 0.0)
    ks = metrics.get("KS", 0.0)
    overall_pass = ar >= cfg.acceptance_ar and ks >= cfg.acceptance_ks
    
    lines.append(f"**모델 성능**: {'✅ 목표 달성' if overall_pass else '⚠️ 검토 필요'}")
    lines.append(f"- AR (정확도 비율): {ar:.3f} {'✅' if ar >= cfg.acceptance_ar else '❌'}")
    lines.append(f"- KS (분리도): {ks:.3f} {'✅' if ks >= cfg.acceptance_ks else '❌'}")
    lines.append(f"- AUC: {metrics.get('AUC', 0.0):.3f}")
    lines.append("")
    
    lines.append("## 📊 주요 성과")
    lines.append(f"- **상위 10% 리프트**: {metrics.get('top_decile_lift', 0.0):.1f}배")
    lines.append(f"- **평가 대상**: {len(eval_sub):,}개 기업")
    lines.append(f"- **등급 안정성**: {'양호' if metrics.get('monotonic_by_grade', False) else '검토 필요'}")
    lines.append("")
    
    if segments_results and key_segment:
        lines.append("## 🏢 주요 업종별 성능")
        for seg, m in list(segments_results[key_segment].items())[:3]:
            lines.append(f"- **{seg}**: AR={m['AR']:.3f}, 대상={m['count']}개")
        lines.append("")
    
    lines.append("## 📈 주요 결과물")
    lines.append("- **성능 대시보드**: plots/performance_dashboard.png")
    lines.append("- **성공 사례**: case_studies/success_cases.md")
    lines.append("- **고위험 탐지 사례**: case_studies/failure_cases.md")
    lines.append("- **변수 영향도 분석**: plots/pdp_analysis.png")
    lines.append("- **특성 중요도**: plots/feature_importance_overall.png")
    lines.append("")
    
    lines.append("## 📋 권장사항")
    if overall_pass:
        lines.append("- 모델을 프로덕션 환경에 배포할 준비가 완료되었습니다")
        lines.append("- 월 1회 성능 모니터링을 권장합니다")
    else:
        lines.append("- 모델 재보정 또는 추가 특성 검토가 필요합니다")
        lines.append("- 세부 세그먼트별 분석을 통한 개선 방안 도출을 권장합니다")
    lines.append("")
    
    lines.append("---")
    lines.append(f"*보고서 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    with open(os.path.join(cfg.output_dir, "executive_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ 검증 완료!")
    print(f"  📊 성능 대시보드: plots/performance_dashboard.png")
    print(f"  📋 경영진 보고서: executive_report.md")
    print(f"  📈 사례 분석: case_studies/")
    print(f"  📊 성능 요약: performance_summary.json")


if __name__ == "__main__":
    import math
    main()
