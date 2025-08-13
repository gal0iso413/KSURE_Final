"""
Final Grading Engine (v1)
=================================

Purpose
- Convert four annual ordinal predictions (class probabilities for years 1..4) into a single continuous 4-year risk score
  using a survival-consistent, hazard-weighted aggregation, then map to discrete grades via supervised, monotonic binning.

Design Highlights
- Per-year ordinal-to-event calibration using empirical event rates r_{t,k} with isotonic regression over classes to enforce monotonicity
- Hazard transform h_t = -ln(1 - p_any_t) with epsilon clipping for numeric safety; final_score = 1 - exp(-Σ w_t h_t)
- Monotonic supervised binning without external deps: quantile pre-bins + Pool-Adjacent-Violators merging + reduction to target grade count
- Explainability: per-year contributions and shares; business-readable reason_code with combined reasons if two top shares are close
- Governance-ready outputs: frozen cutpoints, calibration tables, config snapshot

Inputs (defaults)
- dataset_path: dataset/credit_risk_dataset_step4.csv (must contain risk_year{1..4}, keys for join like 청약번호, 보험청약일자)
- predictions_path: result/predictions/yearly_multiclass_proba.csv (requires proba_y{t}_{0..3} for t=1..4 plus join keys)

Outputs
- result/step10_grading/4_grade_assignments.csv
- result/step10_grading/grade_bins.json
- result/step10_grading/calibration_tables.csv
- result/step10_grading/config_used.json
- result/step10_grading/log.txt

Notes
- This script does not train upstream models; it assumes predictions are already generated.
- Weight optimization is optional and OFF by default to avoid overfitting; you can pass --weights or enable simple grid search.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Configuration dataclasses
# -----------------------------


@dataclass
class GradingConfig:
    dataset_path: str = "dataset/credit_risk_dataset_step4.csv"
    predictions_path: str = "result/predictions/yearly_multiclass_proba.csv"
    output_dir: str = "result/step10_grading"
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
    parser.add_argument("--dataset_path", type=str, default="dataset/credit_risk_dataset_step4.csv")
    parser.add_argument("--predictions_path", type=str, default="result/predictions/yearly_multiclass_proba.csv")
    parser.add_argument("--output_dir", type=str, default="result/step10_grading")
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
    if cfg.key_column not in preds.columns:
        raise ValueError(f"Predictions file missing key column: {cfg.key_column}")


def join_dataset_predictions(df: pd.DataFrame, preds: pd.DataFrame, cfg: GradingConfig) -> pd.DataFrame:
    if cfg.key_column not in df.columns:
        raise ValueError(f"Dataset missing key column: {cfg.key_column}")
    merged = pd.merge(df, preds, on=cfg.key_column, how="inner", suffixes=("", "_pred"))
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
    # Order grades by provided labels
    order = [g for g in labels if g in assignments["grade"].unique().tolist()]
    dist = assignments["grade"].value_counts().reindex(order, fill_value=0)
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
    Overfitting guard: prefer stable weights via simple penalty on std across K folds (temporal blocks by date order if available).
    Default kept OFF; use --auto_weight_selection to enable.
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
            y = (sl["risk_year4"].astype(float) >= 1).astype(int).values
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

    # Read
    df, preds = read_data(cfg)
    validate_predictions(preds, cfg)
    log_lines.append(f"Dataset rows: {len(df):,}; Predictions rows: {len(preds):,}")

    # Join and prepare
    merged = join_dataset_predictions(df, preds, cfg)
    merged = compute_event_flags(merged)
    merged = compute_predicted_classes(merged)

    # Calibration tables
    calib_df, r_tk = compute_yearly_calibration_tables(merged, cfg)
    log_lines.append("Per-year class-to-event calibration completed with isotonic monotonicity.")

    # p_any per year
    merged = compute_p_any(merged, r_tk, cfg.epsilon)

    # Base hazards using current weights for potential CV selection reuse
    merged = compute_hazard_and_score(merged, cfg.weights, cfg.epsilon)

    # Optional: simple CV-based weight selection on development subset with labels
    if cfg.auto_weight_selection:
        dev_mask = ~pd.isna(merged["risk_year4"])  # evaluate vs 4-year event
        dev_df = merged.loc[dev_mask].copy()
        if len(dev_df) > 0:
            new_w = select_weights_by_cv(dev_df, cfg)
            log_lines.append(f"Auto-selected weights from candidates: old={cfg.weights} → new={new_w}")
            cfg.weights = new_w
            # recompute final_score and shares with new weights
            merged = compute_hazard_and_score(merged, cfg.weights, cfg.epsilon)
        else:
            log_lines.append("Auto weight selection skipped (no labeled development rows).")

    # Build supervised monotonic bins on development rows with 4y labels
    dev_mask = ~pd.isna(merged["risk_year4"])  # only where label matured
    dev_rows = merged.loc[dev_mask].copy()
    y_dev = (dev_rows["risk_year4"].astype(float) >= 1).astype(int).values
    scores_dev = dev_rows["final_score"].astype(float).values
    final_bins_df, cutpoints = monotone_supervised_binning(scores_dev, y_dev, cfg)
    log_lines.append(f"Monotonic supervised binning produced {len(cutpoints)+1} bins (target={cfg.target_num_grades}).")

    # Grades assignment to all rows
    labels = cfg.grade_labels or default_grade_labels(len(cutpoints) + 1)
    grades, bin_idx = assign_grade_from_cutpoints(merged["final_score"].astype(float).values, cutpoints, labels)
    merged["bin_id"] = bin_idx.astype(int)
    merged["grade"] = grades

    # Reason codes
    merged["reason_code"] = merged.apply(lambda r: reason_code_from_shares(r, cfg.reason_merge_threshold), axis=1)

    # Persist outputs
    # 1) Assignments
    assign_cols = [cfg.key_column]
    if cfg.date_column in merged.columns:
        assign_cols.append(cfg.date_column)
    for t in range(1, 5):
        assign_cols += [f"p_any_{t}y", f"h_{t}y", f"contrib_{t}y", f"share_{t}y"]
    assign_cols += ["final_score", "bin_id", "grade", "reason_code"]
    assignments = merged[assign_cols].copy()
    assignments_path = os.path.join(cfg.output_dir, "4_grade_assignments.csv")
    assignments.to_csv(assignments_path, index=False)
    # Plot grade distribution
    try:
        plot_grade_distribution(assignments, labels, cfg.output_dir)
    except Exception:
        pass

    # 2) Calibration tables
    calib_path = os.path.join(cfg.output_dir, "calibration_tables.csv")
    calib_df.to_csv(calib_path, index=False)

    # 3) Grade bins metadata
    bins_meta = {
        "execution_date": datetime.now().isoformat(),
        "cutpoints": cutpoints,  # ascending right edges
        "num_bins": len(cutpoints) + 1,
        "labels": labels,
        "final_bins_table": final_bins_df.to_dict(orient="records"),
    }
    with open(os.path.join(cfg.output_dir, "grade_bins.json"), "w", encoding="utf-8") as f:
        json.dump(bins_meta, f, indent=2, ensure_ascii=False)

    # 4) Config snapshot
    cfg_meta = asdict(cfg)
    with open(os.path.join(cfg.output_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_meta, f, indent=2, ensure_ascii=False)

    # 5) Log
    log_lines.insert(0, f"KSURE Grading Engine v1 | {datetime.now().isoformat()}")
    with open(os.path.join(cfg.output_dir, "log.txt"), "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    print(f"✅ Grading completed: {assignments_path}")
    print(f"   Calibration tables: {calib_path}")
    print(f"   Bins meta: {os.path.join(cfg.output_dir, 'grade_bins.json')}")
    print(f"   Config snapshot: {os.path.join(cfg.output_dir, 'config_used.json')}")


if __name__ == "__main__":
    main()