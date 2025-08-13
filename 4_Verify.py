"""
Final Grade Validation (v1)
=================================

Validates the graded outputs from 4_Grade.py using objective metrics and produces
an audit-ready report with plots and summary tables.

Core checks
- Discrimination: AUC, AR, KS, top-decile lift
- Calibration: reliability (by score deciles and by grade), calibration intercept/slope, Brier score
- Monotonicity & separability: observed bad rates by grade with Wilson CIs, Spearman rank
- Stability & drift: PSI between development and out-of-time splits
- Segmentation: metrics by selected segments (e.g., 업종코드1)
- Reporting: metrics_summary.json, grade_level_stats.csv, plots, validation_report.md with Executive Summary

Inputs
- Assignments: result/step10_grading/4_grade_assignments.csv
- Dataset: dataset/credit_risk_dataset_step4.csv (to derive event_4y and segments)

Outputs (default)
- result/step10_grading_validation/metrics_summary.json
- result/step10_grading_validation/grade_level_stats.csv
- result/step10_grading_validation/plots/* (roc.png, cap.png, reliability_*.png, psi_grades.png, migration_heatmap.png [optional])
- result/step10_grading_validation/validation_report.md
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class VerifyConfig:
    assignments_path: str = "result/step10_grading/4_grade_assignments.csv"
    dataset_path: str = "dataset/credit_risk_dataset_step4.csv"
    output_dir: str = "result/step10_grading_validation"
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args() -> VerifyConfig:
    p = argparse.ArgumentParser(description="KSURE Grade Validation (v1)")
    p.add_argument("--assignments_path", type=str, default="result/step10_grading/4_grade_assignments.csv")
    p.add_argument("--dataset_path", type=str, default="dataset/credit_risk_dataset_step4.csv")
    p.add_argument("--output_dir", type=str, default="result/step10_grading_validation")
    p.add_argument("--date_column", type=str, default="보험청약일자")
    p.add_argument("--key_column", type=str, default="청약번호")
    p.add_argument("--dev_start", type=str, default=None)
    p.add_argument("--dev_end", type=str, default=None)
    p.add_argument("--oot_start", type=str, default=None)
    p.add_argument("--oot_end", type=str, default=None)
    p.add_argument("--segment_cols", type=str, default=None, help="Comma-separated segment columns")
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--previous_assignments_path", type=str, default=None)
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
    )


def read_inputs(cfg: VerifyConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assign = pd.read_csv(cfg.assignments_path)
    df = pd.read_csv(cfg.dataset_path)
    if cfg.date_column in assign.columns:
        try:
            assign[cfg.date_column] = pd.to_datetime(assign[cfg.date_column])
        except Exception:
            pass
    if cfg.date_column in df.columns:
        try:
            df[cfg.date_column] = pd.to_datetime(df[cfg.date_column])
        except Exception:
            pass
    return assign, df


def build_eval_frame(assign: pd.DataFrame, df: pd.DataFrame, cfg: VerifyConfig) -> pd.DataFrame:
    if cfg.key_column not in assign.columns:
        raise ValueError(f"Assignments missing key column: {cfg.key_column}")
    if cfg.key_column not in df.columns:
        raise ValueError(f"Dataset missing key column: {cfg.key_column}")

    merged = pd.merge(assign, df[[cfg.key_column, cfg.date_column, "risk_year4"] +
                                 (["업종코드1"] if "업종코드1" in df.columns else [])],
                      on=cfg.key_column, how="left")
    merged["event_4y"] = np.where(~pd.isna(merged["risk_year4"]) & (merged["risk_year4"] >= 1), 1, np.where(~pd.isna(merged["risk_year4"]), 0, np.nan))

    # Split into dev and oot
    merged["split"] = "unspecified"
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
    fpr, tpr, _ = roc_curve(y, score)
    auc = roc_auc_score(y, score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_cap(y: np.ndarray, score: np.ndarray, path: str) -> None:
    n = len(y)
    if n == 0:
        return
    order = np.argsort(-score)
    y_sorted = y[order]
    cum_events = np.cumsum(y_sorted)
    total_events = max(int(y.sum()), 1)
    x = np.arange(1, n + 1) / n
    y_cap = cum_events / total_events
    plt.figure(figsize=(6, 5))
    plt.plot(x, y_cap, label="Model CAP")
    plt.plot(x, x, linestyle="--", color="gray", label="Random")
    # Perfect model line
    perfect_x = np.concatenate([
        np.linspace(0, total_events / n, 100, endpoint=True),
        np.linspace(total_events / n, 1, 100)
    ])
    perfect_y = np.concatenate([
        np.linspace(0, 1, 100, endpoint=True),
        np.ones(100)
    ])
    plt.plot(perfect_x, perfect_y, linestyle=":", color="green", label="Perfect")
    plt.xlabel("Population proportion")
    plt.ylabel("Captured event proportion")
    plt.title("CAP Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_reliability(df_rel: pd.DataFrame, path: str, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.plot(df_rel["pred_mean"], df_rel["obs_rate"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed rate")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    cfg = parse_args()
    ensure_dir(cfg.output_dir)
    plots_dir = os.path.join(cfg.output_dir, "plots")
    ensure_dir(plots_dir)

    assignments, dataset = read_inputs(cfg)
    eval_df = build_eval_frame(assignments, dataset, cfg)

    # Evaluation subset (OOT preferred)
    mask_eval = (~pd.isna(eval_df["event_4y"])) & (eval_df["split"] == "oot")
    if mask_eval.sum() == 0:
        # fallback to dev if oot not available
        mask_eval = (~pd.isna(eval_df["event_4y"])) & (eval_df["split"] == "dev")
    eval_sub = eval_df.loc[mask_eval].copy()

    # Targets and predictions
    y = eval_sub["event_4y"].astype(int).values
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

    # Reliability curves
    rel_dec = reliability_by_bins(y, p, num_bins=10)
    plot_reliability(rel_dec, os.path.join(plots_dir, "reliability_deciles.png"), "Reliability (score deciles)")

    # Reliability by grade
    grade_df = eval_sub.copy()
    rel_grade = grade_df.groupby("grade").agg(pred_mean=("final_score", "mean"), obs_rate=("event_4y", "mean"), count=("event_4y", "size")).reset_index()
    plot_reliability(rel_grade.rename(columns={"pred_mean": "pred_mean", "obs_rate": "obs_rate"}), os.path.join(plots_dir, "reliability_by_grade.png"), "Reliability (by grade)")

    # Monotonicity checks by grade using bin_id order
    by_bin = grade_df.groupby("bin_id").agg(obs_rate=("event_4y", "mean"), count=("event_4y", "size")).reset_index().sort_values("bin_id")
    monotonic_ok = bool(np.all(np.diff(by_bin["obs_rate"].fillna(0.0)) >= -1e-12))
    rho, rho_p = spearmanr(by_bin["bin_id"], by_bin["obs_rate"].fillna(0.0))
    metrics.update({"monotonic_by_grade": monotonic_ok, "spearman_rho": float(rho), "spearman_p": float(rho_p)})

    # Grade-level stats with Wilson CIs and explainability aggregates
    grade_stats_rows = []
    total_eval = len(grade_df)
    for g, grp in grade_df.groupby("grade"):
        n = len(grp)
        k = int(np.nansum(grp["event_4y"].values))
        ci_low, ci_high = wilson_ci(k, n)
        avg_score = float(np.nanmean(grp["final_score"])) if n > 0 else 0.0
        avg_shares = {f"avg_share_{t}y": float(np.nanmean(grp.get(f"share_{t}y", np.nan))) for t in range(1, 5)}
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
    grade_stats = pd.DataFrame(grade_stats_rows).sort_values("grade")
    grade_stats.to_csv(os.path.join(cfg.output_dir, "grade_level_stats.csv"), index=False)

    # Stability / PSI between dev and oot on grades
    dev_grades = eval_df.loc[(eval_df["split"] == "dev") & (~pd.isna(eval_df["grade"])) , "grade"]
    oot_grades = eval_df.loc[(eval_df["split"] == "oot") & (~pd.isna(eval_df["grade"])) , "grade"]
    if len(dev_grades) > 0 and len(oot_grades) > 0:
        psi = psi_for_grades(dev_grades, oot_grades)
        metrics["PSI_grades_dev_vs_oot"] = float(psi)
        # Plot grade distributions
        plt.figure(figsize=(8, 4))
        dist = pd.DataFrame({
            "grade": sorted(set(dev_grades) | set(oot_grades)),
        })
        dist["dev"] = dist["grade"].map(dev_grades.value_counts(normalize=True))
        dist["oot"] = dist["grade"].map(oot_grades.value_counts(normalize=True))
        dist = dist.fillna(0.0)
        dist.set_index("grade").plot(kind="bar", figsize=(8, 4))
        plt.ylabel("Proportion")
        plt.title(f"Grade Distribution (PSI={psi:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "psi_grades.png"), dpi=200, bbox_inches="tight")
        plt.close()
    else:
        metrics["PSI_grades_dev_vs_oot"] = float("nan")

    # Plots: ROC and CAP (on evaluation subset)
    plot_roc(y, p, os.path.join(plots_dir, "roc.png"))
    plot_cap(y, p, os.path.join(plots_dir, "cap.png"))

    # Segmentation analysis
    segments_results: Dict[str, Dict[str, float]] = {}
    seg_cols = cfg.segment_cols or (["업종코드1"] if "업종코드1" in eval_df.columns else [])
    for col in seg_cols:
        segs = eval_sub[col].fillna("<NA>").astype(str).value_counts().index.tolist()
        per_seg: Dict[str, Dict[str, float]] = {}
        for s in segs:
            sub = eval_sub[eval_sub[col].astype(str) == s]
            yy = sub["event_4y"].astype(int).values
            pp = sub["final_score"].astype(float).clip(cfg.epsilon, 1.0 - cfg.epsilon).values
            if len(sub) < 50 or len(np.unique(yy)) < 2:
                continue
            auc = roc_auc_score(yy, pp)
            ar = 2.0 * (auc - 0.5)
            ks = compute_ks(yy, pp)
            per_seg[s] = {"AUC": float(auc), "AR": float(ar), "KS": float(ks), "count": int(len(sub))}
        if per_seg:
            segments_results[col] = per_seg

    # Save metrics summary
    summary = {
        "execution_date": datetime.now().isoformat(),
        "evaluation_rows": int(len(eval_sub)),
        "dev_rows": int(((~pd.isna(eval_df["event_4y"])) & (eval_df["split"] == "dev")).sum()),
        "oot_rows": int(((~pd.isna(eval_df["event_4y"])) & (eval_df["split"] == "oot")).sum()),
        "metrics": metrics,
        "segments": segments_results,
        "acceptance": {
            "AR_pass": bool(metrics.get("AR", 0.0) >= cfg.acceptance_ar),
            "KS_pass": bool(metrics.get("KS", 0.0) >= cfg.acceptance_ks),
            "PSI_pass": bool(np.isnan(metrics.get("PSI_grades_dev_vs_oot", np.nan)) or metrics.get("PSI_grades_dev_vs_oot", 1.0) <= cfg.acceptance_psi),
        }
    }
    with open(os.path.join(cfg.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Report (markdown) with Executive Summary
    lines: List[str] = []
    lines.append("# Validation Report")
    lines.append("")
    lines.append("## Executive Summary")
    ar = metrics.get("AR", float("nan"))
    ks = metrics.get("KS", float("nan"))
    psi = metrics.get("PSI_grades_dev_vs_oot", float("nan"))
    ar_pass = "pass" if ar >= cfg.acceptance_ar else "fail"
    ks_pass = "pass" if ks >= cfg.acceptance_ks else "fail"
    psi_pass = "pass" if (np.isnan(psi) or psi <= cfg.acceptance_psi) else "fail"
    lines.append(f"- Key conclusion: grading system {'meets' if (ar_pass=='pass' and ks_pass=='pass') else 'does not meet'} target performance.")
    lines.append(f"- Main metrics (OOT focus): AR={ar:.3f} ({ar_pass}), KS={ks:.3f} ({ks_pass}), PSI={psi:.3f} ({psi_pass}).")
    lines.append("- Required actions: none if all passes; otherwise investigate segments, recalibration, or governance triggers.")
    lines.append("")
    lines.append("## Global Metrics")
    lines.append(f"- AUC: {metrics.get('AUC', 0.0):.3f}")
    lines.append(f"- AR: {ar:.3f}")
    lines.append(f"- KS: {ks:.3f}")
    lines.append(f"- Top-decile lift: {metrics.get('top_decile_lift', 0.0):.2f}")
    lines.append(f"- Brier: {metrics.get('Brier', 0.0):.4f}")
    lines.append(f"- Calibration slope/intercept: {metrics.get('calibration_slope', 1.0):.3f} / {metrics.get('calibration_intercept', 0.0):.3f}")
    lines.append("")
    lines.append("## Monotonicity by Grade")
    lines.append(f"- Spearman rho (bin_id vs observed bad rate): {metrics.get('spearman_rho', 0.0):.3f} (p={metrics.get('spearman_p', 0.0):.3g})")
    lines.append(f"- Strict monotonic increase: {metrics.get('monotonic_by_grade', False)}")
    lines.append("")
    lines.append("## Stability (PSI)")
    lines.append(f"- PSI (dev vs oot) on grade distribution: {psi:.3f}")
    lines.append("")
    lines.append("## Segments (selected)")
    if segments_results:
        for col, res in segments_results.items():
            lines.append(f"- {col}:")
            for seg, m in res.items():
                lines.append(f"  - {seg}: AR={m['AR']:.3f}, KS={m['KS']:.3f} (n={m['count']})")
    else:
        lines.append("- No segment metrics computed")
    lines.append("")
    lines.append("## Figures")
    lines.append("- ROC: plots/roc.png")
    lines.append("- CAP: plots/cap.png")
    lines.append("- Reliability (deciles): plots/reliability_deciles.png")
    lines.append("- Reliability (by grade): plots/reliability_by_grade.png")
    lines.append("- Grade distribution and PSI: plots/psi_grades.png")

    with open(os.path.join(cfg.output_dir, "validation_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Validation completed. Outputs in: {cfg.output_dir}")


if __name__ == "__main__":
    import math
    main()
