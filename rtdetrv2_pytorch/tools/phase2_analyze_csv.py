"""
Analyze phase-2 CSV and report relationship strength between S5-difference signals
and per-frame quality-gap proxies.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _valid_pairs(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    # Average-rank tie handling.
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter, dtype=np.int64)
    inv[sorter] = np.arange(len(a))
    sorted_a = a[sorter]

    obs = np.r_[True, sorted_a[1:] != sorted_a[:-1]]
    dense = obs.cumsum() - 1
    counts = np.bincount(dense)

    starts = np.cumsum(np.r_[0, counts[:-1]])
    avg_ranks = starts + (counts - 1) / 2.0
    ranks_sorted = avg_ranks[dense]
    return ranks_sorted[inv].astype(np.float64)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson(rx, ry)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if x.size < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else (1.0 - ss_res / ss_tot)
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze phase2 CSV correlations.")
    parser.add_argument("--csv", type=str, required=True, help="Path to phase2 CSV")
    parser.add_argument("--out_json", type=str, default=None, help="Optional output JSON report")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV has no rows: {csv_path}")

    # Standardized metrics aligned with phase2_generate_csv.py
    metrics_to_analyze = [
        "frame_gap",
        "cosine_s5_prev",
        "l1_s5_prev",
        "max_abs_diff_s5_prev",
        "smoothed_max_diff_s5_prev",
        "topk_mean_diff_s5_prev",
        "cosine_s5_anchor",
        "l1_s5_anchor",
        "max_abs_diff_s5_anchor",
        "smoothed_max_diff_s5_anchor",
        "topk_mean_diff_s5_anchor",
    ]

    score_gap = np.array([_to_float(r.get("oracle_minus_student_mean_score")) for r in rows], dtype=np.float64)
    if not np.any(np.isfinite(score_gap)):
        # Fallback to map_gap if score_gap is missing
        score_gap = np.array([_to_float(r.get("map_gap")) for r in rows], dtype=np.float64)

    global_map_gap = _to_float(rows[0].get("avg_map_gap"))
    if math.isnan(global_map_gap):
        # Calculate from per-frame map_gap column if global is missing
        all_m_gaps = [_to_float(r.get("map_gap")) for r in rows]
        valid_m_gaps = [v for v in all_m_gaps if not math.isnan(v)]
        global_map_gap = sum(valid_m_gaps) / len(valid_m_gaps) if valid_m_gaps else float("nan")

    report = {
        "num_rows": len(rows),
        "global_avg_map_gap": global_map_gap,
        "metrics": {},
    }

    # Classification proxy: "High Gap" = top 25% of absolute errors
    abs_err = np.abs(score_gap)
    valid_abs_err = abs_err[np.isfinite(abs_err)]
    if valid_abs_err.size > 0:
        high_gap_threshold = np.percentile(valid_abs_err, 75)
        is_high_gap = abs_err > high_gap_threshold
    else:
        is_high_gap = np.zeros_like(score_gap, dtype=bool)

    for m_name in metrics_to_analyze:
        signal = np.array([_to_float(r.get(m_name)) for r in rows], dtype=np.float64)
        x, y = _valid_pairs(signal, score_gap)
        
        # Separation analysis
        mask = np.isfinite(signal) & np.isfinite(score_gap)
        s_valid = signal[mask]
        h_valid = is_high_gap[mask]
        
        sep_ratio = float("nan")
        if h_valid.any() and not h_valid.all():
            mean_high = np.mean(s_valid[h_valid])
            mean_low = np.mean(s_valid[~h_valid])
            std_all = np.std(s_valid)
            if std_all > 0:
                sep_ratio = abs(mean_high - mean_low) / std_all

        report["metrics"][m_name] = {
            "pearson": _pearson(x, y),
            "spearman": _spearman(x, y),
            "linear_fit": _linear_fit(x, y),
            "separation_ratio": sep_ratio,
            "num_valid": int(x.size)
        }

    def _fmt(v: float) -> str:
        return "nan" if (v is None or not math.isfinite(v)) else f"{v:.6f}"

    print(f"Rows: {report['num_rows']}")
    print(f"Global avg mAP gap: {_fmt(report['global_avg_map_gap'])}")
    print("-" * 110)
    print(f"{'Metric':<35} | {'Pearson':<10} | {'Spearman':<10} | {'SepRatio':<10} | {'R2':<10} | {'Valid'}")
    print("-" * 110)
    for m_name in metrics_to_analyze:
        m_data = report["metrics"][m_name]
        print(
            f"{m_name:<35} | "
            f"{_fmt(m_data['pearson']):<10} | "
            f"{_fmt(m_data['spearman']):<10} | "
            f"{_fmt(m_data['separation_ratio']):<10} | "
            f"{_fmt(m_data['linear_fit']['r2']):<10} | "
            f"{m_data['num_valid']}"
        )

    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved analysis JSON: {out_path}")


if __name__ == "__main__":
    main()
