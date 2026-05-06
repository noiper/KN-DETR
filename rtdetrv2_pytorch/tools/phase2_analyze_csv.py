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

    cosine = np.array([_to_float(r.get("cosine_s5_prev_non_key")) for r in rows], dtype=np.float64)
    l1 = np.array([_to_float(r.get("l1_s5_prev_non_key")) for r in rows], dtype=np.float64)
    score_gap = np.array([_to_float(r.get("oracle_minus_student_mean_score")) for r in rows], dtype=np.float64)
    det_gap = np.array([_to_float(r.get("oracle_minus_student_det_count")) for r in rows], dtype=np.float64)

    global_map_gap = _to_float(rows[0].get("avg_map_gap"))
    global_map50_gap = _to_float(rows[0].get("avg_map50_gap"))

    x1, y1 = _valid_pairs(cosine, score_gap)
    x2, y2 = _valid_pairs(cosine, det_gap)
    x3, y3 = _valid_pairs(l1, score_gap)
    x4, y4 = _valid_pairs(l1, det_gap)

    report = {
        "num_rows": len(rows),
        "num_valid_cosine_score_gap": int(x1.size),
        "num_valid_cosine_det_gap": int(x2.size),
        "global_avg_map_gap": global_map_gap,
        "global_avg_map50_gap": global_map50_gap,
        "metrics": {
            "cosine_vs_score_gap": {
                "pearson": _pearson(x1, y1),
                "spearman": _spearman(x1, y1),
                "linear_fit": _linear_fit(x1, y1),
            },
            "cosine_vs_det_gap": {
                "pearson": _pearson(x2, y2),
                "spearman": _spearman(x2, y2),
                "linear_fit": _linear_fit(x2, y2),
            },
            "l1_vs_score_gap": {
                "pearson": _pearson(x3, y3),
                "spearman": _spearman(x3, y3),
                "linear_fit": _linear_fit(x3, y3),
            },
            "l1_vs_det_gap": {
                "pearson": _pearson(x4, y4),
                "spearman": _spearman(x4, y4),
                "linear_fit": _linear_fit(x4, y4),
            },
        },
    }

    def _fmt(v: float) -> str:
        return "nan" if (v is None or not math.isfinite(v)) else f"{v:.6f}"

    print(f"Rows: {report['num_rows']}")
    print(f"Global avg mAP gap (oracle-student): {_fmt(report['global_avg_map_gap'])}")
    print(f"Global avg mAP50 gap (oracle-student): {_fmt(report['global_avg_map50_gap'])}")
    print(
        "Cosine vs score-gap | "
        f"pearson={_fmt(report['metrics']['cosine_vs_score_gap']['pearson'])}, "
        f"spearman={_fmt(report['metrics']['cosine_vs_score_gap']['spearman'])}, "
        f"r2={_fmt(report['metrics']['cosine_vs_score_gap']['linear_fit']['r2'])}"
    )
    print(
        "L1 vs score-gap     | "
        f"pearson={_fmt(report['metrics']['l1_vs_score_gap']['pearson'])}, "
        f"spearman={_fmt(report['metrics']['l1_vs_score_gap']['spearman'])}, "
        f"r2={_fmt(report['metrics']['l1_vs_score_gap']['linear_fit']['r2'])}"
    )

    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved analysis JSON: {out_path}")


if __name__ == "__main__":
    main()
