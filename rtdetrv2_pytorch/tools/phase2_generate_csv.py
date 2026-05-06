"""
Generate per-non-key analysis CSV for phase-2 dynamic routing.

For each non-key inference record, this script logs:
1) Required similarity signal:
   - Cosine similarity between current non-key S5 and previous non-key S5.
2) Required quality gap summary:
   - Dataset-average mAP/mAP50 gap between oracle heavy-path and non-key-path,
     repeated on each row for downstream joining/analysis.
3) Optional per-frame proxy gaps:
   - Score and detection-count differences between oracle and non-key outputs.
"""

import argparse
import contextlib
import csv
import io
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def format_coco(targets, outputs, results_list):
    """Convert model outputs to COCO result dicts."""
    for target, output in zip(targets, outputs):
        image_id = int(target["image_id"].item())
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()

        for i in range(len(scores)):
            x1, y1, x2, y2 = boxes[i]
            results_list.append(
                {
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(scores[i]),
                }
            )


def evaluate_map(coco_gt, results, img_ids: Optional[Set[int]] = None):
    """Run COCO evaluation and return (mAP, mAP50)."""
    try:
        from pycocotools.cocoeval import COCOeval
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pycocotools is required for COCO mAP computation. Install pycocotools to run this script."
        ) from exc

    if not results and not img_ids:
        return 0.0, 0.0

    coco_dt = coco_gt.loadRes(results if results else [])
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    if img_ids is not None:
        evaluator.params.imgIds = sorted(list(img_ids))
    else:
        evaluator.params.imgIds = sorted(list(set([res["image_id"] for res in results])))

    evaluator.evaluate()
    evaluator.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.summarize()
    if len(evaluator.stats) < 2:
        return 0.0, 0.0
    return float(evaluator.stats[0]), float(evaluator.stats[1])


def extract_video_id(file_name: str) -> str:
    parts = os.path.normpath(file_name).split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return "default_video"


def _mean_score(result: Dict[str, torch.Tensor]) -> float:
    scores = result["scores"]
    if scores.numel() == 0:
        return 0.0
    return float(scores.mean().item())


def _num_dets(result: Dict[str, torch.Tensor]) -> int:
    return int(result["scores"].numel())


def _safe_cosine(curr_flat: torch.Tensor, prev_flat: Optional[torch.Tensor]) -> float:
    if prev_flat is None:
        return float("nan")
    if prev_flat.shape != curr_flat.shape:
        return float("nan")
    return float(F.cosine_similarity(curr_flat, prev_flat, dim=1, eps=1e-8).item())


def _safe_l1(curr_flat: torch.Tensor, prev_flat: Optional[torch.Tensor]) -> float:
    if prev_flat is None:
        return float("nan")
    if prev_flat.shape != curr_flat.shape:
        return float("nan")
    return float(torch.mean(torch.abs(curr_flat - prev_flat)).item())


def parse_args():
    parser = argparse.ArgumentParser(description="Generate phase-2 CSV from temporal inference.")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config yml")
    parser.add_argument("--weights", "-w", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV path (one row per non-key record)",
    )
    parser.add_argument("--summary_json", type=str, default=None, help="Optional summary JSON path")
    parser.add_argument("--nk_per_key", "-n", type=int, default=1, help="Number of non-key frames per key")
    parser.add_argument(
        "--frame_stride",
        "-f",
        type=int,
        default=1,
        help="Stride between key sequences; identical semantics to eval_realtime.py",
    )
    parser.add_argument("--max_batches", type=int, default=-1, help="Stop after N dataloader batches (-1 = all)")
    return parser.parse_args()


def main():
    args = parse_args()
    from src.core import YAMLConfig
    from src.zoo.temporal_rtdetr import TemporalRTDETR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = YAMLConfig(args.config)

    # Match eval_realtime stream assumptions for deterministic per-frame sequence traversal.
    if "val_dataloader" in cfg.yaml_cfg:
        if "batch_size" in cfg.yaml_cfg["val_dataloader"]:
            cfg.yaml_cfg["val_dataloader"]["batch_size"] = 1
        if "drop_last" in cfg.yaml_cfg["val_dataloader"]:
            cfg.yaml_cfg["val_dataloader"]["drop_last"] = False
        if "dataset" in cfg.yaml_cfg["val_dataloader"]:
            cfg.yaml_cfg["val_dataloader"]["dataset"]["max_frame_gap"] = 1
            cfg.yaml_cfg["val_dataloader"]["dataset"]["frame_stride"] = 1
            cfg.yaml_cfg["val_dataloader"]["dataset"]["pair_sampling_strategy"] = "all"

    base_model = cfg.model.to(device)
    hidden_dim = 256
    num_queries = 300
    if "RTDETRTransformerv2" in cfg.yaml_cfg:
        decoder_cfg = cfg.yaml_cfg["RTDETRTransformerv2"]
        hidden_dim = decoder_cfg.get("hidden_dim", 256)
        num_queries = decoder_cfg.get("num_queries", 300)
    elif "RTDETRTransformer" in cfg.yaml_cfg:
        decoder_cfg = cfg.yaml_cfg["RTDETRTransformer"]
        hidden_dim = decoder_cfg.get("hidden_dim", 256)
        num_queries = decoder_cfg.get("num_queries", 300)

    model = TemporalRTDETR(
        backbone=base_model.backbone,
        encoder=getattr(base_model, "encoder", None),
        decoder=getattr(base_model, "decoder", None),
        num_classes=cfg.yaml_cfg.get("num_classes", 80),
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        use_lightweight_decoder=cfg.yaml_cfg.get("use_lightweight_decoder", False),
        reuse_position=cfg.yaml_cfg.get("reuse_position", 0),
    ).to(device)

    print(f"Loading weights from: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))

    is_decoupled = any("lightweight_decoder.dec_score_head" in k for k in state_dict.keys())
    if is_decoupled:
        print("  [Auto-Detect] Decoupled non-key heads found. Decoupling model...")
        model.decouple_non_key_prediction_heads()

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    base_val_loader = cfg.val_dataloader
    from torch.utils.data import DataLoader

    val_dataloader = DataLoader(
        dataset=base_val_loader.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=base_val_loader.num_workers,
        collate_fn=base_val_loader.collate_fn,
        drop_last=False,
    )
    coco_gt = val_dataloader.dataset.coco
    postprocessor = cfg.postprocessor

    rows: List[Dict[str, object]] = []
    student_results: List[Dict[str, object]] = []
    oracle_results: List[Dict[str, object]] = []
    eval_img_ids_non_key: Set[int] = set()

    cycle_len = max(args.frame_stride, args.nk_per_key + 1)
    cycle_step = 0
    last_video_id = None
    latest_key_image_id: Optional[int] = None
    latest_key_s5_flat: Optional[torch.Tensor] = None
    prev_non_key_s5_flat: Optional[torch.Tensor] = None

    print(f"Streaming with cycle: 1 key : {args.nk_per_key} non-key (stride={cycle_len})")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Generating phase2 CSV")):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            img_key, target_key, img_non_key, target_non_key = batch

            # Video boundary reset for consistent temporal history.
            key_img_id = int(target_key[0]["image_id"].item())
            img_info = val_dataloader.dataset.img_id_to_info[key_img_id]
            current_video_id = extract_video_id(img_info["file_name"])
            if last_video_id is not None and current_video_id != last_video_id:
                cycle_step = 0
                prev_non_key_s5_flat = None
                latest_key_s5_flat = None
            last_video_id = current_video_id

            step = cycle_step % cycle_len
            if step >= args.nk_per_key:
                cycle_step += 1
                continue

            if step == 0:
                img_key = img_key.to(device)
                out_k, key_s5 = model.forward_key_frame(img_key, None, return_backbone_s5=True)
                _ = out_k
                latest_key_image_id = key_img_id
                # [B, C, H, W] -> [B, C*H*W]
                latest_key_s5_flat = key_s5.flatten(1).detach()

            if img_non_key is not None and len(img_non_key) > 0:
                img_non_key = img_non_key.to(device)
                orig_sizes_nk = torch.stack([t["orig_size"] for t in target_non_key], dim=0).to(device)

                # Student/non-key path with key cache.
                out_nk, non_key_s5 = model.forward_non_key_frame(
                    img_non_key,
                    None,
                    return_backbone_s5=True,
                )
                res_nk_batch = postprocessor(out_nk, orig_sizes_nk)

                # Preserve key-cache state before oracle heavy pass.
                cached_ccff = model.cached_ccff
                cached_content = model.cached_content
                cached_points = model.cached_points_unact

                # Oracle/heavy path on non-key frame.
                out_oracle = model.forward_key_frame(img_non_key, None)
                res_oracle_batch = postprocessor(out_oracle, orig_sizes_nk)

                # Restore key-cache state for subsequent non-key frames in the same cycle.
                model.cached_ccff = cached_ccff
                model.cached_content = cached_content
                model.cached_points_unact = cached_points

                format_coco(target_non_key, res_nk_batch, student_results)
                format_coco(target_non_key, res_oracle_batch, oracle_results)

                # [B, C, H, W] -> [B, C*H*W]
                curr_non_key_s5_flat = non_key_s5.flatten(1).detach()
                cosine_prev_non_key = _safe_cosine(curr_non_key_s5_flat, prev_non_key_s5_flat)
                l1_prev_non_key = _safe_l1(curr_non_key_s5_flat, prev_non_key_s5_flat)
                cosine_latest_key = _safe_cosine(curr_non_key_s5_flat, latest_key_s5_flat)

                for sample_idx, (student_det, oracle_det, target) in enumerate(
                    zip(res_nk_batch, res_oracle_batch, target_non_key)
                ):
                    non_key_image_id = int(target["image_id"].item())
                    eval_img_ids_non_key.add(non_key_image_id)

                    student_mean_score = _mean_score(student_det)
                    oracle_mean_score = _mean_score(oracle_det)
                    student_det_count = _num_dets(student_det)
                    oracle_det_count = _num_dets(oracle_det)

                    rows.append(
                        {
                            "record_index": len(rows),
                            "batch_index": batch_idx,
                            "sample_index": sample_idx,
                            "video_id": current_video_id,
                            "key_image_id": latest_key_image_id if latest_key_image_id is not None else -1,
                            "non_key_image_id": non_key_image_id,
                            "cosine_s5_prev_non_key": cosine_prev_non_key,
                            "l1_s5_prev_non_key": l1_prev_non_key,
                            "cosine_s5_latest_key": cosine_latest_key,
                            "student_mean_score": student_mean_score,
                            "oracle_mean_score": oracle_mean_score,
                            "oracle_minus_student_mean_score": oracle_mean_score - student_mean_score,
                            "student_det_count": student_det_count,
                            "oracle_det_count": oracle_det_count,
                            "oracle_minus_student_det_count": oracle_det_count - student_det_count,
                            "avg_map_gap": math.nan,
                            "avg_map50_gap": math.nan,
                            "student_map": math.nan,
                            "student_map50": math.nan,
                            "oracle_map": math.nan,
                            "oracle_map50": math.nan,
                        }
                    )

                prev_non_key_s5_flat = curr_non_key_s5_flat

            cycle_step += 1

    if not rows:
        raise RuntimeError("No non-key records were collected. Check dataloader or cycle parameters.")

    student_map, student_map50 = evaluate_map(coco_gt, student_results, eval_img_ids_non_key)
    oracle_map, oracle_map50 = evaluate_map(coco_gt, oracle_results, eval_img_ids_non_key)
    avg_map_gap = oracle_map - student_map
    avg_map50_gap = oracle_map50 - student_map50

    for row in rows:
        row["avg_map_gap"] = avg_map_gap
        row["avg_map50_gap"] = avg_map50_gap
        row["student_map"] = student_map
        row["student_map50"] = student_map50
        row["oracle_map"] = oracle_map
        row["oracle_map50"] = oracle_map50

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV with {len(rows)} non-key records: {output_csv}")

    print(
        "Aggregate non-key quality gap | "
        f"mAP gap (oracle-student): {avg_map_gap:.6f}, "
        f"mAP50 gap (oracle-student): {avg_map50_gap:.6f}"
    )

    if args.summary_json is not None:
        import json

        payload = {
            "num_rows": len(rows),
            "num_eval_non_key_images": len(eval_img_ids_non_key),
            "student_map": student_map,
            "student_map50": student_map50,
            "oracle_map": oracle_map,
            "oracle_map50": oracle_map50,
            "avg_map_gap": avg_map_gap,
            "avg_map50_gap": avg_map50_gap,
            "cycle_len": cycle_len,
            "nk_per_key": args.nk_per_key,
            "frame_stride": args.frame_stride,
        }
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
