"""
Generate per-non-key analysis CSV for phase-2 dynamic routing.

For each non-key inference record, this script logs:
1) Required similarity signal:
   - Cosine similarity between current non-key S5 and previous non-key S5.
2) Required quality gap summary:
   - Per-frame mAP/mAP50 gap between oracle heavy-path and non-key-path.
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
from typing import Dict, List, Optional

# Mock tensorboard to avoid ModuleNotFoundError
class MockSummaryWriter:
    def __init__(self, *args, **kwargs): pass
    def add_scalar(self, *args, **kwargs): pass
    def close(self): pass

sys.modules['torch.utils.tensorboard'] = type('tensorboard', (), {'SummaryWriter': MockSummaryWriter})
sys.modules['tensorboard'] = type('tensorboard', (), {})

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def format_coco_single(target, output):
    """Convert model output for a SINGLE image to COCO result dicts."""
    image_id = int(target["image_id"].item())
    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()

    results = []
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i]
        results.append(
            {
                "image_id": image_id,
                "category_id": int(labels[i]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(scores[i]),
            }
        )
    return results


def evaluate_map_single(coco_gt, results, img_id: int):
    """Run COCO evaluation for a SINGLE image."""
    try:
        from pycocotools.cocoeval import COCOeval
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pycocotools is required for COCO mAP computation. Install pycocotools to run this script."
        ) from exc

    if not results:
        # If no results, but we still need to evaluate this ID, 
        # COCOeval will correctly report 0 if we provide empty list.
        pass

    # COCOeval typically expects a 'dataset' object or file. 
    # To evaluate a SINGLE image from a larger GT, we must restrict it.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results if results else [])
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = [img_id]

        evaluator.evaluate()
        evaluator.accumulate()
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


def _safe_max_abs_diff(curr: torch.Tensor, prev: Optional[torch.Tensor]) -> float:
    if prev is None:
        return float("nan")
    # curr, prev are [B, C, H, W]
    diff = (curr - prev).abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]
    return float(diff.max().item())


def _safe_smoothed_max_diff(curr: torch.Tensor, prev: Optional[torch.Tensor], kernel_size: int = 3) -> float:
    if prev is None:
        return float("nan")
    # curr, prev are [B, C, H, W]
    diff = (curr - prev).abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]
    padding = kernel_size // 2
    smoothed = F.avg_pool2d(diff, kernel_size=kernel_size, stride=1, padding=padding)
    return float(smoothed.max().item())


def _safe_topk_mean_diff(curr: torch.Tensor, prev: Optional[torch.Tensor], k: int = 10) -> float:
    if prev is None:
        return float("nan")
    # curr, prev are [B, C, H, W]
    diff = (curr - prev).abs().mean(dim=1).flatten(1)  # [B, H*W]
    k = min(k, diff.shape[1])
    topk = torch.topk(diff, k=k).values
    return float(topk.mean().item())


def parse_args():
    parser = argparse.ArgumentParser(description="Generate phase-2 CSV for per-frame error analysis.")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config yml")
    parser.add_argument("--weights", "-w", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--output_csv", "-o", type=str, default=None, help="Output CSV path (defaults to dynamic naming)")
    parser.add_argument("--summary_json", type=str, default=None, help="Optional summary JSON path")
    parser.add_argument("--nk_per_key", "-n", type=int, default=1, help="Number of non-key frames per key")
    parser.add_argument("--frame_stride", "-f", type=int, default=1, help="Stride between key sequences")
    parser.add_argument("--max_batches", type=int, default=-1, help="Stop after N dataloader batches (-1 = all)")
    args = parser.parse_args()
    
    if args.output_csv is None:
        cfg_name = Path(args.config).stem.replace("phase1_", "").replace("_detection", "")
        weight_name = Path(args.weights).stem
        args.output_csv = f"tables/{cfg_name}_level{args.nk_per_key}_{weight_name}.csv"
        
    return args


def main():
    args = parse_args()
    from src.core import YAMLConfig
    from src.zoo.temporal_rtdetr import TemporalRTDETR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = YAMLConfig(args.config)

    # Force batch_size=1
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

    cycle_len = max(args.frame_stride, args.nk_per_key + 1)
    cycle_step = 0
    last_video_id = None
    latest_key_image_id: Optional[int] = None
    latest_key_s5: Optional[torch.Tensor] = None
    latest_key_s5_flat: Optional[torch.Tensor] = None
    prev_non_key_s5: Optional[torch.Tensor] = None
    prev_non_key_s5_flat: Optional[torch.Tensor] = None

    print(f"Streaming with cycle: 1 key : {args.nk_per_key} non-key (stride={cycle_len})")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Generating Per-Frame Analysis")):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            img_key, target_key, img_non_key, target_non_key = batch

            key_img_id = int(target_key[0]["image_id"].item())
            img_info = val_dataloader.dataset.img_id_to_info[key_img_id]
            current_video_id = extract_video_id(img_info["file_name"])
            if last_video_id is not None and current_video_id != last_video_id:
                cycle_step = 0
                prev_non_key_s5 = None
                prev_non_key_s5_flat = None
                latest_key_s5 = None
                latest_key_s5_flat = None
            last_video_id = current_video_id

            step = cycle_step % cycle_len
            if step >= args.nk_per_key:
                cycle_step += 1
                continue

            if step == 0:
                img_key = img_key.to(device)
                out_k, key_s5 = model.forward_key_frame(img_key, None, return_backbone_s5=True)
                latest_key_image_id = key_img_id
                latest_key_s5 = key_s5.detach()
                latest_key_s5_flat = key_s5.flatten(1).detach()

            if img_non_key is not None and len(img_non_key) > 0:
                img_non_key = img_non_key.to(device)
                orig_sizes_nk = torch.stack([t["orig_size"] for t in target_non_key], dim=0).to(device)
                non_key_image_id = int(target_non_key[0]["image_id"].item())

                # Student path
                out_nk, non_key_s5 = model.forward_non_key_frame(img_non_key, None, return_backbone_s5=True)
                res_nk_batch = postprocessor(out_nk, orig_sizes_nk)

                # Oracle path (Preserve cache)
                cached_ccff = [f.clone() for f in model.cached_ccff] if model.cached_ccff else None
                cached_content = model.cached_content.clone() if model.cached_content is not None else None
                cached_points = model.cached_points_unact.clone() if model.cached_points_unact is not None else None

                out_oracle = model.forward_key_frame(img_non_key, None)
                res_oracle_batch = postprocessor(out_oracle, orig_sizes_nk)

                # Restore
                model.cached_ccff = cached_ccff
                model.cached_content = cached_content
                model.cached_points_unact = cached_points

                # Per-frame mAP
                student_coco = format_coco_single(target_non_key[0], res_nk_batch[0])
                oracle_coco = format_coco_single(target_non_key[0], res_oracle_batch[0])
                
                s_map, s_map50 = evaluate_map_single(coco_gt, student_coco, non_key_image_id)
                o_map, o_map50 = evaluate_map_single(coco_gt, oracle_coco, non_key_image_id)

                # Signals
                curr_non_key_s5 = non_key_s5.detach()
                curr_non_key_s5_flat = non_key_s5.flatten(1).detach()
                
                rows.append(
                    {
                        "video_id": current_video_id,
                        "image_id": non_key_image_id,
                        "frame_gap": int(non_key_image_id - latest_key_image_id),
                        "student_map": s_map,
                        "student_map50": s_map50,
                        "oracle_map": o_map,
                        "oracle_map50": o_map50,
                        "map_gap": o_map - s_map,
                        "map50_gap": o_map50 - s_map50,
                        "cosine_s5_prev": _safe_cosine(curr_non_key_s5_flat, prev_non_key_s5_flat),
                        "l1_s5_prev": _safe_l1(curr_non_key_s5_flat, prev_non_key_s5_flat),
                        "max_abs_diff_s5_prev": _safe_max_abs_diff(curr_non_key_s5, prev_non_key_s5),
                        "smoothed_max_diff_s5_prev": _safe_smoothed_max_diff(curr_non_key_s5, prev_non_key_s5),
                        "topk_mean_diff_s5_prev": _safe_topk_mean_diff(curr_non_key_s5, prev_non_key_s5),
                        "cosine_s5_anchor": _safe_cosine(curr_non_key_s5_flat, latest_key_s5_flat),
                        "l1_s5_anchor": _safe_l1(curr_non_key_s5_flat, latest_key_s5_flat),
                        "max_abs_diff_s5_anchor": _safe_max_abs_diff(curr_non_key_s5, latest_key_s5),
                        "smoothed_max_diff_s5_anchor": _safe_smoothed_max_diff(curr_non_key_s5, latest_key_s5),
                        "topk_mean_diff_s5_anchor": _safe_topk_mean_diff(curr_non_key_s5, latest_key_s5),
                        "oracle_mean_score": _mean_score(res_oracle_batch[0]),
                        "student_mean_score": _mean_score(res_nk_batch[0]),
                        "oracle_num_dets": _num_dets(res_oracle_batch[0]),
                        "student_num_dets": _num_dets(res_nk_batch[0]),
                    }
                )

                prev_non_key_s5 = curr_non_key_s5
                prev_non_key_s5_flat = curr_non_key_s5_flat

            cycle_step += 1

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved Per-Frame Analysis CSV: {output_csv}")


if __name__ == "__main__":
    main()
