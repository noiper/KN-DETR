import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict

# Ensure python path is correct
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.zoo.temporal_rtdetr import TemporalRTDETR
from src.zoo.rtdetr.box_ops import box_iou, box_cxcywh_to_xyxy

def box_xywh_to_xyxy(boxes):
    """Converts COCO [x, y, w, h] to [x1, y1, x2, y2]"""
    if isinstance(boxes, list):
        boxes = torch.tensor(boxes)
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)

def extract_video_id(file_name):
    parts = os.path.normpath(file_name).split(os.sep)
    return parts[0] if len(parts) > 1 else "default_video"

def main():
    parser = argparse.ArgumentParser(description="Diagnose IoU Distribution for Key vs Non-Key Paths")
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--weights', '-w', type=str, required=True)
    parser.add_argument('--nk_per_key', '-n', type=int, default=1)
    parser.add_argument('--score_thr', '-t', type=float, default=0.3, help="Only consider predictions above this score")
    parser.add_argument('--output', '-o', type=str, default='plots/iou_histogram.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = YAMLConfig(args.config)
    
    # 1. Build Model
    base_model = cfg.model.to(device)
    model = TemporalRTDETR(
        backbone=base_model.backbone,
        encoder=getattr(base_model, 'encoder', None),
        decoder=getattr(base_model, 'decoder', None),
        num_classes=cfg.yaml_cfg.get('num_classes', 80),
        hidden_dim=256,
        num_queries=300,
        use_lightweight_decoder=cfg.yaml_cfg.get('use_lightweight_decoder', True),
        reuse_position=cfg.yaml_cfg.get('reuse_position', 0),
    ).to(device)

    # 2. Load Weights
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    if any('lightweight_decoder.dec_score_head' in k for k in state_dict.keys()):
        model.decouple_non_key_prediction_heads()
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 3. Setup Dataloader
    from torch.utils.data import DataLoader
    base_val_loader = cfg.val_dataloader
    val_dataloader = DataLoader(
        dataset=base_val_loader.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=base_val_loader.num_workers,
        collate_fn=base_val_loader.collate_fn,
        drop_last=False
    )
    postprocessor = cfg.postprocessor

    key_ious = []
    nk_ious = []
    key_confs = []
    nk_confs = []

    cycle_len = args.nk_per_key + 1
    cycle_step = 0
    last_video_id = None

    print(f"Starting Diagnostic (1 Key : {args.nk_per_key} Non-Key)...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader)):
            img_key, target_key, img_non_key, target_non_key = batch
            
            # Video Boundary logic
            img_id = int(target_key[0]['image_id'].item())
            img_info = val_dataloader.dataset.img_id_to_info[img_id]
            current_video_id = extract_video_id(img_info['file_name'])
            if last_video_id is not None and current_video_id != last_video_id:
                cycle_step = 0
            last_video_id = current_video_id

            step = cycle_step % cycle_len
            if step >= cycle_len: 
                cycle_step += 1
                continue

            # Process Key Frame (step 0)
            if step == 0:
                img_key = img_key.to(device)
                out = model.forward_key_frame(img_key, None)
                target = target_key
                target_list = key_ious
                conf_target_list = key_confs
            else:
                img_non_key = img_non_key.to(device)
                out = model.forward_non_key_frame(img_non_key, None)
                target = target_non_key
                target_list = nk_ious
                conf_target_list = nk_confs

            # Calculate Best IoU for each GT box
            orig_size = torch.stack([t["orig_size"] for t in target], dim=0).to(device)
            results = postprocessor(out, orig_size)[0] # batch_size is 1
            
            # filter by score
            keep = results['scores'] > args.score_thr
            pred_boxes = results['boxes'][keep] # [M, 4] in xyxy
            pred_scores = results['scores'][keep]
            
            # get GT boxes
            gt_boxes_raw = target[0]['boxes'] # [N, 4]
            if gt_boxes_raw.numel() == 0:
                cycle_step += 1
                continue
                
            # Detect if GT is relative or absolute
            is_relative = (gt_boxes_raw <= 1.0).all()
            if is_relative:
                # Convert relative cxcywh to absolute xyxy
                h, w = orig_size[0]
                # Scale boxes
                gt_boxes_abs = gt_boxes_raw * torch.tensor([w, h, w, h], device=gt_boxes_raw.device)
                gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes_abs)
            else:
                # Assume absolute COCO xywh
                gt_boxes_xyxy = box_xywh_to_xyxy(gt_boxes_raw).to(device)

            if pred_boxes.numel() == 0:
                # No predictions above threshold, IoU is 0 for all GTs
                target_list.extend([0.0] * gt_boxes_xyxy.shape[0])
                conf_target_list.extend([0.0] * gt_boxes_xyxy.shape[0])
            else:
                # Pairwise IoU: [N_gt, M_pred]
                ious, _ = box_iou(gt_boxes_xyxy, pred_boxes)
                # Best IoU per GT
                best_iou_vals, best_indices = ious.max(dim=1)
                target_list.extend(best_iou_vals.cpu().numpy().tolist())
                
                # Confidence of these best spatial matches
                matched_confs = pred_scores[best_indices].cpu().numpy()
                conf_target_list.extend(matched_confs.tolist())

            cycle_step += 1

    # 4. Plotting IoU
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 50)
    plt.hist(key_ious, bins=bins, alpha=0.5, label=f'Key Frames (N={len(key_ious)})', color='blue', density=True)
    plt.hist(nk_ious, bins=bins, alpha=0.5, label=f'Non-Key Frames (N={len(nk_ious)})', color='orange', density=True)
    
    plt.title(f'IoU Distribution Comparison (Score Thr: {args.score_thr})')
    plt.xlabel('IoU with Ground Truth')
    plt.ylabel('Density')
    
    plt.axvline(np.mean(key_ious), color='blue', linestyle='dashed', linewidth=2, label=f'Key Mean: {np.mean(key_ious):.3f}')
    plt.axvline(np.mean(nk_ious), color='orange', linestyle='dashed', linewidth=2, label=f'NK Mean: {np.mean(nk_ious):.3f}')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output)
    print(f"IoU Plot saved to {args.output}")

    # 5. Plotting Confidence
    plt.figure(figsize=(10, 6))
    plt.hist(key_confs, bins=bins, alpha=0.5, label=f'Key Frames', color='blue', density=True)
    plt.hist(nk_confs, bins=bins, alpha=0.5, label=f'Non-Key Frames', color='orange', density=True)
    
    plt.title(f'Matched Prediction Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    
    plt.axvline(np.mean(key_confs), color='blue', linestyle='dashed', linewidth=2, label=f'Key Mean: {np.mean(key_confs):.3f}')
    plt.axvline(np.mean(nk_confs), color='orange', linestyle='dashed', linewidth=2, label=f'NK Mean: {np.mean(nk_confs):.3f}')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    conf_output = args.output.replace('iou', 'conf') if 'iou' in args.output else args.output.replace('.png', '_conf.png')
    plt.savefig(conf_output)
    print(f"Confidence Plot saved to {conf_output}")

    print(f"--- Summary ---")
    print(f"Key Avg IoU: {np.mean(key_ious):.4f} | Avg Conf: {np.mean(key_confs):.4f}")
    print(f"NK  Avg IoU: {np.mean(nk_ious):.4f} | Avg Conf: {np.mean(nk_confs):.4f}")

if __name__ == '__main__':
    main()
