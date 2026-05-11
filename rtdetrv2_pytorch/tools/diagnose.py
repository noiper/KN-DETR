import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict

# Ensure python path is correct
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.zoo.temporal_rtdetr import TemporalRTDETR
from src.zoo.rtdetr.box_ops import box_iou, box_cxcywh_to_xyxy

def record_stats(results, target, iou_list, conf_list, score_thr, device, actual_size):
    """
    results: dict from postprocessor with 'boxes' (absolute xyxy)
    target: list containing a dict with 'boxes' (GT)
    actual_size: tensor of shape [1, 2] containing [W, H] of the image tensor
    """
    # filter by score
    keep = results['scores'] > score_thr
    pred_boxes = results['boxes'][keep] # [M, 4] xyxy
    pred_scores = results['scores'][keep]
    
    # get GT boxes
    gt_boxes_raw = target[0]['boxes'] # [N, 4]
    if gt_boxes_raw.numel() == 0:
        return
        
    w, h = actual_size[0, 0], actual_size[0, 1]
    
    # Handle GT Boxes conversion to absolute xyxy in the 640 space (actual_size)
    is_normalized = (gt_boxes_raw <= 1.01).all()
    if is_normalized:
        gt_boxes_abs = gt_boxes_raw.to(device) * torch.tensor([w, h, w, h], device=device)
        gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes_abs)
    else:
        gt_boxes_xyxy = gt_boxes_raw.to(device)

    if pred_boxes.numel() == 0:
        iou_list.extend([0.0] * gt_boxes_xyxy.shape[0])
        conf_list.extend([0.0] * gt_boxes_xyxy.shape[0])
    else:
        # Pairwise IoU: [N_gt, M_pred]
        ious, _ = box_iou(gt_boxes_xyxy, pred_boxes)
        # Best IoU per GT
        best_iou_vals, best_indices = ious.max(dim=1)
        iou_list.extend(best_iou_vals.cpu().numpy().tolist())
        
        # Confidence of matched predictions
        matched_confs = pred_scores[best_indices].cpu().numpy()
        conf_list.extend(matched_confs.tolist())

def extract_video_id(file_name):
    parts = os.path.normpath(file_name).split(os.sep)
    return parts[0] if len(parts) > 1 else "default_video"

def prepare_targets_for_loss(targets, device):
    """
    Criterion expects normalized cxcywh boxes in target['boxes']
    """
    new_targets = []
    for t in targets:
        nt = {k: v.to(device) for k, v in t.items()}
        # Standardize boxes to normalized cxcywh if they aren't already
        boxes = nt['boxes']
        is_normalized = (boxes <= 1.01).all()
        if not is_normalized:
            w, h = nt['orig_size'][0], nt['orig_size'][1]
            # Convert xyxy -> normalized cxcywh
            # Note: This is a bit tricky since we don't know the exact transform history,
            # but usually 'boxes' in target are absolute xyxy if ConvertBoxes(normalize=True) was skipped.
            # We use orig_size because that's what's typically in the dict.
            boxes[:, 2:] -= boxes[:, :2] # xyxy -> xywh
            boxes[:, :2] += boxes[:, 2:] / 2 # xywh -> cxcywh
            boxes /= torch.tensor([w, h, w, h], device=device)
            nt['boxes'] = boxes
        new_targets.append(nt)
    return new_targets

def main():
    parser = argparse.ArgumentParser(description="Diagnose IoU Distribution for Key vs Non-Key Paths")
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--weights', '-w', type=str, required=True)
    parser.add_argument('--nk_per_key', '-n', type=int, default=1)
    parser.add_argument('--score_thr', '-t', type=float, default=0.3)
    parser.add_argument('--output', '-o', type=str, default='plots/iou_histogram.png')
    parser.add_argument('--baseline', action='store_true')
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

    # 3. Setup Dataloader and Criterion
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
    criterion = cfg.criterion.to(device)
    criterion.eval()

    key_ious, nk_ious = [], []
    key_confs, nk_confs = [], []
    
    # Loss tracking
    loss_stats = {
        'key': {'class': [], 'box': []},
        'nk':  {'class': [], 'box': []}
    }

    last_video_id = None
    nk_counter = 0
    last_key_results_norm = None

    mode_str = "BASELINE" if args.baseline else "MODEL"
    print(f"Starting Diagnostic ({mode_str})...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader)):
            img_key, target_key, img_non_key, target_non_key = batch
            
            # Metadata for sizing and video tracking
            img_id = int(target_key[0]['image_id'].item())
            img_info = val_dataloader.dataset.img_id_to_info[img_id]
            current_video_id = extract_video_id(img_info['file_name'])
            force_key = (last_video_id is not None and current_video_id != last_video_id)
            last_video_id = current_video_id

            actual_size_key = torch.tensor([[img_key.shape[-1], img_key.shape[-2]]], device=device) # [W, H]
            actual_size_nk = torch.tensor([[img_non_key.shape[-1], img_non_key.shape[-2]]], device=device) # [W, H]

            # Prepare targets for loss (expects normalized cxcywh)
            tk_loss = prepare_targets_for_loss(target_key, device)
            tnk_loss = prepare_targets_for_loss(target_non_key, device)

            if nk_counter == 0 or force_key:
                # Refresh Key Path
                img_key = img_key.to(device)
                out_key = model.forward_key_frame(img_key, None)
                
                # Metrics (IoU/Conf)
                results_key = postprocessor(out_key, actual_size_key)[0]
                record_stats(results_key, target_key, key_ious, key_confs, args.score_thr, device, actual_size_key)
                
                # Loss Reporting
                l_dict = criterion(out_key, tk_loss)
                loss_stats['key']['class'].append(l_dict['loss_vfl'].item())
                loss_stats['key']['box'].append((l_dict['loss_bbox'] + l_dict['loss_giou']).item())
                
                if args.baseline:
                    norm_size = torch.tensor([[1.0, 1.0]], device=device)
                    last_key_results_norm = postprocessor(out_key, norm_size)[0]
                
                nk_counter = args.nk_per_key
            else:
                nk_counter -= 1

            # Process Non-Key Frame
            if args.baseline:
                results_nk = {
                    'boxes': last_key_results_norm['boxes'] * actual_size_nk.repeat(1, 2),
                    'scores': last_key_results_norm['scores'],
                    'labels': last_key_results_norm['labels']
                }
                # Note: Loss for baseline isn't directly computable via criterion without running model
            else:
                img_non_key = img_non_key.to(device)
                out_nk = model.forward_non_key_frame(img_non_key, None)
                results_nk = postprocessor(out_nk, actual_size_nk)[0]
                
                # Loss Reporting
                l_dict = criterion(out_nk, tnk_loss)
                loss_stats['nk']['class'].append(l_dict['loss_vfl'].item())
                loss_stats['nk']['box'].append((l_dict['loss_bbox'] + l_dict['loss_giou']).item())
            
            record_stats(results_nk, target_non_key, nk_ious, nk_confs, args.score_thr, device, actual_size_nk)

    # 4. Plotting
    plt.figure(figsize=(14, 6))
    bins = np.linspace(0, 1, 51)
    
    plt.subplot(1, 2, 1)
    if key_ious:
        plt.hist(key_ious, bins=bins, alpha=0.5, label=f'Key (Mean: {np.mean(key_ious):.3f})', color='blue', density=True)
    if nk_ious:
        plt.hist(nk_ious, bins=bins, alpha=0.5, label=f'Non-Key (Mean: {np.mean(nk_ious):.3f})', color='orange', density=True)
    plt.title(f'IoU Distribution ({mode_str})')
    plt.xlabel('IoU')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    if key_confs:
        plt.hist(key_confs, bins=bins, alpha=0.5, label=f'Key (Mean: {np.mean(key_confs):.3f})', color='blue', density=True)
    if nk_confs:
        plt.hist(nk_confs, bins=bins, alpha=0.5, label=f'Non-Key (Mean: {np.mean(nk_confs):.3f})', color='orange', density=True)
    plt.title(f'Confidence Distribution ({mode_str})')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output)
    
    print(f"\nSummary ({mode_str}):")
    print(f"  Key Path:  Avg IoU: {np.mean(key_ious):.4f} | Avg Conf: {np.mean(key_confs):.4f}")
    print(f"  NK Path:   Avg IoU: {np.mean(nk_ious):.4f} | Avg Conf: {np.mean(nk_confs):.4f}")
    
    print(f"\nLoss Analysis (Raw Criterion Values):")
    if loss_stats['key']['class']:
        print(f"  Key Loss:  Class: {np.mean(loss_stats['key']['class']):.4f} | Box: {np.mean(loss_stats['key']['box']):.4f}")
    if loss_stats['nk']['class']:
        print(f"  NK Loss:   Class: {np.mean(loss_stats['nk']['class']):.4f} | Box: {np.mean(loss_stats['nk']['box']):.4f}")

if __name__ == '__main__':
    main()
