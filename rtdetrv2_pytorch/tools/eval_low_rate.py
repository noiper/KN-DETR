"""
Real-Time Temporal Inference Simulator
Simulates a live continuous streaming environment with configurable K-NK ratios.
Tracks Latency, Peak VRAM Memory Allocation, and Combined COCO mAP.
"""

import os
import sys
import time
import argparse
import contextlib
import io
import torch
import numpy as np
from tqdm import tqdm

# Ensure python path is correct when run from terminal
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.zoo.temporal_rtdetr import TemporalRTDETR
from src.zoo.rtdetr.box_ops import box_iou, box_cxcywh_to_xyxy
from pycocotools.cocoeval import COCOeval
from typing import Dict

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

def record_tp_fp_stats(results, target, tp_scores_list, fp_scores_list, score_thr, device, actual_size, iou_thr=0.5):
    """
    results: dict from postprocessor with 'boxes' (absolute xyxy)
    target: list containing a dict with 'boxes' (GT)
    """
    keep = results['scores'] > score_thr
    pred_boxes = results['boxes'][keep]
    pred_scores = results['scores'][keep]
    pred_labels = results['labels'][keep]
    
    gt_boxes_raw = target[0]['boxes']
    gt_labels = target[0]['labels']
    
    if pred_boxes.numel() == 0:
        return

    if gt_boxes_raw.numel() == 0:
        fp_scores_list.extend(pred_scores.cpu().numpy().tolist())
        return

    w, h = actual_size[0, 0], actual_size[0, 1]
    is_normalized = (gt_boxes_raw <= 1.01).all()
    if is_normalized:
        gt_boxes_abs = gt_boxes_raw.to(device) * torch.tensor([w, h, w, h], device=device)
        gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes_abs)
    else:
        gt_boxes_xyxy = gt_boxes_raw.to(device)

    # [N_gt, M_pred]
    ious, _ = box_iou(gt_boxes_xyxy, pred_boxes)
    
    # --- Move to CPU once to avoid synchronization bottleneck ---
    pred_scores_np = pred_scores.cpu().numpy()
    pred_labels_np = pred_labels.cpu().numpy()
    gt_labels_np = gt_labels.cpu().numpy()
    ious_np = ious.cpu().numpy()
    
    # Sort predictions by score descending
    indices = np.argsort(-pred_scores_np)
    matched_gt = np.zeros(gt_boxes_xyxy.shape[0], dtype=bool)
    
    for idx in indices:
        label = pred_labels_np[idx]
        best_iou = -1
        best_gt_idx = -1
        
        for g_idx in range(gt_boxes_xyxy.shape[0]):
            if gt_labels_np[g_idx] == label:
                iou = ious_np[g_idx, idx]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
        
        if best_iou >= iou_thr and not matched_gt[best_gt_idx]:
            tp_scores_list.append(float(pred_scores_np[idx]))
            matched_gt[best_gt_idx] = True
        else:
            fp_scores_list.append(float(pred_scores_np[idx]))

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
            boxes[:, 2:] -= boxes[:, :2] # xyxy -> xywh
            boxes[:, :2] += boxes[:, 2:] / 2 # xywh -> cxcywh
            boxes /= torch.tensor([w, h, w, h], device=device)
            nt['boxes'] = boxes
        new_targets.append(nt)
    return new_targets

def format_coco(targets, outputs, results_list):
    """Converts tensor outputs to the exact dictionary format required by COCOeval"""
    for target, output in zip(targets, outputs):
        image_id = int(target['image_id'].item())
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()

        for i in range(len(scores)):
            x1, y1, x2, y2 = boxes[i]
            results_list.append({
                "image_id": image_id,
                "category_id": int(labels[i]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(scores[i])
            })

def scale_results(results, score_scale):
    if score_scale == 1.0:
        return results
    scaled = []
    for det in results:
        score = float(det['score']) * score_scale
        out = det.copy()
        out['score'] = score
        scaled.append(out)
    return scaled

def parse_scale_grid(grid_text):
    values = []
    for token in grid_text.split(','):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("score scale grid cannot be empty")
    return values

def evaluate_map(coco_gt, results, title, img_ids=None):
    """Runs pycocotools evaluation and returns (mAP, mAP50)."""
    if not results and not img_ids:
        return 0.0, 0.0
        
    if not results:
        coco_dt = coco_gt.loadRes([])
    else:
        coco_dt = coco_gt.loadRes(results)
        
    evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # STRICTLY LIMIT EVALUATION TO THE IMAGES PREDICTED
    # Prevents artificial deflation when evaluating partial streams
    if img_ids is not None:
        evaluator.params.imgIds = sorted(list(img_ids))
    else:
        predicted_img_ids = sorted(list(set([res['image_id'] for res in results])))
        evaluator.params.imgIds = predicted_img_ids
    
    evaluator.evaluate()
    evaluator.accumulate()
    # COCOeval fills `stats` during summarize(); silence the default table output.
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.summarize()
    if len(evaluator.stats) < 2:
        return 0.0, 0.0
    return evaluator.stats[0], evaluator.stats[1]

def extract_video_id(file_name):
    """Extract video ID from filename (matches ViratTemporalDataset logic)"""
    import os
    parts = os.path.normpath(file_name).split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return "default_video"

def _extract_total_loss(loss_dict: Dict[str, torch.Tensor]) -> float:
    """Extracts main detection loss, ignoring auxiliary and denoising."""
    relevant_keys = [k for k in loss_dict.keys() if not any(x in k for x in ['_aux_', '_dn_', '_enc_'])]
    if not relevant_keys:
        return 0.0
    return sum(loss_dict[k] for k in relevant_keys).item()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Temporal RT-DETR in Real-Time Simulation")
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config yml')
    parser.add_argument('--weights','-w',  type=str, required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--warmup', type=int, default=10, help='Ignore first N batches for timing/memory')
    parser.add_argument('--skip', '-s', type=int, default=0,
                        help='Number of frames to skip between evaluations. Default 0.')
    parser.add_argument('--baseline', action='store_true',
                        help='Baseline: reuse key-frame detections directly for non-key frames')
    parser.add_argument('--key_score', '-ks', type=float, default=1.0,
                        help='Multiply key-path confidence scores by this factor before evaluation')
    parser.add_argument('--nonkey_score', '-ns', type=float, default=1.0,
                        help='Multiply non-key-path confidence scores by this factor before evaluation')
    parser.add_argument('--score_thr', '-st', type=float, default=0.3,
                        help='Confidence threshold for diagnostic metrics (IoU, TP/FP)')
    parser.add_argument('--tune_score', '-ts', action='store_true',
                        help='Grid search key/non-key score scales for best combined mAP')
    parser.add_argument('--score_grid', type=str, default='0.8,0.9,1.0,1.1,1.2',
                        help='Comma-separated scale grid for --tune_score')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Deployment Device: {device}")
    
    # 1. Load the raw config
    cfg = YAMLConfig(args.config)
    
    # --- HARDCODE BATCH SIZE TO 1 FOR REAL-TIME SIMULATION ---
    if 'val_dataloader' in cfg.yaml_cfg:
        print("Forcing validation batch_size=1 and drop_last=False for accurate real-time metrics.")
        if 'batch_size' in cfg.yaml_cfg['val_dataloader']:
            cfg.yaml_cfg['val_dataloader']['batch_size'] = 1
        if 'drop_last' in cfg.yaml_cfg['val_dataloader']:
            cfg.yaml_cfg['val_dataloader']['drop_last'] = False

        if 'dataset' in cfg.yaml_cfg['val_dataloader']:
            print("Forcing dataset max_frame_gap=1, frame_stride=1, pair_sampling_strategy='all' to simulate continuous stream.")
            cfg.yaml_cfg['val_dataloader']['dataset']['max_frame_gap'] = 1
            cfg.yaml_cfg['val_dataloader']['dataset']['frame_stride'] = 1
            cfg.yaml_cfg['val_dataloader']['dataset']['pair_sampling_strategy'] = 'all'
    
    # 2. Build Model Architecture
    base_model = cfg.model.to(device)
    hidden_dim = 256
    num_queries = 300
    if 'RTDETRTransformerv2' in cfg.yaml_cfg:
        decoder_cfg = cfg.yaml_cfg['RTDETRTransformerv2']
        hidden_dim = decoder_cfg.get('hidden_dim', 256)
        num_queries = decoder_cfg.get('num_queries', 300)
    elif 'RTDETRTransformer' in cfg.yaml_cfg:
        decoder_cfg = cfg.yaml_cfg['RTDETRTransformer']
        hidden_dim = decoder_cfg.get('hidden_dim', 256)
        num_queries = decoder_cfg.get('num_queries', 300)
    
    model = TemporalRTDETR(
        backbone=base_model.backbone,
        encoder=getattr(base_model, 'encoder', None),
        decoder=getattr(base_model, 'decoder', None),
        num_classes=cfg.yaml_cfg.get('num_classes', 80),
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        use_lightweight_decoder=cfg.yaml_cfg.get('use_lightweight_decoder', False),
        reuse_position=cfg.yaml_cfg.get('reuse_position', 0),
    ).to(device)
    
    # 3. Load Weights
    print(f"Loading weights from {args.weights}...")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))

    # --- AUTO-DECOUPLE DETECTION ---
    # If the checkpoint contains decoupled non-key heads, we MUST decouple the model
    # before loading to avoid overwriting the heavy decoder's heads with the student's.
    is_decoupled = any('lightweight_decoder.dec_score_head' in k for k in state_dict.keys())
    if is_decoupled:
        print("   [Auto-Detect] Decoupled prediction heads found in checkpoint. Decoupling model...")
        model.decouple_non_key_prediction_heads()
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # --- PHYSICAL DATALOADER REBUILD FOR BATCH_SIZE=1 ---
    base_val_loader = cfg.val_dataloader
    from torch.utils.data import DataLoader
    from src.data.transforms import ConvertBoxes, SanitizeBoundingBoxes
    
    # Add necessary box conversions for criterion compatibility
    # These won't affect COCOeval as it uses image_id to look up ground truth
    base_val_loader.dataset.transforms.transforms.append(SanitizeBoundingBoxes(min_size=1))
    base_val_loader.dataset.transforms.transforms.append(ConvertBoxes(fmt='cxcywh', normalize=True))

    print("Rebuilding validation dataloader to force batch_size=1...")
    val_dataloader = DataLoader(
        dataset=base_val_loader.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=base_val_loader.num_workers,
        collate_fn=base_val_loader.collate_fn,
        drop_last=False
    )
    # -----------------------------------------------------
    coco_gt = val_dataloader.dataset.coco
    postprocessor = cfg.postprocessor
    criterion = cfg.criterion
    criterion.eval()
    print(f"Non-key mode: {'baseline (reuse key detections)' if args.baseline else 'model forward'}")
    
    res_key = []
    res_nk = []
    eval_img_ids_key = set()
    eval_img_ids_nk = set()
    latest_key_results = None
    latest_key_results_norm = None
    out_k = None

    # Diagnostic Stats
    key_ious, nk_ious = [], []
    key_confs, nk_confs = [], []
    key_tp_scores, key_fp_scores = [], []
    nk_tp_scores, nk_fp_scores = [], []
    
    loss_stats = {
        'key': {'class': [], 'box': []},
        'nk':  {'class': [], 'box': []}
    }
    
    metrics = {
        'k_time': 0.0, 'k_mem': 0.0, 'k_frames': 0, 'k_loss': 0.0,
        'nk_time': 0.0, 'nk_mem': 0.0, 'nk_frames': 0, 'nk_loss': 0.0
    }

    frame_idx = 0
    last_video_id = None

    print(f"\n--- INITIATING LOW RATE SIMULATION (Skip: {args.skip}) ---")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc="Streaming Video")):
            img_key, target_key, _, _ = batch
            
            # --- VIDEO BOUNDARY DETECTION & CYCLE RESET ---
            img_id = int(target_key[0]['image_id'].item())
            img_info = val_dataloader.dataset.img_id_to_info[img_id]
            current_video_id = extract_video_id(img_info['file_name'])
            
            if last_video_id is not None and current_video_id != last_video_id:
                # Video changed! Reset the simulation cycle to start with a Key frame.
                frame_idx = 0
            last_video_id = current_video_id
            
            # LOW RATE SKIP LOGIC
            if frame_idx % (args.skip + 1) != 0:
                frame_idx += 1
                continue
            
            eval_idx = frame_idx // (args.skip + 1)
            is_key_frame = (eval_idx % 2 == 0)
            
            # Use original image size for ALL evaluation and diagnostic coordinate scaling
            orig_size = target_key[0]['orig_size'].to(device) # [W, H]
            target_loss = prepare_targets_for_loss(target_key, device)

            # ==========================================
            # KEY FRAME PASS
            # ==========================================
            if is_key_frame:
                img_key = img_key.to(device)
                target_key = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_key]
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                
                t0 = time.perf_counter()
                
                # Forward pass natively caches the features for upcoming Non-Key frames
                out_k = model.forward_key_frame(img_key, None)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                orig_sizes_k = torch.stack([t["orig_size"] for t in target_key], dim=0).to(device)
                latest_key_results = postprocessor(out_k, orig_sizes_k)[0]
                
                # Diagnostics
                record_stats(latest_key_results, target_key, key_ious, key_confs, args.score_thr, device, orig_size.unsqueeze(0))
                record_tp_fp_stats(latest_key_results, target_key, key_tp_scores, key_fp_scores, args.score_thr, device, orig_size.unsqueeze(0))

                if i >= args.warmup:
                    metrics['k_time'] += (t1 - t0)
                    metrics['k_frames'] += 1
                    if torch.cuda.is_available():
                        metrics['k_mem'] += torch.cuda.max_memory_allocated() / (1024 ** 2)
                    
                    # Loss tracking
                    loss_dict = criterion(out_k, target_loss)
                    metrics['k_loss'] += _extract_total_loss(loss_dict)
                    loss_stats['key']['class'].append(loss_dict['loss_vfl'].item())
                    loss_stats['key']['box'].append((loss_dict['loss_bbox'] + loss_dict['loss_giou']).item())
                
                format_coco(target_key, [latest_key_results], res_key)
                for t in target_key:
                    eval_img_ids_key.add(int(t['image_id'].item()))
                
                if args.baseline:
                    norm_size = torch.tensor([[1.0, 1.0]], device=device)
                    latest_key_results_norm = postprocessor(out_k, norm_size)[0]
            
            # ==========================================
            # NON-KEY FRAME PASS
            # ==========================================
            else:
                target_key = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_key]
                if args.baseline:
                    if latest_key_results_norm is None:
                        raise RuntimeError("No cached key results available for non-key propagation")
                    t2 = time.perf_counter()
                    # Scale normalized key boxes to the CURRENT frame's original resolution
                    res_nk_batch = {
                        'boxes': latest_key_results_norm['boxes'] * orig_size.repeat(2),
                        'scores': latest_key_results_norm['scores'],
                        'labels': latest_key_results_norm['labels']
                    }
                    t3 = time.perf_counter()

                    # Baseline Loss Computation:
                    # We pass the cached key outputs through the criterion against the current NK ground truth
                    if i >= args.warmup:
                        # Construct a mock output dictionary that the criterion can process
                        # latest_key_results_norm contains 'boxes', 'scores', 'labels'
                        # We need 'pred_logits' and 'pred_boxes' in normalized cxcywh
                        # latest_key_results_norm['boxes'] is already normalized xyxy, need to convert
                        
                        # Note: This assumes out_k was cached. We need out_k from the Key pass.
                        # Since out_k is exactly what produced latest_key_results_norm, 
                        # we can just use the cached out_k for loss computation against tnk_loss.
                        l_dict_nk = criterion(out_k, target_loss)
                        metrics['nk_loss'] += _extract_total_loss(l_dict_nk)
                        loss_stats['nk']['class'].append(l_dict_nk['loss_vfl'].item())
                        loss_stats['nk']['box'].append((l_dict_nk['loss_bbox'] + l_dict_nk['loss_giou']).item())
                else:
                    img_key = img_key.to(device)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    
                    t2 = time.perf_counter()
                    
                    # Relies on the cache stored during key frame pass
                    out_nk = model.forward_non_key_frame(img_key, None)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    
                    orig_sizes_nk = torch.stack([t["orig_size"] for t in target_key], dim=0).to(device)
                    res_nk_batch = postprocessor(out_nk, orig_sizes_nk)[0]

                    if i >= args.warmup:
                        # Loss tracking for non-key model
                        loss_dict = criterion(out_nk, target_loss)
                        metrics['nk_loss'] += _extract_total_loss(loss_dict)
                        loss_stats['nk']['class'].append(loss_dict['loss_vfl'].item())
                        loss_stats['nk']['box'].append((loss_dict['loss_bbox'] + loss_dict['loss_giou']).item())
                
                # Diagnostics
                record_stats(res_nk_batch, target_key, nk_ious, nk_confs, args.score_thr, device, orig_size.unsqueeze(0))
                record_tp_fp_stats(res_nk_batch, target_key, nk_tp_scores, nk_fp_scores, args.score_thr, device, orig_size.unsqueeze(0))

                if i >= args.warmup:
                    if not args.baseline:
                        metrics['nk_time'] += (t3 - t2)
                        metrics['nk_frames'] += 1
                        if torch.cuda.is_available():
                            metrics['nk_mem'] += torch.cuda.max_memory_allocated() / (1024 ** 2)
                    else:
                        # For baseline, we still track frames but time is near-zero
                        metrics['nk_frames'] += 1
                
                format_coco(target_key, [res_nk_batch], res_nk)
                for t in target_key:
                    eval_img_ids_nk.add(int(t['image_id'].item()))
            
            frame_idx += 1

    # Calculate Averages
    avg_k_time = (metrics['k_time'] / metrics['k_frames']) * 1000 if metrics['k_frames'] else 0
    avg_nk_time = (metrics['nk_time'] / metrics['nk_frames']) * 1000 if metrics['nk_frames'] else 0
    
    avg_k_mem = (metrics['k_mem'] / metrics['k_frames']) if metrics['k_frames'] > 0 else 0
    avg_nk_mem = (metrics['nk_mem'] / metrics['nk_frames']) if metrics['nk_frames'] > 0 else 0
    
    avg_k_loss = (metrics['k_loss'] / metrics['k_frames']) if metrics['k_frames'] > 0 else 0
    avg_nk_loss = (metrics['nk_loss'] / metrics['nk_frames']) if metrics['nk_frames'] > 0 else 0

    key_scale = args.key_score
    nonkey_scale = args.nonkey_score
    combined_img_ids = eval_img_ids_key | eval_img_ids_nk

    if args.tune_score:
        grid = parse_scale_grid(args.score_grid)
        best = None
        for ks in grid:
            for ns in grid:
                scaled_key = scale_results(res_key, ks)
                scaled_nk = scale_results(res_nk, ns)
                
                # Filter out overlapping image IDs from non-key results
                filtered_nk = [det for det in scaled_nk if det['image_id'] not in eval_img_ids_key]
                
                combined_map_tmp, combined_map50_tmp = evaluate_map(
                    coco_gt, scaled_key + filtered_nk, "COMBINED OVERALL AVERAGE", combined_img_ids
                )
                score = (combined_map50_tmp, combined_map_tmp)
                if best is None or score > best['score']:
                    best = {
                        'key_scale': ks,
                        'nonkey_scale': ns,
                        'combined_map': combined_map_tmp,
                        'combined_map50': combined_map50_tmp,
                        'score': score,
                    }
        key_scale = best['key_scale']
        nonkey_scale = best['nonkey_scale']
        print(f"Tuned score scales: key={key_scale:.3f}, non-key={nonkey_scale:.3f}")

    scaled_res_key = scale_results(res_key, key_scale)
    scaled_res_nk = scale_results(res_nk, nonkey_scale)
    
    # Filter out overlapping image IDs from non-key results for final combined metric
    final_filtered_nk = [det for det in scaled_res_nk if det['image_id'] not in eval_img_ids_key]
    scaled_combined = scaled_res_key + final_filtered_nk

    map_k, map50_k = evaluate_map(coco_gt, scaled_res_key, "HEAVY KEY MODEL ONLY", eval_img_ids_key)
    map_nk, map50_nk = evaluate_map(coco_gt, scaled_res_nk, "LIGHTWEIGHT NON-KEY MODEL ONLY", eval_img_ids_nk)
    combined_map, combined_map50 = evaluate_map(coco_gt, scaled_combined, "COMBINED OVERALL AVERAGE", combined_img_ids)

    print("\n" + "="*70)
    print(f"FINAL SUMMARY (Skip: {args.skip})")
    print("="*70)
    print(f"Score scales -> key: {key_scale:.3f}, non-key: {nonkey_scale:.3f}")
    print(f"Key      mAP: {map_k:.4f} | mAP50: {map50_k:.4f} | Loss: {avg_k_loss:.4f}")
    print(f"Non-Key  mAP: {map_nk:.4f} | mAP50: {map50_nk:.4f} | Loss: {avg_nk_loss:.4f}")
    print(f"Combined mAP: {combined_map:.4f} | mAP50: {combined_map50:.4f}")
    print("-"*70)
    print(f"Key Latency: {avg_k_time:.2f} ms | Key VRAM: {avg_k_mem:.2f} MB")
    if not args.baseline:
        print(f"Non-Key Latency: {avg_nk_time:.2f} ms | Non-Key VRAM: {avg_nk_mem:.2f} MB")
        if avg_nk_time > 0:
            print(f"Speedup (Key/Non-Key): {avg_k_time / avg_nk_time:.2f}x")
    print("="*70)

    # Diagnostics Summary
    print("\n" + "="*70)
    print("DIAGNOSTICS SUMMARY")
    print("="*70)
    print(f"  Key Path:  Avg IoU: {np.mean(key_ious):.4f} | Avg Conf: {np.mean(key_confs):.4f}")
    print(f"  NK Path:   Avg IoU: {np.mean(nk_ious):.4f} | Avg Conf: {np.mean(nk_confs):.4f}")
    
    def get_sep(tp, fp):
        if not tp or not fp: return 0.0
        return np.mean(tp) - np.mean(fp)

    print(f"\nTP/FP Score Separation (mean_tp - mean_fp):")
    print(f"  Key Path:  {get_sep(key_tp_scores, key_fp_scores):.4f}")
    print(f"  NK Path:   {get_sep(nk_tp_scores, nk_fp_scores):.4f}")
    
    print(f"\nDetailed Loss Analysis (Raw Criterion Values):")
    if loss_stats['key']['class']:
        print(f"  Key Loss:  Class: {np.mean(loss_stats['key']['class']):.4f} | Box: {np.mean(loss_stats['key']['box']):.4f}")
    if loss_stats['nk']['class']:
        print(f"  NK Loss:   Class: {np.mean(loss_stats['nk']['class']):.4f} | Box: {np.mean(loss_stats['nk']['box']):.4f}")
    print("="*70)

if __name__ == '__main__':
    main()
