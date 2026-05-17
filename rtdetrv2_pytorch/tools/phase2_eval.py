import argparse
import os
import sys
import time
import contextlib
import io
from typing import Any, Dict, List, Set

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.zoo.temporal_rtdetr import TemporalRTDETR
from src.zoo.rtdetr.box_ops import box_iou, box_cxcywh_to_xyxy
from pycocotools.cocoeval import COCOeval


def extract_video_id(file_name: str) -> str:
    """Extract video ID from filename (matches ViratTemporalDataset logic)"""
    parts = os.path.normpath(file_name).split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return "default_video"


def _build_temporal_model(cfg: Any, device: torch.device) -> Any:
    from src.zoo.temporal_rtdetr import TemporalRTDETR

    base_model = cfg.model.to(device)
    backbone = base_model.backbone
    encoder = getattr(base_model, 'encoder', None)
    decoder = getattr(base_model, 'decoder', None)

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

    return TemporalRTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        num_classes=cfg.yaml_cfg.get('num_classes', 80),
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        use_lightweight_decoder=cfg.yaml_cfg.get('use_lightweight_decoder', True),
        reuse_position=cfg.yaml_cfg.get('reuse_position', 0),
        enable_apg=cfg.yaml_cfg.get('enable_apg', True),
        apg_in_channels=cfg.yaml_cfg.get('apg_in_channels', 512),
        apg_hidden_channels=cfg.yaml_cfg.get('apg_hidden_channels', 64),
        apg_pool_size=cfg.yaml_cfg.get('apg_pool_size', 4),
    ).to(device)


def _load_checkpoint(model: Any, ckpt_path: str, device: torch.device) -> None:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))

    # --- AUTO-DECOUPLE DETECTION ---
    is_decoupled = any('lightweight_decoder.dec_score_head' in k for k in state_dict.keys())
    if is_decoupled and hasattr(model, 'decouple_non_key_prediction_heads'):
        print("   [Auto-Detect] Decoupled prediction heads found in checkpoint. Decoupling model...")
        model.decouple_non_key_prediction_heads()

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded checkpoint from {ckpt_path}')
    print(f'  Missing keys: {len(missing)}')
    print(f'  Unexpected keys: {len(unexpected)}')


def _extract_total_loss(loss_dict: Dict[str, torch.Tensor]) -> float:
    """Extracts main detection loss, ignoring auxiliary and denoising."""
    relevant_keys = [k for k in loss_dict.keys() if not any(x in k for x in ['_aux_', '_dn_', '_enc_'])]
    if not relevant_keys:
        return 0.0
    return sum(loss_dict[k] for k in relevant_keys).item()


def _extract_loss_components(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Extract class/box loss components from criterion outputs."""
    class_keys = []
    box_keys = []
    for k in loss_dict.keys():
        if any(x in k for x in ['_aux_', '_dn_', '_enc_']):
            continue
        if k.startswith('loss_vfl') or k.startswith('loss_focal') or k.startswith('loss_class'):
            class_keys.append(k)
        if k.startswith('loss_bbox') or k.startswith('loss_giou') or k.startswith('loss_l1'):
            box_keys.append(k)

    class_loss = sum(loss_dict[k] for k in class_keys).item() if class_keys else 0.0
    box_loss = sum(loss_dict[k] for k in box_keys).item() if box_keys else 0.0
    return {'class': class_loss, 'box': box_loss}


def _mean_or_zero(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _prepare_targets_for_loss(targets, device):
    """
    Criterion expects normalized cxcywh boxes in target['boxes'].
    """
    prepared = []
    for t in targets:
        nt = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        boxes = nt['boxes']
        if boxes.numel() == 0:
            prepared.append(nt)
            continue

        is_normalized = (boxes <= 1.01).all()
        if not is_normalized:
            h, w = nt['orig_size'][0], nt['orig_size'][1]
            boxes = boxes.clone()
            boxes[:, 2:] -= boxes[:, :2]  # xyxy -> xywh
            boxes[:, :2] += boxes[:, 2:] / 2  # xywh -> cxcywh
            boxes /= torch.tensor([w, h, w, h], device=device)
            nt['boxes'] = boxes
        prepared.append(nt)
    return prepared


def _record_iou_conf_stats(results, targets, iou_list, conf_list, score_thr, device):
    """
    Record best-match IoU and confidence per GT for a single-image result batch.
    """
    keep = results['scores'] > score_thr
    pred_boxes = results['boxes'][keep]
    pred_scores = results['scores'][keep]

    target = targets[0]
    gt_boxes_raw = target['boxes']
    if gt_boxes_raw.numel() == 0:
        return

    h, w = target['orig_size'][0], target['orig_size'][1]
    is_normalized = (gt_boxes_raw <= 1.01).all()
    if is_normalized:
        gt_boxes_abs = gt_boxes_raw.to(device) * torch.tensor([w, h, w, h], device=device)
        gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes_abs)
    else:
        gt_boxes_xyxy = gt_boxes_raw.to(device)

    if pred_boxes.numel() == 0:
        iou_list.extend([0.0] * gt_boxes_xyxy.shape[0])
        conf_list.extend([0.0] * gt_boxes_xyxy.shape[0])
        return

    ious, _ = box_iou(gt_boxes_xyxy, pred_boxes)
    best_iou_vals, best_indices = ious.max(dim=1)
    iou_list.extend(best_iou_vals.cpu().numpy().tolist())
    conf_list.extend(pred_scores[best_indices].cpu().numpy().tolist())


def _record_tp_fp_scores(results, targets, tp_scores, fp_scores, score_thr, device, iou_thr=0.5):
    """
    Record TP/FP scores by class-matched IoU assignment for a single-image result batch.
    """
    keep = results['scores'] > score_thr
    pred_boxes = results['boxes'][keep]
    pred_scores = results['scores'][keep]
    pred_labels = results['labels'][keep]

    target = targets[0]
    gt_boxes_raw = target['boxes']
    gt_labels = target['labels']

    if pred_boxes.numel() == 0:
        return
    if gt_boxes_raw.numel() == 0:
        fp_scores.extend(pred_scores.cpu().numpy().tolist())
        return

    h, w = target['orig_size'][0], target['orig_size'][1]
    is_normalized = (gt_boxes_raw <= 1.01).all()
    if is_normalized:
        gt_boxes_abs = gt_boxes_raw.to(device) * torch.tensor([w, h, w, h], device=device)
        gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes_abs)
    else:
        gt_boxes_xyxy = gt_boxes_raw.to(device)

    ious, _ = box_iou(gt_boxes_xyxy, pred_boxes)
    pred_scores_np = pred_scores.cpu().numpy()
    pred_labels_np = pred_labels.cpu().numpy()
    gt_labels_np = gt_labels.cpu().numpy()
    ious_np = ious.cpu().numpy()

    indices = np.argsort(-pred_scores_np)
    matched_gt = np.zeros(gt_boxes_xyxy.shape[0], dtype=bool)

    for idx in indices:
        label = pred_labels_np[idx]
        best_iou = -1.0
        best_gt_idx = -1
        for g_idx in range(gt_boxes_xyxy.shape[0]):
            if gt_labels_np[g_idx] == label:
                iou = ious_np[g_idx, idx]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

        if best_gt_idx >= 0 and best_iou >= iou_thr and not matched_gt[best_gt_idx]:
            tp_scores.append(float(pred_scores_np[idx]))
            matched_gt[best_gt_idx] = True
        else:
            fp_scores.append(float(pred_scores_np[idx]))


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


def evaluate_map(coco_gt, results, img_ids=None):
    """Runs pycocotools evaluation and returns (mAP, mAP50, mAP75)."""
    if not results and not img_ids:
        return 0.0, 0.0, 0.0
        
    if not results:
        coco_dt = coco_gt.loadRes([])
    else:
        coco_dt = coco_gt.loadRes(results)
        
    evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
    
    if img_ids is not None:
        evaluator.params.imgIds = sorted(list(img_ids))
    else:
        predicted_img_ids = sorted(list(set([res['image_id'] for res in results])))
        evaluator.params.imgIds = predicted_img_ids
    
    evaluator.evaluate()
    evaluator.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.summarize()
    if len(evaluator.stats) < 3:
        return 0.0, 0.0, 0.0
    return float(evaluator.stats[0]), float(evaluator.stats[1]), float(evaluator.stats[2])


@torch.no_grad()
def evaluate_apg_stream(
    model: Any,
    cfg: Any,
    threshold: float,
    nk_per_key: int,
    device: torch.device,
    args: argparse.Namespace
) -> Dict[str, Any]:
    model.eval()
    if model.apg is None:
        raise RuntimeError('APG is disabled. Set enable_apg=True in config/checkpoint.')

    # --- DATALOADER REBUILD FOR STREAM SIMULATION ---
    if 'val_dataloader' in cfg.yaml_cfg:
        cfg.yaml_cfg['val_dataloader']['batch_size'] = 1
        cfg.yaml_cfg['val_dataloader']['drop_last'] = False
        cfg.yaml_cfg['val_dataloader']['shuffle'] = False
        if 'dataset' in cfg.yaml_cfg['val_dataloader']:
            cfg.yaml_cfg['val_dataloader']['dataset']['max_frame_gap'] = 1
            cfg.yaml_cfg['val_dataloader']['dataset']['frame_stride'] = 1
            cfg.yaml_cfg['val_dataloader']['dataset']['pair_sampling_strategy'] = 'all'

    from torch.utils.data import DataLoader
    from src.data.transforms import ConvertBoxes, SanitizeBoundingBoxes
    
    base_val_loader = cfg.val_dataloader
    # Add necessary box conversions for criterion compatibility
    # These won't affect COCOeval as it uses image_id to look up ground truth
    base_val_loader.dataset.transforms.transforms.append(SanitizeBoundingBoxes(min_size=1))
    base_val_loader.dataset.transforms.transforms.append(ConvertBoxes(fmt='cxcywh', normalize=True))

    val_dataloader = DataLoader(
        dataset=base_val_loader.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=base_val_loader.num_workers,
        collate_fn=base_val_loader.collate_fn,
        drop_last=False
    )
    
    coco_gt = val_dataloader.dataset.coco
    postprocessor = cfg.postprocessor
    criterion = cfg.criterion
    criterion.eval()
    
    res_key = []
    res_nk = []
    eval_img_ids_key = set()
    eval_img_ids_nk = set()

    diag_stats = {
        'key': {
            'ious': [],
            'confs': [],
            'tp_scores': [],
            'fp_scores': [],
            'class_losses': [],
            'box_losses': [],
        },
        'nk': {
            'ious': [],
            'confs': [],
            'tp_scores': [],
            'fp_scores': [],
            'class_losses': [],
            'box_losses': [],
        },
    }
    
    total_key_loss = 0.0
    total_nk_loss = 0.0
    num_key_evals = 0
    num_nk_evals = 0

    key_decisions = 0
    forced_key_decisions = 0
    apg_key_votes = 0
    total_frames = 0
    nk_since_key = 0
    last_video_id = None

    print(f"\n--- INITIATING APG STREAM SIMULATION (Threshold: {threshold} | Max NK: {nk_per_key}) ---")

    for i, batch in enumerate(tqdm(val_dataloader, desc="Streaming Video")):
        img_key, target_key, img_non_key, target_non_key = batch
        
        # Video boundary detection
        img_id_k = int(target_key[0]['image_id'].item())
        img_info_k = val_dataloader.dataset.img_id_to_info[img_id_k]
        current_video_id = extract_video_id(img_info_k['file_name'])
        
        # 1. HANDLE START OF VIDEO (Initial Key Frame F0)
        if last_video_id is None or current_video_id != last_video_id:
            img_key = img_key.to(device)
            target_key = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_key]
            target_key_loss = _prepare_targets_for_loss(target_key, device)
            out_k = model.forward_key_frame(img_key, None)
            
            # Loss tracking
            loss_dict = criterion(out_k, target_key_loss)
            total_key_loss += _extract_total_loss(loss_dict)
            num_key_evals += 1
            loss_components = _extract_loss_components(loss_dict)
            diag_stats['key']['class_losses'].append(loss_components['class'])
            diag_stats['key']['box_losses'].append(loss_components['box'])

            orig_sizes_k = torch.stack([t["orig_size"] for t in target_key], dim=0).to(device)
            res_k_batch = postprocessor(out_k, orig_sizes_k)
            format_coco(target_key, res_k_batch, res_key)
            _record_iou_conf_stats(
                res_k_batch[0], target_key_loss,
                diag_stats['key']['ious'], diag_stats['key']['confs'],
                score_thr=args.diag_score_thr, device=device,
            )
            _record_tp_fp_scores(
                res_k_batch[0], target_key_loss,
                diag_stats['key']['tp_scores'], diag_stats['key']['fp_scores'],
                score_thr=args.diag_score_thr, device=device, iou_thr=args.diag_iou_thr,
            )
            for t in target_key:
                eval_img_ids_key.add(int(t['image_id'].item()))
            
            key_decisions += 1
            total_frames += 1
            nk_since_key = 0
            last_video_id = current_video_id

        # 2. EVALUATE CURRENT FRAME (img_non_key, which is F_{i+1})
        cur_img = img_non_key.to(device)
        cur_target = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_non_key]
        cur_target_loss = _prepare_targets_for_loss(cur_target, device)
        
        prev_key_s5 = model.cached_key_s5
        cur_s5 = model.extract_s5(cur_img)
        _, prob = model.forward_apg(prev_key_s5, cur_s5)
        apg_vote_key = bool(prob.item() > threshold)
        apg_key_votes += int(apg_vote_key)

        force_key = nk_since_key >= nk_per_key
        use_key = force_key or apg_vote_key

        if use_key:
            if force_key:
                forced_key_decisions += 1
            key_decisions += 1
            out = model.forward_key_frame(cur_img, None)
            nk_since_key = 0
            is_key_frame = True
            
            # Loss tracking
            loss_dict = criterion(out, cur_target_loss)
            total_key_loss += _extract_total_loss(loss_dict)
            num_key_evals += 1
            loss_components = _extract_loss_components(loss_dict)
            diag_stats['key']['class_losses'].append(loss_components['class'])
            diag_stats['key']['box_losses'].append(loss_components['box'])
        else:
            out = model.forward_non_key_frame(cur_img, None)
            nk_since_key += 1
            is_key_frame = False
            
            # Loss tracking
            loss_dict = criterion(out, cur_target_loss)
            total_nk_loss += _extract_total_loss(loss_dict)
            num_nk_evals += 1
            loss_components = _extract_loss_components(loss_dict)
            diag_stats['nk']['class_losses'].append(loss_components['class'])
            diag_stats['nk']['box_losses'].append(loss_components['box'])

        total_frames += 1
        orig_sizes = torch.stack([t["orig_size"] for t in cur_target], dim=0).to(device)
        res_batch = postprocessor(out, orig_sizes)
        
        if is_key_frame:
            format_coco(cur_target, res_batch, res_key)
            _record_iou_conf_stats(
                res_batch[0], cur_target_loss,
                diag_stats['key']['ious'], diag_stats['key']['confs'],
                score_thr=args.diag_score_thr, device=device,
            )
            _record_tp_fp_scores(
                res_batch[0], cur_target_loss,
                diag_stats['key']['tp_scores'], diag_stats['key']['fp_scores'],
                score_thr=args.diag_score_thr, device=device, iou_thr=args.diag_iou_thr,
            )
            for t in cur_target:
                eval_img_ids_key.add(int(t['image_id'].item()))
        else:
            format_coco(cur_target, res_batch, res_nk)
            _record_iou_conf_stats(
                res_batch[0], cur_target_loss,
                diag_stats['nk']['ious'], diag_stats['nk']['confs'],
                score_thr=args.diag_score_thr, device=device,
            )
            _record_tp_fp_scores(
                res_batch[0], cur_target_loss,
                diag_stats['nk']['tp_scores'], diag_stats['nk']['fp_scores'],
                score_thr=args.diag_score_thr, device=device, iou_thr=args.diag_iou_thr,
            )
            for t in cur_target:
                eval_img_ids_nk.add(int(t['image_id'].item()))

    # --- SCORE TUNING & COMBINATION ---
    nonkey_scale = args.nonkey_score
    combined_img_ids = eval_img_ids_key | eval_img_ids_nk

    if args.tune_score:
        print("\nTuning confidence scores...")
        grid = [float(x.strip()) for x in args.score_grid.split(',') if x.strip()]
        best = None
        for ns in grid:
            scaled_nk = scale_results(res_nk, ns)
            filtered_nk = [det for det in scaled_nk if det['image_id'] not in eval_img_ids_key]
            
            m_ap, m_ap50, _ = evaluate_map(coco_gt, res_key + filtered_nk, combined_img_ids)
            score = (m_ap, m_ap50)
            if best is None or score > best['score']:
                best = {'ns': ns, 'score': score}
        nonkey_scale = best['ns']
        print(f"Best non-key scale: {nonkey_scale:.3f}")

    scaled_res_nk = scale_results(res_nk, nonkey_scale)
    final_filtered_nk = [det for det in scaled_res_nk if det['image_id'] not in eval_img_ids_key]
    
    map_k, map50_k, _ = evaluate_map(coco_gt, res_key, eval_img_ids_key)
    map_nk, map50_nk, _ = evaluate_map(coco_gt, scaled_res_nk, eval_img_ids_nk)
    c_map, c_map50, c_map75 = evaluate_map(coco_gt, res_key + final_filtered_nk, combined_img_ids)

    key_ratio = key_decisions / max(1, total_frames)

    def _tp_fp_sep(tp_scores: List[float], fp_scores: List[float]) -> float:
        if not tp_scores or not fp_scores:
            return 0.0
        return float(np.mean(tp_scores) - np.mean(fp_scores))
    
    return {
        'key_mAP': map_k, 'key_mAP50': map50_k,
        'nk_mAP': map_nk, 'nk_mAP50': map50_nk,
        'combined_mAP': c_map, 'combined_mAP50': c_map50, 'combined_mAP75': c_map75,
        'key_ratio': key_ratio,
        'apg_key_vote_ratio': apg_key_votes / max(1, total_frames),
        'forced_key_ratio': forced_key_decisions / max(1, total_frames),
        'avg_interval': 1.0 / key_ratio if key_ratio > 0 else float('inf'),
        'total_frames': total_frames,
        'nonkey_score': nonkey_scale,
        'avg_key_loss': total_key_loss / max(1, num_key_evals),
        'avg_nk_loss': total_nk_loss / max(1, num_nk_evals),
        'avg_key_iou': _mean_or_zero(diag_stats['key']['ious']),
        'avg_nk_iou': _mean_or_zero(diag_stats['nk']['ious']),
        'avg_key_conf': _mean_or_zero(diag_stats['key']['confs']),
        'avg_nk_conf': _mean_or_zero(diag_stats['nk']['confs']),
        'key_score_sep': _tp_fp_sep(diag_stats['key']['tp_scores'], diag_stats['key']['fp_scores']),
        'nk_score_sep': _tp_fp_sep(diag_stats['nk']['tp_scores'], diag_stats['nk']['fp_scores']),
        'avg_key_class_loss': _mean_or_zero(diag_stats['key']['class_losses']),
        'avg_nk_class_loss': _mean_or_zero(diag_stats['nk']['class_losses']),
        'avg_key_box_loss': _mean_or_zero(diag_stats['key']['box_losses']),
        'avg_nk_box_loss': _mean_or_zero(diag_stats['nk']['box_losses']),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate APG-routed Temporal RT-DETR")
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--checkpoint', '-w', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.5, help='APG probability threshold')
    parser.add_argument('--nk_per_key', '-n', type=int, default=10, 
                        help='Hard max number of non-key frames before forcing a key frame.')
    parser.add_argument('--nonkey_score', '-ns', type=float, default=1.0)
    parser.add_argument('--tune_score', '-ts', action='store_true')
    parser.add_argument('--score_grid', type=str, default='1.0,1.02,1.04,1.06,1.08,1.10,1.12,1.14,1.16,1.18,1.20')
    parser.add_argument('--diag_score_thr', type=float, default=-1.0, help='Score threshold for IoU/conf and TP/FP diagnostics. Use <0 for no filtering (default).')
    parser.add_argument('--diag_iou_thr', type=float, default=0.5, help='IoU threshold for TP/FP assignment in diagnostics.')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = YAMLConfig(args.config)
    model = _build_temporal_model(cfg, device)
    _load_checkpoint(model, args.checkpoint, device)

    stats = evaluate_apg_stream(
        model=model,
        cfg=cfg,
        threshold=args.threshold,
        nk_per_key=args.nk_per_key,
        device=device,
        args=args
    )

    print("\n" + "="*70)
    print(f"APG STREAM EVALUATION SUMMARY (Threshold: {args.threshold} | Max NK: {args.nk_per_key})")
    print("="*70)
    print(f"Score scale -> non-key: {stats['nonkey_score']:.3f}")
    print(f"Key Only    mAP: {stats['key_mAP']:.4f} | mAP50: {stats['key_mAP50']:.4f} | Loss: {stats['avg_key_loss']:.4f}")
    print(f"Non-Key Only mAP: {stats['nk_mAP']:.4f} | mAP50: {stats['nk_mAP50']:.4f} | Loss: {stats['avg_nk_loss']:.4f}")
    print(f"Combined     mAP: {stats['combined_mAP']:.4f} | mAP50: {stats['combined_mAP50']:.4f}")
    print("-"*70)
    print("Diagnostics (APG-routed):")
    print(f"  Key Path:  Avg IoU: {stats['avg_key_iou']:.4f} | Avg Conf: {stats['avg_key_conf']:.4f}")
    print(f"  NK Path:   Avg IoU: {stats['avg_nk_iou']:.4f} | Avg Conf: {stats['avg_nk_conf']:.4f}")
    print("\nTP/FP Score Separation (mean_tp - mean_fp):")
    print(f"  Key Path:  {stats['key_score_sep']:.4f}")
    print(f"  NK Path:   {stats['nk_score_sep']:.4f}")
    print("\nLoss Analysis (Raw Criterion Values):")
    print(f"  Key Loss:  Class: {stats['avg_key_class_loss']:.4f} | Box: {stats['avg_key_box_loss']:.4f}")
    print(f"  NK Loss:   Class: {stats['avg_nk_class_loss']:.4f} | Box: {stats['avg_nk_box_loss']:.4f}")
    print("-"*70)
    print(f"Key Ratio: {stats['key_ratio']:.4f} ({stats['key_ratio']*100:.2f}%)")
    print(f"  - APG Votes: {stats['apg_key_vote_ratio']*100:.2f}%")
    print(f"  - Forced (Guard Rail): {stats['forced_key_ratio']*100:.2f}%")
    print(f"Avg Key Interval: {stats['avg_interval']:.2f} frames")
    print(f"Total Frames: {stats['total_frames']}")
    print("="*70)


if __name__ == '__main__':
    main()
