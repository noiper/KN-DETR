import argparse
import os
import sys
from typing import Any, Dict, List, Set

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def extract_video_id(file_name: str) -> str:
    parts = os.path.normpath(file_name).split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return "default_video"


def _build_temporal_model(cfg: Any, device: torch.device) -> Any:
    from src.zoo.temporal_rtdetr import TemporalRTDETR

    base_model = cfg.model.to(device)
    backbone = base_model.backbone
    encoder = base_model.encoder
    decoder = base_model.decoder

    hidden_dim = 256
    num_queries = 300
    if 'RTDETRTransformerv2' in cfg.yaml_cfg:
        decoder_cfg = cfg.yaml_cfg['RTDETRTransformerv2']
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

    has_non_key_prediction_heads = any(
        k.startswith('lightweight_decoder.dec_score_head')
        or k.startswith('lightweight_decoder.dec_bbox_head')
        or k.startswith('lightweight_decoder.query_pos_head')
        for k in state_dict.keys()
    )
    if has_non_key_prediction_heads and hasattr(model, 'lightweight_decoder') and model.lightweight_decoder is not None:
        model.decouple_non_key_prediction_heads()

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded checkpoint from {ckpt_path}')
    print(f'  Missing keys: {len(missing)}')
    print(f'  Unexpected keys: {len(unexpected)}')


def _build_stream_val_dataloader(cfg: Any):
    from torch.utils.data import DataLoader

    if 'val_dataloader' in cfg.yaml_cfg:
        cfg.yaml_cfg['val_dataloader']['batch_size'] = 1
        cfg.yaml_cfg['val_dataloader']['drop_last'] = False
        cfg.yaml_cfg['val_dataloader']['shuffle'] = False

        if 'dataset' in cfg.yaml_cfg['val_dataloader']:
            # Force sequential stream semantics for eval, independent of phase-1 train settings.
            cfg.yaml_cfg['val_dataloader']['dataset']['max_frame_gap'] = 1
            cfg.yaml_cfg['val_dataloader']['dataset']['frame_stride'] = 1
            cfg.yaml_cfg['val_dataloader']['dataset']['pair_sampling_strategy'] = 'all'

    base_val_loader = cfg.val_dataloader
    return DataLoader(
        dataset=base_val_loader.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=base_val_loader.num_workers,
        collate_fn=base_val_loader.collate_fn,
        drop_last=False,
    )




def _scale_results(results: List[Dict], score_scale: float) -> List[Dict]:
    """Scale detection scores by a factor, clamping to [0, 1]."""
    if score_scale == 1.0:
        return results
    scaled = []
    for det in results:
        score = max(0.0, min(1.0, float(det['score']) * score_scale))
        out = det.copy()
        out['score'] = score
        scaled.append(out)
    return scaled


def _run_coco_eval(coco_gt, results_list: List[Dict], eval_img_ids: Set[int]) -> Dict[str, float]:
    from pycocotools.cocoeval import COCOeval
    import contextlib
    import io

    if not results_list:
        return {'mAP': 0.0, 'mAP@50': 0.0, 'mAP@75': 0.0}

    coco_dt = coco_gt.loadRes(results_list)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = sorted(eval_img_ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    # Silence the default table output from summarize()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()
    return {
        'mAP': float(coco_eval.stats[0]),
        'mAP@50': float(coco_eval.stats[1]),
        'mAP@75': float(coco_eval.stats[2]),
    }


@torch.no_grad()
def evaluate_apg_stream(
    model: Any,
    val_loader,
    postprocessor,
    threshold: float,
    nk_per_key: int,
    device: torch.device,
    key_score: float = 1.0,
    nonkey_score: float = 1.0,
    tune_score: bool = False,
    score_grid: List[float] = None,
) -> Dict[str, float]:
    model.eval()
    if model.apg is None:
        raise RuntimeError('APG is disabled. Set enable_apg=True in config/checkpoint.')
    if nk_per_key < 0:
        raise ValueError(f'nk_per_key must be >= 0, got {nk_per_key}')
    if not hasattr(val_loader.dataset, 'coco'):
        raise RuntimeError('Validation dataset must expose .coco for COCO evaluation.')
    
    if score_grid is None:
        score_grid = [0.8, 0.9, 1.0, 1.1, 1.2]

    coco_gt = val_loader.dataset.coco
    results_key = []
    results_nk = []
    eval_img_ids_key: Set[int] = set()
    eval_img_ids_nk: Set[int] = set()

    key_decisions = 0
    forced_key_decisions = 0
    apg_key_votes = 0
    total_frames = 0
    nk_since_key = 0
    has_key_cache = False
    last_video_id = None

    for batch_idx, (img_key, target_key, img_non_key, target_non_key) in enumerate(val_loader):
        img_key = img_key.to(device)
        img_non_key = img_non_key.to(device)

        # Batch is forced to 1 in stream mode, but keep code robust to any batch size.
        batch_size = img_non_key.shape[0]
        routed_outputs: List[Dict] = []
        routed_targets: List[Dict] = []
        routed_is_key: List[bool] = []

        for b in range(batch_size):
            key_img = img_key[b:b + 1]
            cur_img = img_non_key[b:b + 1]
            cur_target = target_non_key[b]

            key_target = target_key[b]
            key_img_id = int(key_target['image_id'].item())
            key_img_info = val_loader.dataset.img_id_to_info[key_img_id]
            current_video_id = extract_video_id(key_img_info['file_name'])

            if (last_video_id is not None and current_video_id != last_video_id) or not has_key_cache:
                # Force cache reset at video boundary/start.
                model.forward_key_frame(key_img, None)
                nk_since_key = 0
                has_key_cache = True
            last_video_id = current_video_id

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
                outputs = model.forward_key_frame(cur_img, None)
                nk_since_key = 0
                is_key = True
            else:
                outputs = model.forward_non_key_frame(cur_img, None)
                nk_since_key += 1
                is_key = False

            total_frames += 1
            routed_outputs.append(outputs)
            routed_targets.append(cur_target)
            routed_is_key.append(is_key)

        orig_sizes = torch.stack([t['orig_size'] for t in routed_targets], dim=0).to(device)
        processed = postprocessor(
            {
                'pred_logits': torch.cat([o['pred_logits'] for o in routed_outputs], dim=0),
                'pred_boxes': torch.cat([o['pred_boxes'] for o in routed_outputs], dim=0),
            },
            orig_sizes,
        )
        
        # Accumulate key and non-key results separately
        for target, output, is_key_frame in zip(routed_targets, processed, routed_is_key):
            image_id = int(target['image_id'].item())
            
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()

            for i in range(len(scores)):
                x1, y1, x2, y2 = boxes[i]
                w, h = x2 - x1, y2 - y1
                det = {
                    'image_id': image_id,
                    'category_id': int(labels[i]),
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(scores[i]),
                }
                
                if is_key_frame:
                    results_key.append(det)
                    if image_id not in eval_img_ids_key:
                        eval_img_ids_key.add(image_id)
                else:
                    results_nk.append(det)
                    if image_id not in eval_img_ids_nk:
                        eval_img_ids_nk.add(image_id)

        if batch_idx % 100 == 0:
            print(f'Processed {batch_idx}/{len(val_loader)} batches')

    # Compute mAP for key and non-key separately
    stats_key = _run_coco_eval(coco_gt, results_key, eval_img_ids_key)
    stats_nk = _run_coco_eval(coco_gt, results_nk, eval_img_ids_nk)
    
    # Confidence tuning via grid search
    if tune_score:
        best = None
        for ks in score_grid:
            for ns in score_grid:
                scaled_key = _scale_results(results_key, ks)
                scaled_nk = _scale_results(results_nk, ns)
                
                # Filter out overlapping image IDs from non-key results
                filtered_nk = [det for det in scaled_nk if det['image_id'] not in eval_img_ids_key]
                combined_results = scaled_key + filtered_nk
                combined_img_ids = eval_img_ids_key | eval_img_ids_nk
                
                stats_combined = _run_coco_eval(coco_gt, combined_results, combined_img_ids)
                score = (stats_combined['mAP'], stats_combined['mAP@50'])
                
                if best is None or score > best['score']:
                    best = {
                        'key_scale': ks,
                        'nonkey_scale': ns,
                        'combined': stats_combined,
                        'score': score,
                    }
        
        key_score = best['key_scale']
        nonkey_score = best['nonkey_scale']
        print(f"Tuned score scales: key={key_score:.3f}, non-key={nonkey_score:.3f}")
    
    # Apply final scaling
    scaled_key = _scale_results(results_key, key_score)
    scaled_nk = _scale_results(results_nk, nonkey_score)
    
    # Compute final combined mAP
    filtered_nk = [det for det in scaled_nk if det['image_id'] not in eval_img_ids_key]
    combined_results = scaled_key + filtered_nk
    combined_img_ids = eval_img_ids_key | eval_img_ids_nk
    stats_combined = _run_coco_eval(coco_gt, combined_results, combined_img_ids)
    
    key_ratio = key_decisions / max(1, total_frames)
    apg_key_vote_ratio = apg_key_votes / max(1, total_frames)
    forced_key_ratio = forced_key_decisions / max(1, total_frames)

    return {
        # Key path metrics
        'key_mAP': stats_key['mAP'],
        'key_mAP@50': stats_key['mAP@50'],
        'key_mAP@75': stats_key['mAP@75'],
        # Non-key path metrics
        'nk_mAP': stats_nk['mAP'],
        'nk_mAP@50': stats_nk['mAP@50'],
        'nk_mAP@75': stats_nk['mAP@75'],
        # Combined metrics (primary reporting)
        'combined_mAP': stats_combined['mAP'],
        'combined_mAP@50': stats_combined['mAP@50'],
        'combined_mAP@75': stats_combined['mAP@75'],
        # Routing metrics
        'key_ratio': key_ratio,
        'apg_key_vote_ratio': apg_key_vote_ratio,
        'forced_key_ratio': forced_key_ratio,
        'avg_interval_proxy': (1.0 / key_ratio) if key_ratio > 0 else float('inf'),
        'nk_per_key': float(nk_per_key),
        # Score scales
        'key_score_scale': key_score,
        'nonkey_score_scale': nonkey_score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--checkpoint', '-w', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument(
        '--nk_per_key',
        '-n',
        type=int,
        default=10,
        help='Hard max number of non-key frames between two key frames. Once reached, force key regardless of APG.',
    )
    parser.add_argument(
        '--key_score',
        '-ks',
        type=float,
        default=1.0,
        help='Multiply key-path confidence scores by this factor before evaluation',
    )
    parser.add_argument(
        '--nonkey_score',
        '-ns',
        type=float,
        default=1.0,
        help='Multiply non-key-path confidence scores by this factor before evaluation',
    )
    parser.add_argument(
        '--tune_score',
        '-ts',
        action='store_true',
        help='Grid search key/non-key score scales for best combined mAP',
    )
    parser.add_argument(
        '--score_grid',
        type=str,
        default='0.8,0.9,1.0,1.1,1.2',
        help='Comma-separated scale grid for --tune_score',
    )
    args = parser.parse_args()
    
    from src.core import YAMLConfig

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = YAMLConfig(args.config)
    model = _build_temporal_model(cfg, device)
    _load_checkpoint(model, args.checkpoint, device)
    val_loader = _build_stream_val_dataloader(cfg)
    postprocessor = cfg.postprocessor
    
    # Parse score grid
    score_grid = [float(x.strip()) for x in args.score_grid.split(',') if x.strip()]

    stats = evaluate_apg_stream(
        model=model,
        val_loader=val_loader,
        postprocessor=postprocessor,
        threshold=args.threshold,
        nk_per_key=args.nk_per_key,
        device=device,
        key_score=args.key_score,
        nonkey_score=args.nonkey_score,
        tune_score=args.tune_score,
        score_grid=score_grid,
    )
    print('\nAPG Stream Evaluation (Combined mAP):\n')
    print('Key Path:')
    print(f'  mAP: {stats["key_mAP"]:.6f} | mAP@50: {stats["key_mAP@50"]:.6f} | mAP@75: {stats["key_mAP@75"]:.6f}')
    print('Non-Key Path:')
    print(f'  mAP: {stats["nk_mAP"]:.6f} | mAP@50: {stats["nk_mAP@50"]:.6f} | mAP@75: {stats["nk_mAP@75"]:.6f}')
    print('Combined (Primary):')
    print(f'  mAP: {stats["combined_mAP"]:.6f} | mAP@50: {stats["combined_mAP@50"]:.6f} | mAP@75: {stats["combined_mAP@75"]:.6f}')
    print('Routing & Scaling:')
    for k, v in stats.items():
        if k not in ['key_mAP', 'key_mAP@50', 'key_mAP@75', 'nk_mAP', 'nk_mAP@50', 'nk_mAP@75', 'combined_mAP', 'combined_mAP@50', 'combined_mAP@75']:
            if isinstance(v, float):
                print(f'  {k}: {v:.6f}')
            else:
                print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
