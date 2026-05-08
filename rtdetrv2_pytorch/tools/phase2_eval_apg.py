import argparse
import os
import sys
from typing import Any, Dict, List, Set

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


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
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded checkpoint from {ckpt_path}')
    print(f'  Missing keys: {len(missing)}')
    print(f'  Unexpected keys: {len(unexpected)}')


def _accumulate(results_list: List[Dict], targets: List[Dict], outputs: List[Dict], seen_ids: Set[int]) -> None:
    for target, output in zip(targets, outputs):
        image_id = int(target['image_id'].item())
        if image_id in seen_ids:
            continue
        seen_ids.add(image_id)

        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()

        for i in range(len(scores)):
            x1, y1, x2, y2 = boxes[i]
            w, h = x2 - x1, y2 - y1
            results_list.append({
                'image_id': image_id,
                'category_id': int(labels[i]),
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'score': float(scores[i]),
            })


def _run_coco_eval(coco_gt, results_list: List[Dict], eval_img_ids: Set[int]) -> Dict[str, float]:
    from pycocotools.cocoeval import COCOeval

    if not results_list:
        return {'mAP': 0.0, 'mAP@50': 0.0, 'mAP@75': 0.0}

    coco_dt = coco_gt.loadRes(results_list)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = sorted(eval_img_ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        'mAP': float(coco_eval.stats[0]),
        'mAP@50': float(coco_eval.stats[1]),
        'mAP@75': float(coco_eval.stats[2]),
    }


@torch.no_grad()
def evaluate_apg(model: Any, cfg: Any, threshold: float, device: torch.device) -> Dict[str, float]:
    model.eval()
    if model.apg is None:
        raise RuntimeError('APG is disabled. Set enable_apg=True in config/checkpoint.')

    val_loader = cfg.val_dataloader
    postprocessor = cfg.postprocessor
    if not hasattr(val_loader.dataset, 'coco'):
        raise RuntimeError('Validation dataset must expose .coco for COCO evaluation.')
    coco_gt = val_loader.dataset.coco

    results = []
    eval_img_ids: Set[int] = set()
    key_decisions = 0
    total_frames = 0

    for batch_idx, (img_key, target_key, img_non_key, target_non_key) in enumerate(val_loader):
        img_key = img_key.to(device)
        img_non_key = img_non_key.to(device)

        batch_size = img_key.shape[0]
        routed_outputs: List[Dict] = []
        routed_targets: List[Dict] = []

        for b in range(batch_size):
            key_img = img_key[b:b + 1]
            cur_img = img_non_key[b:b + 1]
            cur_target = target_non_key[b]

            # Cache previous key context from provided temporal pair.
            model.forward_key_frame(key_img, None)
            prev_key_s5 = model.cached_key_s5
            cur_s5 = model.extract_s5(cur_img)

            _, prob = model.forward_apg(prev_key_s5, cur_s5)
            use_key = bool(prob.item() > threshold)
            total_frames += 1
            if use_key:
                key_decisions += 1
                outputs = model.forward_key_frame(cur_img, None)
            else:
                outputs = model.forward_non_key_frame(cur_img, None)

            routed_outputs.append(outputs)
            routed_targets.append(cur_target)

        orig_sizes = torch.stack([t['orig_size'] for t in routed_targets], dim=0).to(device)
        processed = postprocessor(
            {
                'pred_logits': torch.cat([o['pred_logits'] for o in routed_outputs], dim=0),
                'pred_boxes': torch.cat([o['pred_boxes'] for o in routed_outputs], dim=0),
            },
            orig_sizes,
        )
        _accumulate(results, routed_targets, processed, eval_img_ids)

        if batch_idx % 100 == 0:
            print(f'Processed {batch_idx}/{len(val_loader)} batches')

    coco_stats = _run_coco_eval(coco_gt, results, eval_img_ids)
    key_ratio = key_decisions / max(1, total_frames)
    return {
        **coco_stats,
        'key_ratio': key_ratio,
        'avg_interval_proxy': (1.0 / key_ratio) if key_ratio > 0 else float('inf'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--checkpoint', '-w', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    from src.core import YAMLConfig

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = YAMLConfig(args.config)
    model = _build_temporal_model(cfg, device)
    _load_checkpoint(model, args.checkpoint, device)

    stats = evaluate_apg(model, cfg, args.threshold, device)
    print('\nAPG Evaluation:')
    for k, v in stats.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.6f}')
        else:
            print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
