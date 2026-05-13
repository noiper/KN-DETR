import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from PIL import Image

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

    model = TemporalRTDETR(
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
    return model


def _extract_target_loss(loss_dict: Dict[str, torch.Tensor], loss_type: str = 'both') -> torch.Tensor:
    """
    Extract the specified detection loss components from the loss dict.
    Filters out auxiliary, denoising, and encoder losses.
    """
    relevant_keys = []
    for k in loss_dict.keys():
        # Always exclude auxiliary heads, denoising parts, and encoder-only losses
        if any(x in k for x in ['_aux_', '_dn_', '_enc_']):
            continue

        is_cls = k.startswith('loss_vfl') or k.startswith('loss_focal') or k.startswith('loss_class')
        is_box = k.startswith('loss_bbox') or k.startswith('loss_giou') or k.startswith('loss_l1')

        if loss_type == 'class' and is_cls:
            relevant_keys.append(k)
        elif loss_type == 'box' and is_box:
            relevant_keys.append(k)
        elif loss_type == 'both' and (is_cls or is_box):
            relevant_keys.append(k)
        
    if not relevant_keys:
        raise RuntimeError(f'No {loss_type} loss keys found in loss_dict keys: {list(loss_dict.keys())}')
        
    return sum(loss_dict[k] for k in relevant_keys)


def _extract_prediction_confidence(outputs: Dict[str, torch.Tensor], topk: int) -> torch.Tensor:
    """
    Extract a scalar confidence proxy from detection logits.
    Uses mean top-k max-class sigmoid confidence over queries.
    """
    if 'pred_logits' not in outputs:
        raise RuntimeError('Expected pred_logits in outputs for confidence extraction')
    logits = outputs['pred_logits']  # [B, Q, C]
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise RuntimeError(f'Expected pred_logits shape [1, Q, C], got {tuple(logits.shape)}')

    query_conf = logits.sigmoid().max(dim=-1).values  # [1, Q]
    if topk <= 0:
        return query_conf.mean(dim=1).squeeze(0)

    k = min(int(topk), query_conf.shape[1])
    return query_conf.topk(k, dim=1).values.mean(dim=1).squeeze(0)


def _prepare_targets_for_loss(targets: List[Dict], device: torch.device) -> List[Dict]:
    """
    Criterion expects normalized cxcywh boxes in target['boxes'].
    """
    prepared: List[Dict] = []
    for t in targets:
        nt = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        boxes = nt.get('boxes', None)
        if not isinstance(boxes, torch.Tensor) or boxes.numel() == 0:
            prepared.append(nt)
            continue

        boxes = boxes.clone()
        is_normalized = (boxes <= 1.01).all()
        if not is_normalized:
            # If boxes look like xyxy, convert to cxcywh first.
            if boxes.shape[-1] == 4 and (boxes[:, 2:] >= boxes[:, :2]).all():
                boxes[:, 2:] -= boxes[:, :2]  # xyxy -> xywh
                boxes[:, :2] += boxes[:, 2:] / 2  # xywh -> cxcywh

            h, w = nt['orig_size'][0], nt['orig_size'][1]
            scale = torch.tensor([w, h, w, h], device=device, dtype=boxes.dtype)
            boxes = boxes / scale

        nt['boxes'] = boxes
        prepared.append(nt)
    return prepared


def _load_tuning_weights(model: Any, tuning_path: str, device: torch.device) -> None:
    checkpoint = torch.load(tuning_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded tuning weights from {tuning_path}')
    print(f'  Missing keys: {len(missing)}')
    print(f'  Unexpected keys: {len(unexpected)}')


def _freeze_detector_train_apg_only(model: Any) -> None:
    """
    Aggressively freeze all detector parameters and verify that only APG is trainable.
    """
    # 1. Freeze everything
    for name, param in model.named_parameters():
        param.requires_grad = False

    if model.apg is None:
        raise RuntimeError('APG is disabled. Set enable_apg=True in config.')

    # 2. Unfreeze ONLY APG parameters
    for name, param in model.apg.named_parameters():
        param.requires_grad = True

    # 3. Explicit verification for problematic layers
    # Ensures they stay frozen even if shared/referenced elsewhere
    for name, param in model.named_parameters():
        if any(x in name for x in ['decoder.query_pos_head', 'decoder.dec_score_head', 'decoder.dec_bbox_head',
                                   'lightweight_decoder.query_pos_head', 'lightweight_decoder.dec_score_head', 'lightweight_decoder.dec_bbox_head']):
            param.requires_grad = False

    # 4. Print summary to terminal for user verification
    unfrozen = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"--- Parameter Freezing Summary ---")
    print(f"  Total parameters: {len(list(model.parameters()))}")
    print(f"  Unfrozen parameters: {len(unfrozen)}")
    if len(unfrozen) > 0:
        print(f"  Unfrozen roots: {set([n.split('.')[0] for n in unfrozen])}")
    print(f"----------------------------------")


class ThresholdEMA:
    def __init__(self, momentum=0.99, init_value=0.5):
        self.momentum = momentum
        self.value = init_value
    
    def update(self, batch_loss_mean):
        if isinstance(batch_loss_mean, torch.Tensor):
            batch_loss_mean = batch_loss_mean.item()
        self.value = (self.momentum * self.value + 
                     (1 - self.momentum) * batch_loss_mean)
        return self.value


def _build_adjacent_index(dataset: Any) -> Tuple[Dict[int, List[int]], Dict[int, str], Optional[Path]]:
    """
    Build chronological frame list per video.
    Returns:
      - adjacent_map: image_id -> full chronological image_id list of its video
      - id_to_file: image_id -> relative file path
      - root_dir: dataset root path if available
    """
    adjacent_map: Dict[int, List[int]] = {}
    id_to_file: Dict[int, str] = {}
    root_dir = Path(dataset.root_dir) if hasattr(dataset, 'root_dir') else None

    if not hasattr(dataset, 'video_frames'):
        return adjacent_map, id_to_file, root_dir

    for _, frames in dataset.video_frames.items():
        ids = [int(f['id']) for f in frames]
        for frame in frames:
            image_id = int(frame['id'])
            id_to_file[image_id] = frame['file_name']
            adjacent_map[image_id] = ids
    return adjacent_map, id_to_file, root_dir


def _sample_adjacent_candidates(
    non_key_image_id: int,
    m: int,
    adjacent_map: Dict[int, List[int]],
    allow_same_frame: bool,
) -> List[int]:
    """
    Deterministic preceding selection.
    """
    video_ids = adjacent_map.get(non_key_image_id, [])
    if not video_ids:
        return []
    
    try:
        idx = video_ids.index(non_key_image_id)
    except ValueError:
        return []

    if allow_same_frame:
        # Select [idx - m + 1, ..., idx]
        start_idx = max(0, idx - m + 1)
        end_idx = idx + 1
    else:
        # Select [idx - m, ..., idx - 1]
        start_idx = max(0, idx - m)
        end_idx = idx
        
    return video_ids[start_idx:end_idx]


def _load_and_resize_key_image(
    image_id: int,
    id_to_file: Dict[int, str],
    root_dir: Optional[Path],
    ref_hw: Tuple[int, int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if root_dir is None or image_id not in id_to_file:
        return None
    img_path = root_dir / id_to_file[image_id]
    if not img_path.exists():
        return None

    with Image.open(img_path).convert('RGB') as img:
        resized = TVF.resize(img, list(ref_hw))
        tensor = TVF.to_tensor(resized).unsqueeze(0).to(device)
    return tensor


def train_one_epoch(
    model: Any,
    criterion,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    class_scale: float,
    conf_lambda: float,
    conf_topk: int,
    sampled_keys: int,
    ema: ThresholdEMA,
    threshold_type: str,
    allow_same_frame: bool,
    adjacent_map: Dict[int, List[int]],
    id_to_file: Dict[int, str],
    root_dir: Optional[Path],
    print_freq: int,
) -> Dict[str, float]:
    model.eval()
    model.apg.train()
    criterion.eval()

    total_loss = 0.0
    total_pairs = 0
    total_positive = 0
    total_cls_raw = 0.0
    total_gap_raw = 0.0
    total_target_score = 0.0
    target_score_min = float('inf')
    target_score_max = float('-inf')

    for batch_idx, (img_key, _, img_non_key, target_non_key) in enumerate(dataloader):
        img_key = img_key.to(device)
        img_non_key = img_non_key.to(device)
        target_non_key = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_non_key]

        batch_size = img_key.shape[0]
        optimizer.zero_grad()
        batch_loss = torch.zeros((), device=device)
        batch_pairs = 0
        batch_positive = 0
        batch_target_scores = []

        for b in range(batch_size):
            current_img = img_non_key[b:b + 1]
            current_target: List[Dict] = [target_non_key[b]]
            current_target_loss = _prepare_targets_for_loss(current_target, device)
            current_image_id = int(current_target[0]['image_id'].item())

            with torch.no_grad():
                current_s5 = model.extract_s5(current_img).detach()
            current_h, current_w = int(current_img.shape[-2]), int(current_img.shape[-1])

            candidate_image_ids = _sample_adjacent_candidates(
                non_key_image_id=current_image_id,
                m=sampled_keys,
                adjacent_map=adjacent_map,
                allow_same_frame=allow_same_frame,
            )

            candidate_class_losses: List[torch.Tensor] = []
            candidate_conf_gaps: List[torch.Tensor] = []
            candidate_key_s5: List[torch.Tensor] = []

            with torch.no_grad():
                outputs_key_current = model.forward_key_frame(current_img, None)
                current_key_conf = _extract_prediction_confidence(outputs_key_current, topk=conf_topk).detach()

                # Fallback: at least use paired key frame from this sample.
                if not candidate_image_ids:
                    candidate_key_img = img_key[b:b + 1]
                    model.forward_key_frame(candidate_key_img, None)
                    outputs_nk = model.forward_non_key_frame(current_img, None)
                    loss_dict = criterion(outputs_nk, current_target_loss)
                    target_class_loss = _extract_target_loss(loss_dict, 'class')
                    non_key_conf = _extract_prediction_confidence(outputs_nk, topk=conf_topk).detach()
                    conf_gap = torch.clamp(current_key_conf - non_key_conf, min=0.0)
                    candidate_class_losses.append(target_class_loss.detach())
                    candidate_conf_gaps.append(conf_gap)
                    candidate_key_s5.append(model.cached_key_s5.detach())
                else:
                    for cand_image_id in candidate_image_ids:
                        candidate_key_img = _load_and_resize_key_image(
                            image_id=cand_image_id,
                            id_to_file=id_to_file,
                            root_dir=root_dir,
                            ref_hw=(current_h, current_w),
                            device=device,
                        )
                        if candidate_key_img is None:
                            continue
                        model.forward_key_frame(candidate_key_img, None)
                        outputs_nk = model.forward_non_key_frame(current_img, None)
                        loss_dict = criterion(outputs_nk, current_target_loss)
                        target_class_loss = _extract_target_loss(loss_dict, 'class')
                        non_key_conf = _extract_prediction_confidence(outputs_nk, topk=conf_topk).detach()
                        conf_gap = torch.clamp(current_key_conf - non_key_conf, min=0.0)
                        candidate_class_losses.append(target_class_loss.detach())
                        candidate_conf_gaps.append(conf_gap)
                        candidate_key_s5.append(model.cached_key_s5.detach())

            if not candidate_class_losses:
                continue

            class_losses = torch.stack(candidate_class_losses)  # [num_candidates]
            conf_gaps = torch.stack(candidate_conf_gaps)  # [num_candidates]
            # Multiplicative coupling: L_apg = lambda_cls * L_cls * (1 + lambda_conf * gap)
            target_scores = class_scale * class_losses * (1 + conf_lambda * conf_gaps)

            if threshold_type == 'ema':
                threshold = beta * ema.value
            elif threshold_type == 'per_sample':
                # Use the closest frame (last element) as baseline
                threshold = beta * target_scores[-1]
            else:
                threshold = beta * torch.min(target_scores)

            pseudo_labels = (target_scores > threshold).float()

            apg_logits = []
            for key_s5 in candidate_key_s5:
                logit, _ = model.forward_apg(key_s5, current_s5)
                apg_logits.append(logit.squeeze(0))
            apg_logits = torch.stack(apg_logits)  # [num_candidates]

            sample_loss = F.binary_cross_entropy_with_logits(apg_logits, pseudo_labels)
            batch_loss = batch_loss + sample_loss
            batch_pairs += pseudo_labels.numel()
            batch_positive += int(pseudo_labels.sum().item())
            total_cls_raw += float(class_losses.sum().item())
            total_gap_raw += float(conf_gaps.sum().item())
            total_target_score += float(target_scores.sum().item())
            batch_target_scores.extend(target_scores.tolist())
            target_score_min = min(target_score_min, float(target_scores.min().item()))
            target_score_max = max(target_score_max, float(target_scores.max().item()))

        if batch_size > 0:
            batch_loss = batch_loss / batch_size
        
        if batch_target_scores:
            ema.update(sum(batch_target_scores) / len(batch_target_scores))

        batch_loss.backward()
        optimizer.step()

        total_loss += float(batch_loss.item())
        total_pairs += batch_pairs
        total_positive += batch_positive

        if batch_idx % print_freq == 0:
            pos_rate = (batch_positive / max(1, batch_pairs))
            avg_target = total_target_score / max(1, total_pairs)
            print(
                f'Batch [{batch_idx}/{len(dataloader)}] apg_loss={batch_loss.item():.6f} '
                f'pos_rate={pos_rate:.4f} target_mean={avg_target:.4f} lambda={conf_lambda:.3f}'
            )

    avg_loss = total_loss / max(1, len(dataloader))
    pos_rate = total_positive / max(1, total_pairs)
    if target_score_min == float('inf'):
        target_score_min = 0.0
        target_score_max = 0.0

    return {
        'apg_loss': avg_loss,
        'pseudo_positive_rate': pos_rate,
        'cls_raw_mean': total_cls_raw / max(1, total_pairs),
        'conf_gap_raw_mean': total_gap_raw / max(1, total_pairs),
        'target_score_mean': total_target_score / max(1, total_pairs),
        'target_score_min': target_score_min,
        'target_score_max': target_score_max,
    }


@torch.no_grad()
def evaluate_one_epoch(
    model: Any,
    criterion,
    dataloader,
    device: torch.device,
    beta: float,
    class_scale: float,
    conf_lambda: float,
    conf_topk: int,
    sampled_keys: int,
    ema: ThresholdEMA,
    threshold_type: str,
    allow_same_frame: bool,
    adjacent_map: Dict[int, List[int]],
    id_to_file: Dict[int, str],
    root_dir: Optional[Path],
    print_freq: int,
) -> Dict[str, float]:
    model.eval()
    criterion.eval()

    total_loss = 0.0
    total_pairs = 0
    total_positive = 0
    total_cls_raw = 0.0
    total_gap_raw = 0.0
    total_target_score = 0.0
    target_score_min = float('inf')
    target_score_max = float('-inf')

    for batch_idx, (img_key, _, img_non_key, target_non_key) in enumerate(dataloader):
        img_key = img_key.to(device)
        img_non_key = img_non_key.to(device)
        target_non_key = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_non_key]

        batch_size = img_key.shape[0]
        batch_loss = torch.zeros((), device=device)
        batch_pairs = 0
        batch_positive = 0
        batch_target_scores = []

        for b in range(batch_size):
            current_img = img_non_key[b:b + 1]
            current_target: List[Dict] = [target_non_key[b]]
            current_target_loss = _prepare_targets_for_loss(current_target, device)
            current_image_id = int(current_target[0]['image_id'].item())
            current_h, current_w = int(current_img.shape[-2]), int(current_img.shape[-1])

            current_s5 = model.extract_s5(current_img).detach()
            candidate_image_ids = _sample_adjacent_candidates(
                non_key_image_id=current_image_id,
                m=sampled_keys,
                adjacent_map=adjacent_map,
                allow_same_frame=allow_same_frame,
            )

            candidate_class_losses: List[torch.Tensor] = []
            candidate_conf_gaps: List[torch.Tensor] = []
            candidate_key_s5: List[torch.Tensor] = []

            outputs_key_current = model.forward_key_frame(current_img, None)
            current_key_conf = _extract_prediction_confidence(outputs_key_current, topk=conf_topk).detach()

            # Fallback: at least use paired key frame from this sample.
            if not candidate_image_ids:
                candidate_key_img = img_key[b:b + 1]
                model.forward_key_frame(candidate_key_img, None)
                outputs_nk = model.forward_non_key_frame(current_img, None)
                loss_dict = criterion(outputs_nk, current_target_loss)
                target_class_loss = _extract_target_loss(loss_dict, 'class')
                non_key_conf = _extract_prediction_confidence(outputs_nk, topk=conf_topk).detach()
                conf_gap = torch.clamp(current_key_conf - non_key_conf, min=0.0)
                candidate_class_losses.append(target_class_loss.detach())
                candidate_conf_gaps.append(conf_gap)
                candidate_key_s5.append(model.cached_key_s5.detach())
            else:
                for cand_image_id in candidate_image_ids:
                    candidate_key_img = _load_and_resize_key_image(
                        image_id=cand_image_id,
                        id_to_file=id_to_file,
                        root_dir=root_dir,
                        ref_hw=(current_h, current_w),
                        device=device,
                    )
                    if candidate_key_img is None:
                        continue
                    model.forward_key_frame(candidate_key_img, None)
                    outputs_nk = model.forward_non_key_frame(current_img, None)
                    loss_dict = criterion(outputs_nk, current_target_loss)
                    target_class_loss = _extract_target_loss(loss_dict, 'class')
                    non_key_conf = _extract_prediction_confidence(outputs_nk, topk=conf_topk).detach()
                    conf_gap = torch.clamp(current_key_conf - non_key_conf, min=0.0)
                    candidate_class_losses.append(target_class_loss.detach())
                    candidate_conf_gaps.append(conf_gap)
                    candidate_key_s5.append(model.cached_key_s5.detach())

            if not candidate_class_losses:
                continue

            class_losses = torch.stack(candidate_class_losses)  # [num_candidates]
            conf_gaps = torch.stack(candidate_conf_gaps)  # [num_candidates]
            # Multiplicative coupling: L_apg = lambda_cls * L_cls * (1 + lambda_conf * gap)
            target_scores = class_scale * class_losses * (1 + conf_lambda * conf_gaps)

            if threshold_type == 'ema':
                threshold = beta * ema.value
            elif threshold_type == 'per_sample':
                # Use the closest frame (last element) as baseline
                threshold = beta * target_scores[-1]
            else:
                threshold = beta * torch.min(target_scores)

            pseudo_labels = (target_scores > threshold).float()

            apg_logits = []
            for key_s5 in candidate_key_s5:
                logit, _ = model.forward_apg(key_s5, current_s5)
                apg_logits.append(logit.squeeze(0))
            apg_logits = torch.stack(apg_logits)  # [num_candidates]

            sample_loss = F.binary_cross_entropy_with_logits(apg_logits, pseudo_labels)
            batch_loss = batch_loss + sample_loss
            batch_pairs += pseudo_labels.numel()
            batch_positive += int(pseudo_labels.sum().item())
            total_cls_raw += float(class_losses.sum().item())
            total_gap_raw += float(conf_gaps.sum().item())
            total_target_score += float(target_scores.sum().item())
            batch_target_scores.extend(target_scores.tolist())
            target_score_min = min(target_score_min, float(target_scores.min().item()))
            target_score_max = max(target_score_max, float(target_scores.max().item()))

        if batch_size > 0:
            batch_loss = batch_loss / batch_size

        if batch_target_scores:
            ema.update(sum(batch_target_scores) / len(batch_target_scores))

        total_loss += float(batch_loss.item())
        total_pairs += batch_pairs
        total_positive += batch_positive

        if batch_idx % print_freq == 0:
            pos_rate = (batch_positive / max(1, batch_pairs))
            avg_target = total_target_score / max(1, total_pairs)
            print(
                f'[VAL] Batch [{batch_idx}/{len(dataloader)}] apg_loss={batch_loss.item():.6f} '
                f'pos_rate={pos_rate:.4f} target_mean={avg_target:.4f} lambda={conf_lambda:.3f}'
            )

    avg_loss = total_loss / max(1, len(dataloader))
    pos_rate = total_positive / max(1, total_pairs)
    if target_score_min == float('inf'):
        target_score_min = 0.0
        target_score_max = 0.0

    return {
        'apg_loss': avg_loss,
        'pseudo_positive_rate': pos_rate,
        'cls_raw_mean': total_cls_raw / max(1, total_pairs),
        'conf_gap_raw_mean': total_gap_raw / max(1, total_pairs),
        'target_score_mean': total_target_score / max(1, total_pairs),
        'target_score_min': target_score_min,
        'target_score_max': target_score_max,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--tuning', '-t', type=str, required=True, help='Temporal checkpoint to warm start detector paths.')
    parser.add_argument('--resume', '-r', type=str, default=None, help='Path to checkpoint to resume training from.')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--loss', type=str, choices=['class'], default=None, help='Routing target uses class loss only (loss_vfl/loss_focal/loss_class).')
    parser.add_argument('--class_scale', type=float, default=None, help='Fixed multiplier for class loss in the routing target.')
    parser.add_argument('--conf_lambda', type=float, default=None, help='Lambda for confidence gap term in target score.')
    parser.add_argument('--conf_topk', type=int, default=None, help='Top-k queries used for confidence proxy extraction.')
    parser.add_argument('--sampled_keys', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--candidate_window', type=int, default=None)
    parser.add_argument('--allow_same_frame', action='store_true')
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--eval_val', action='store_true', help='Run APG pseudo-label validation on val set each epoch.')
    parser.add_argument('--val_print_freq', type=int, default=None, help='Print frequency for val pseudo-label pass.')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    from src.core import YAMLConfig

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = YAMLConfig(args.config)
    model = _build_temporal_model(cfg, device)

    if model.apg is None:
        raise RuntimeError('APG is disabled by config. Set enable_apg: True.')

    if hasattr(model, 'decouple_non_key_prediction_heads'):
        model.decouple_non_key_prediction_heads()
        print('Decoupled non-key prediction heads before loading warm-start weights.')

    _load_tuning_weights(model, args.tuning, device)
    _freeze_detector_train_apg_only(model)

    criterion = cfg.criterion
    train_dataloader = cfg.train_dataloader

    epochs = args.epochs if args.epochs is not None else int(cfg.yaml_cfg.get('apg_epochs', cfg.yaml_cfg.get('epoches', 5)))
    beta = args.beta if args.beta is not None else float(cfg.yaml_cfg.get('apg_beta', 1.5))
    loss_type = args.loss if args.loss is not None else str(cfg.yaml_cfg.get('apg_loss_type', 'class'))
    class_scale = args.class_scale if args.class_scale is not None else float(cfg.yaml_cfg.get('apg_class_scale', 1.0))
    conf_lambda = args.conf_lambda if args.conf_lambda is not None else float(cfg.yaml_cfg.get('apg_conf_lambda', 1.0))
    conf_topk = args.conf_topk if args.conf_topk is not None else int(cfg.yaml_cfg.get('apg_conf_topk', 50))
    sampled_keys = args.sampled_keys if args.sampled_keys is not None else int(cfg.yaml_cfg.get('apg_sampled_keys', 4))
    threshold_type = str(cfg.yaml_cfg.get('apg_threshold_type', 'local'))
    val_threshold_type = str(cfg.yaml_cfg.get('apg_val_threshold_type', 'local'))
    candidate_window = args.candidate_window if args.candidate_window is not None else int(cfg.yaml_cfg.get('apg_candidate_window', cfg.yaml_cfg.get('max_frame_gap', 10)))
    allow_same_frame = bool(args.allow_same_frame or cfg.yaml_cfg.get('apg_allow_same_frame', False))
    eval_val = bool(args.eval_val or cfg.yaml_cfg.get('apg_eval_val', False))
    val_print_freq = args.val_print_freq if args.val_print_freq is not None else int(cfg.yaml_cfg.get('apg_val_print_freq', args.print_freq))
    lr = args.lr if args.lr is not None else float(cfg.yaml_cfg.get('apg_lr', 1e-4))
    seed = args.seed if args.seed is not None else int(cfg.yaml_cfg.get('seed', 42))
    output_dir = Path(args.output_dir if args.output_dir else cfg.yaml_cfg.get('output_dir', './output/phase2_apg'))
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    optimizer = torch.optim.AdamW(model.apg.parameters(), lr=lr, weight_decay=1e-4)
    train_ema = ThresholdEMA(momentum=0.99, init_value=0.5)
    val_ema = ThresholdEMA(momentum=0.99, init_value=0.5)

    adjacent_map, id_to_file, root_dir = _build_adjacent_index(train_dataloader.dataset)
    val_dataloader = None
    val_adjacent_map: Dict[int, List[int]] = {}
    val_id_to_file: Dict[int, str] = {}
    val_root_dir: Optional[Path] = None
    if eval_val:
        if not hasattr(cfg, 'val_dataloader'):
            raise RuntimeError('Validation dataloader is required when eval_val is enabled.')
        val_dataloader = cfg.val_dataloader
        val_adjacent_map, val_id_to_file, val_root_dir = _build_adjacent_index(val_dataloader.dataset)

    start_epoch = 0
    best_loss = float('inf')

    if loss_type != 'class':
        raise ValueError(f'apg_loss_type must be "class" for the configured target, got: {loss_type}')
    if class_scale < 0.0:
        raise ValueError(f'class_scale must be >= 0, got: {class_scale}')
    if conf_lambda < 0.0:
        raise ValueError(f'conf_lambda must be >= 0, got: {conf_lambda}')
    if conf_topk < 0:
        raise ValueError(f'conf_topk must be >= 0, got: {conf_topk}')
    if val_print_freq <= 0:
        raise ValueError(f'val_print_freq must be > 0, got: {val_print_freq}')
    if threshold_type not in ('local', 'ema', 'per_sample'):
        raise ValueError(f'apg_threshold_type must be one of [local, ema, per_sample], got: {threshold_type}')
    if val_threshold_type not in ('local', 'ema', 'per_sample'):
        raise ValueError(f'apg_val_threshold_type must be one of [local, ema, per_sample], got: {val_threshold_type}')

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        if 'train_ema_value' in checkpoint:
            train_ema.value = checkpoint['train_ema_value']
        if 'val_ema_value' in checkpoint:
            val_ema.value = checkpoint['val_ema_value']
        # Backward compatibility
        if 'ema_value' in checkpoint and 'train_ema_value' not in checkpoint:
            train_ema.value = checkpoint['ema_value']
            
        print(f"Successfully resumed from epoch {start_epoch} (best_loss: {best_loss:.6f}, train_ema: {train_ema.value:.6f})")

    print(f'Device: {device}')
    print(f'APG epochs: {epochs}')
    print(f'APG beta: {beta}')
    print(f'APG loss type: {loss_type}')
    print(f'APG class scale: {class_scale}')
    print(f'APG confidence lambda: {conf_lambda}')
    print(f'APG confidence top-k: {conf_topk}')
    print(f'APG sampled keys (m): {sampled_keys}')
    print(f'APG threshold type: {threshold_type}')
    print(f'APG val threshold type: {val_threshold_type}')
    print(f'APG allow same frame: {allow_same_frame}')
    print(f'APG eval val: {eval_val}')
    print(f'APG val print freq: {val_print_freq}')
    print(f'APG lr: {lr}')
    print(f'Output dir: {output_dir}')

    for epoch in range(start_epoch, epochs):
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            beta=beta,
            class_scale=class_scale,
            conf_lambda=conf_lambda,
            conf_topk=conf_topk,
            sampled_keys=sampled_keys,
            ema=train_ema,
            threshold_type=threshold_type,
            allow_same_frame=allow_same_frame,
            adjacent_map=adjacent_map,
            id_to_file=id_to_file,
            root_dir=root_dir,
            print_freq=args.print_freq,
        )
        val_stats = None
        if eval_val and val_dataloader is not None:
            val_stats = evaluate_one_epoch(
                model=model,
                criterion=criterion,
                dataloader=val_dataloader,
                device=device,
                beta=beta,
                class_scale=class_scale,
                conf_lambda=conf_lambda,
                conf_topk=conf_topk,
                sampled_keys=sampled_keys,
                ema=val_ema,
                threshold_type=val_threshold_type,
                allow_same_frame=allow_same_frame,
                adjacent_map=val_adjacent_map,
                id_to_file=val_id_to_file,
                root_dir=val_root_dir,
                print_freq=val_print_freq,
            )

        if val_stats is None:
            print(
                f'Epoch [{epoch + 1}/{epochs}] apg_loss={train_stats["apg_loss"]:.6f} '
                f'pos_rate={train_stats["pseudo_positive_rate"]:.4f} '
                f'thresh_type={threshold_type} '
                f'target_range=[{train_stats["target_score_min"]:.4f}, {train_stats["target_score_max"]:.4f}] '
                f'target_mean={train_stats["target_score_mean"]:.4f} '
                f'train_ema={train_ema.value:.4f}'
            )
        else:
            print(
                f'Epoch [{epoch + 1}/{epochs}] train_loss={train_stats["apg_loss"]:.6f} '
                f'val_loss={val_stats["apg_loss"]:.6f} '
                f'pos_rate=[{train_stats["pseudo_positive_rate"]:.4f}, {val_stats["pseudo_positive_rate"]:.4f}] '
                f'thresh_type=[{threshold_type}, {val_threshold_type}] '
                f'ema=[{train_ema.value:.4f}, {val_ema.value:.4f}] '
                f'val_target_range=[{val_stats["target_score_min"]:.4f}, {val_stats["target_score_max"]:.4f}]'
            )

        metrics = dict(train_stats)
        if val_stats is not None:
            metrics.update({f'val_{k}': v for k, v in val_stats.items()})

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_loss': best_loss,
            'train_ema_value': train_ema.value,
            'val_ema_value': val_ema.value,
            'apg_config': {
                'beta': beta,
                'loss_type': loss_type,
                'class_scale': class_scale,
                'conf_lambda': conf_lambda,
                'conf_topk': conf_topk,
                'eval_val': eval_val,
                'sampled_keys': sampled_keys,
                'threshold_type': threshold_type,
                'val_threshold_type': val_threshold_type,
                'allow_same_frame': allow_same_frame,
                'lr': lr,
            },
        }
        epoch_path = output_dir / f"{epoch:02d}.pth"
        torch.save(checkpoint, epoch_path)

        latest_path = output_dir / 'apg_latest.pth'
        torch.save(checkpoint, latest_path)

        if train_stats['apg_loss'] < best_loss:
            best_loss = train_stats['apg_loss']
            checkpoint['best_loss'] = best_loss # Update best_loss in current checkpoint for next save
            best_path = output_dir / 'apg_best.pth'
            torch.save(checkpoint, best_path)
            print(f'New best APG checkpoint: {best_path}')


if __name__ == '__main__':
    main()
