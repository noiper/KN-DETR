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


def _extract_cls_loss(loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    cls_keys = []
    for k in loss_dict.keys():
        if not (k.startswith('loss_vfl') or k.startswith('loss_focal')):
            continue
        if '_aux_' in k or '_dn_' in k or '_enc_' in k:
            continue
        cls_keys.append(k)
    if not cls_keys:
        raise RuntimeError(f'No classification loss key found in loss_dict keys: {list(loss_dict.keys())}')
    return sum(loss_dict[k] for k in cls_keys)


def _load_tuning_weights(model: Any, tuning_path: str, device: torch.device) -> None:
    checkpoint = torch.load(tuning_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded tuning weights from {tuning_path}')
    print(f'  Missing keys: {len(missing)}')
    print(f'  Unexpected keys: {len(unexpected)}')


def _freeze_detector_train_apg_only(model: Any) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if model.apg is None:
        raise RuntimeError('APG is disabled. Set enable_apg=True in config.')
    for param in model.apg.parameters():
        param.requires_grad = True


def _build_adjacent_index(dataset: Any) -> Tuple[Dict[int, List[int]], Dict[int, str], Optional[Path]]:
    """
    Build per-frame adjacent candidate index from dataset metadata.
    Returns:
      - adjacent_map: image_id -> ordered adjacent image_id list (same video)
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
        for i, frame in enumerate(frames):
            image_id = int(frame['id'])
            id_to_file[image_id] = frame['file_name']
            neighbors = [ids[j] for j in sorted(range(len(ids)), key=lambda j: abs(j - i)) if j != i]
            adjacent_map[image_id] = neighbors
    return adjacent_map, id_to_file, root_dir


def _sample_adjacent_candidates(
    non_key_image_id: int,
    candidate_window: int,
    sampled_keys: int,
    adjacent_map: Dict[int, List[int]],
    allow_same_frame: bool,
) -> List[int]:
    neighbors = adjacent_map.get(non_key_image_id, [])
    if candidate_window > 0:
        neighbors = neighbors[:candidate_window]
    if not allow_same_frame:
        neighbors = [nid for nid in neighbors if nid != non_key_image_id]
    if not neighbors:
        return []
    sample_count = min(sampled_keys, len(neighbors))
    perm = torch.randperm(len(neighbors))[:sample_count].tolist()
    return [neighbors[i] for i in perm]


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
    sampled_keys: int,
    candidate_window: int,
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

    for batch_idx, (img_key, target_key, img_non_key, target_non_key) in enumerate(dataloader):
        img_key = img_key.to(device)
        img_non_key = img_non_key.to(device)
        target_non_key = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_non_key]

        batch_size = img_key.shape[0]
        optimizer.zero_grad()
        batch_loss = 0.0
        batch_pairs = 0
        batch_positive = 0

        for b in range(batch_size):
            current_img = img_non_key[b:b + 1]
            current_target: List[Dict] = [target_non_key[b]]
            current_image_id = int(current_target[0]['image_id'].item())

            with torch.no_grad():
                current_s5 = model.extract_s5(current_img).detach()
            current_h, current_w = int(current_img.shape[-2]), int(current_img.shape[-1])

            candidate_image_ids = _sample_adjacent_candidates(
                non_key_image_id=current_image_id,
                candidate_window=candidate_window,
                sampled_keys=sampled_keys,
                adjacent_map=adjacent_map,
                allow_same_frame=allow_same_frame,
            )

            candidate_cls_losses: List[torch.Tensor] = []
            candidate_key_s5: List[torch.Tensor] = []

            with torch.no_grad():
                # Fallback: at least use paired key frame from this sample.
                if not candidate_image_ids:
                    candidate_key_img = img_key[b:b + 1]
                    model.forward_key_frame(candidate_key_img, None)
                    outputs_nk = model.forward_non_key_frame(current_img, None)
                    loss_dict = criterion(outputs_nk, current_target)
                    cls_loss = _extract_cls_loss(loss_dict)
                    candidate_cls_losses.append(cls_loss.detach())
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
                        loss_dict = criterion(outputs_nk, current_target)
                        cls_loss = _extract_cls_loss(loss_dict)
                        candidate_cls_losses.append(cls_loss.detach())
                        candidate_key_s5.append(model.cached_key_s5.detach())

            if not candidate_cls_losses:
                continue

            cls_losses = torch.stack(candidate_cls_losses)  # [num_candidates]
            epsilon = beta * torch.min(cls_losses)
            pseudo_labels = (cls_losses > epsilon).float()

            apg_logits = []
            for key_s5 in candidate_key_s5:
                logit, _ = model.forward_apg(key_s5, current_s5)
                apg_logits.append(logit.squeeze(0))
            apg_logits = torch.stack(apg_logits)  # [num_candidates]

            sample_loss = F.binary_cross_entropy_with_logits(apg_logits, pseudo_labels)
            batch_loss = batch_loss + sample_loss
            batch_pairs += pseudo_labels.numel()
            batch_positive += int(pseudo_labels.sum().item())

        if batch_size > 0:
            batch_loss = batch_loss / batch_size

        batch_loss.backward()
        optimizer.step()

        total_loss += float(batch_loss.item())
        total_pairs += batch_pairs
        total_positive += batch_positive

        if batch_idx % print_freq == 0:
            pos_rate = (batch_positive / max(1, batch_pairs))
            print(f'Batch [{batch_idx}/{len(dataloader)}] apg_loss={batch_loss.item():.6f} pos_rate={pos_rate:.4f}')

    avg_loss = total_loss / max(1, len(dataloader))
    pos_rate = total_positive / max(1, total_pairs)
    return {'apg_loss': avg_loss, 'pseudo_positive_rate': pos_rate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--tuning', '-t', type=str, required=True, help='Temporal checkpoint to warm start detector paths.')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--sampled_keys', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--candidate_window', type=int, default=None)
    parser.add_argument('--allow_same_frame', action='store_true')
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    from src.core import YAMLConfig

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = YAMLConfig(args.config)
    model = _build_temporal_model(cfg, device)

    if model.apg is None:
        raise RuntimeError('APG is disabled by config. Set enable_apg: True.')

    _load_tuning_weights(model, args.tuning, device)
    _freeze_detector_train_apg_only(model)

    criterion = cfg.criterion
    train_dataloader = cfg.train_dataloader

    epochs = args.epochs if args.epochs is not None else int(cfg.yaml_cfg.get('apg_epochs', cfg.yaml_cfg.get('epoches', 5)))
    beta = args.beta if args.beta is not None else float(cfg.yaml_cfg.get('apg_beta', 1.5))
    sampled_keys = args.sampled_keys if args.sampled_keys is not None else int(cfg.yaml_cfg.get('apg_sampled_keys', 10))
    candidate_window = args.candidate_window if args.candidate_window is not None else int(cfg.yaml_cfg.get('apg_candidate_window', cfg.yaml_cfg.get('max_frame_gap', 10)))
    allow_same_frame = bool(args.allow_same_frame or cfg.yaml_cfg.get('apg_allow_same_frame', False))
    lr = args.lr if args.lr is not None else float(cfg.yaml_cfg.get('apg_lr', 1e-4))
    seed = args.seed if args.seed is not None else int(cfg.yaml_cfg.get('seed', 42))
    output_dir = Path(args.output_dir if args.output_dir else cfg.yaml_cfg.get('output_dir', './output/phase2_apg'))
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    optimizer = torch.optim.AdamW(model.apg.parameters(), lr=lr, weight_decay=1e-4)
    adjacent_map, id_to_file, root_dir = _build_adjacent_index(train_dataloader.dataset)

    print(f'Device: {device}')
    print(f'APG epochs: {epochs}')
    print(f'APG beta: {beta}')
    print(f'APG sampled keys: {sampled_keys}')
    print(f'APG candidate window: {candidate_window}')
    print(f'APG allow same frame: {allow_same_frame}')
    print(f'APG lr: {lr}')
    print(f'Output dir: {output_dir}')

    best_loss = float('inf')
    for epoch in range(epochs):
        stats = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            beta=beta,
            sampled_keys=sampled_keys,
            candidate_window=candidate_window,
            allow_same_frame=allow_same_frame,
            adjacent_map=adjacent_map,
            id_to_file=id_to_file,
            root_dir=root_dir,
            print_freq=args.print_freq,
        )
        print(f'Epoch [{epoch + 1}/{epochs}] apg_loss={stats["apg_loss"]:.6f} pos_rate={stats["pseudo_positive_rate"]:.4f}')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': stats,
            'apg_config': {
                'beta': beta,
                'sampled_keys': sampled_keys,
                'candidate_window': candidate_window,
                'allow_same_frame': allow_same_frame,
                'lr': lr,
            },
        }
        latest_path = output_dir / 'apg_latest.pth'
        torch.save(checkpoint, latest_path)

        if stats['apg_loss'] < best_loss:
            best_loss = stats['apg_loss']
            best_path = output_dir / 'apg_best.pth'
            torch.save(checkpoint, best_path)
            print(f'New best APG checkpoint: {best_path}')


if __name__ == '__main__':
    main()
