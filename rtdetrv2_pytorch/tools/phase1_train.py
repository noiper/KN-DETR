"""
Phase 1 Training Script for Temporal RT-DETR
Run with: python rtdetrv2_pytorch/tools/phase1_training_v2.py -c rtdetrv2_pytorch/configs/rtdetrv2/phase1_virat_r18vd.yml --pretrained best_virat.pth --training_strategy freeze_key
KD only run with: python rtdetrv2_pytorch/tools/phase1_training_v2.py -c rtdetrv2_pytorch/configs/rtdetrv2/phase1_virat_r18vd.yml --pretrained <model_path> --training_strategy freeze_key --kd_only
"""

import os 
import sys
import hashlib


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from typing import Dict, List, Tuple, Optional

from src.core._config import BaseConfig

from src.zoo.temporal_rtdetr import TemporalRTDETR
from src.core import YAMLConfig
from pycocotools.cocoeval import COCOeval

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

KEY_PATH_PREFIXES = ('backbone.', 'encoder.', 'decoder.')
NON_KEY_HEAD_PREFIXES = (
    'lightweight_decoder.dec_score_head',
    'lightweight_decoder.dec_bbox_head',
    'lightweight_decoder.query_pos_head',
)


def _state_fingerprint(state_dict: Dict[str, torch.Tensor], prefixes: Tuple[str, ...]) -> Dict[str, object]:
    hasher = hashlib.sha256()
    matched_keys = sorted(k for k in state_dict.keys() if k.startswith(prefixes))
    for key in matched_keys:
        tensor = state_dict[key]
        if not torch.is_tensor(tensor):
            continue
        hasher.update(key.encode('utf-8'))
        hasher.update(str(tuple(tensor.shape)).encode('utf-8'))
        hasher.update(str(tensor.dtype).encode('utf-8'))
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return {
        'matched_keys': len(matched_keys),
        'sha256': hasher.hexdigest(),
    }


def _compare_prefixed_state_dicts(
    source_state: Dict[str, torch.Tensor],
    loaded_state: Dict[str, torch.Tensor],
    prefixes: Tuple[str, ...],
) -> Dict[str, object]:
    source_keys = sorted(k for k in source_state.keys() if k.startswith(prefixes))
    missing_in_loaded = []
    mismatched = []
    max_abs_diff = 0.0

    for key in source_keys:
        if key not in loaded_state:
            missing_in_loaded.append(key)
            continue

        src_tensor = source_state[key]
        loaded_tensor = loaded_state[key]

        if src_tensor.shape != loaded_tensor.shape:
            mismatched.append((key, 'shape_mismatch'))
            continue

        if not torch.equal(src_tensor, loaded_tensor):
            if src_tensor.dtype.is_floating_point and loaded_tensor.dtype.is_floating_point:
                diff = (src_tensor.detach().cpu() - loaded_tensor.detach().cpu()).abs().max().item()
                max_abs_diff = max(max_abs_diff, diff)
                mismatched.append((key, f'max_abs_diff={diff:.6g}'))
            else:
                mismatched.append((key, 'value_mismatch'))

    return {
        'source_prefixed_keys': len(source_keys),
        'missing_in_loaded': missing_in_loaded,
        'mismatched': mismatched,
        'max_abs_diff': max_abs_diff,
    }

class Phase1Trainer:
    """
    Training Strategies:
    --- The Two-Stage Curriculum ---
    - 'kd_only': Stage 1 - Train fusion blocks via Feature MSE. (same_frame disabled)
    - 'decoder_only': Stage 2 - Train light decoder via Hungarian. (same_frame disabled)
    
    --- Joint & Fine-tuning Modes ---
    - 'kd': Joint Tuning - Train fusion + light decoder via Hungarian + Feature MSE.
    - 'freeze_key': Train fusion + light decoder via Hungarian. (same_frame supported)
    - 'joint': Train fusion + light decoder + key decoder via Hungarian. (same_frame supported)
    """
    def __init__(
        self,
        model: TemporalRTDETR,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        postprocessor: nn.Module,
        cfg: BaseConfig,
        device: torch.device,
        lambda_non_key: float = 0.5,
        output_dir: str = 'output',
        print_freq: int = 50,
        clip_max_norm: float = 0.1,
        training_strategy: str = 'joint',
        same_frame: bool = False,
        reuse_match_indices: bool = False,
        provenance: Optional[Dict[str, object]] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.postprocessor = postprocessor
        self.cfg = cfg
        self.device = device
        self.lambda_non_key = lambda_non_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.print_freq = print_freq
        self.clip_max_norm = clip_max_norm
        self.training_strategy = training_strategy
        self.same_frame = same_frame
        self.reuse_match_indices = reuse_match_indices
        self.provenance = provenance or {}
        self.lambda_score = getattr(self.cfg, 'lambda_score', 15.0)
        
        valid_strategies = ['kd', 'kd_only', 'decoder_only', 'freeze_key', 'joint']
        if self.training_strategy not in valid_strategies:
            raise ValueError(f"Unknown training strategy: {self.training_strategy}. Must be one of {valid_strategies}")
        if self.same_frame and self.training_strategy in ['kd', 'kd_only', 'decoder_only']:
            print(f"  [Warning] --same_frame is not supported for '{self.training_strategy}'. Disabling it.")
            self.same_frame = False

        print(f"\nTraining Strategy: {self.training_strategy}")
        if self.training_strategy == 'kd':
               self.lambda_kd = getattr(self.cfg, 'lambda_kd', 150.0)
               print(f"  - Feature KD weight (lambda_kd): {self.lambda_kd}")
               print(f"  - Score KD weight (lambda_score): {self.lambda_score}")

        if self.same_frame:
            print("--same_frame is active.")

    def _sanitize_cached_indices(
        self,
        cached_indices: List[Tuple[torch.Tensor, torch.Tensor]],
        targets: List[Dict],
        num_queries: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        sanitized = []
        for (src_idx, tgt_idx), target in zip(cached_indices, targets):
            src_idx = src_idx.to(self.device, dtype=torch.int64)
            tgt_idx = tgt_idx.to(self.device, dtype=torch.int64)
            max_tgt = target['labels'].shape[0]
            valid = (
                (src_idx >= 0)
                & (src_idx < num_queries)
                & (tgt_idx >= 0)
                & (tgt_idx < max_tgt)
            )
            sanitized.append((src_idx[valid], tgt_idx[valid]))
        return sanitized

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch with temporal frame pairs
        """
        self.model.train()

        if self.training_strategy in ['kd', 'kd_only', 'decoder_only', 'freeze_key']:
            for name, module in self.model.named_modules():
                # Force backbone and encoder normalizations to stay frozen
                if 'backbone' in name or 'encoder' in name:
                    module.eval()
                if 'decoder' in name and 'lightweight_decoder' not in name:
                    module.eval()
        total_loss = 0.0
        total_loss_key = 0.0
        total_loss_non_key = 0.0

        # Track KD components for logging
        total_feat_mse = 0.0
        total_score_mse = 0.0

        for batch_idx, batch in enumerate(self.dataloader):
            img_key, target_key, img_non_key, target_non_key = batch
            key_indices = None

            if self.same_frame:
                img_non_key = img_key.clone()
                # Safely clone the target dictionaries
                target_non_key = [{k: v.clone() if isinstance(v, torch.Tensor) else v 
                                  for k, v in t.items()} for t in target_key]

            img_key = img_key.to(self.device)
            img_non_key = img_non_key.to(self.device)
            target_key = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in t.items()} for t in target_key]
            target_non_key = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in t.items()} for t in target_non_key]

            self.optimizer.zero_grad()

            loss = None
            loss_key_value = 0.0
            loss_non_key_value = 0.0

            # 1. KEY FRAME PATH (The Teacher)
            if self.training_strategy == 'joint':
                outputs_key = self.model.forward_key_frame(img_key, target_key)
                if self.reuse_match_indices:
                    loss_dict_key, key_indices = self.criterion(
                        outputs_key, target_key, return_indices=True
                    )
                else:
                    loss_dict_key = self.criterion(outputs_key, target_key)
                # Remove all aux loss (drop multi-layer and DN losses)
                final_loss_dict = {
                    k: v for k, v in loss_dict_key.items() 
                    if not (k[-1].isdigit() or 'dn' in k)
                }
                loss_key = sum(final_loss_dict.values())
                loss = loss_key
                loss_key_value = loss_key.item()
            else:
                # For kd, kd_only, decoder_only, and freeze_key
                self.model.backbone.eval()
                if hasattr(self.model, 'encoder'):
                    self.model.encoder.eval()
                if self.training_strategy in ['kd', 'kd_only', 'decoder_only'] and hasattr(self.model, 'decoder'):
                    if hasattr(self.model, 'decoder'):
                        self.model.decoder.eval()
                    with torch.no_grad():
                        outputs_key = self.model.forward_key_frame(img_key, target_key)
                        # Always get key indices for 'kd' strategy to enable Score Distillation
                        if self.reuse_match_indices or self.training_strategy in ['kd', 'kd_only']:
                            _, key_indices = self.criterion(
                                outputs_key, target_key, return_indices=True
                            )

            # 2. NON-KEY FRAME PATH (The Student)
            if self.training_strategy in ['kd', 'kd_only']:

                with torch.no_grad():
                    teacher_backbone_feats = self.model.backbone(img_non_key)
                    teacher_c3, teacher_c4, teacher_c5 = teacher_backbone_feats[-3:]
                    teacher_ccff = self.model.encoder([teacher_c3, teacher_c4, teacher_c5])

                outputs_non_key, student_fused = self.model.forward_non_key_frame(
                    img_non_key, target_non_key, return_fused=True
                )

                # --- Feature KD Logic ---
                loss_kd_feat = 0.0
                for s_feat, t_feat in zip(student_fused, teacher_ccff):
                    loss_kd_feat += F.mse_loss(F.normalize(s_feat, dim=1), F.normalize(t_feat, dim=1))

                # --- Score KD Logic ---
                criterion_kwargs = {}
                if self.reuse_match_indices:
                    if key_indices is None:
                        raise RuntimeError("Expected key matcher indices, but got None with --reuse_match_indices")
                    num_queries = outputs_non_key['pred_boxes'].shape[1]
                    criterion_kwargs['cached_indices'] = self._sanitize_cached_indices(
                        key_indices, target_non_key, num_queries
                    )

                loss_dict_non_key, non_key_indices = self.criterion(
                    outputs_non_key, target_non_key, **criterion_kwargs, return_indices=True
                )
                loss_hungarian = sum(loss_dict_non_key.values())

                loss_kd_score = 0.0
                batch_size = outputs_non_key['pred_logits'].shape[0]
                student_probs = outputs_non_key['pred_logits'].sigmoid()
                with torch.no_grad():
                    teacher_probs = outputs_key['pred_logits'].sigmoid()

                num_matched_common = 0
                for b in range(batch_size):
                    k_src, k_tgt = key_indices[b]
                    nk_src, nk_tgt = non_key_indices[b]
                    k_map = {t.item(): s for s, t in zip(k_src, k_tgt)}
                    nk_map = {t.item(): s for s, t in zip(nk_src, nk_tgt)}
                    common_gts = set(k_map.keys()) & set(nk_map.keys())
                    for g in common_gts:
                        s_idx = nk_map[g]; t_idx = k_map[g]
                        loss_kd_score += F.mse_loss(student_probs[b, s_idx], teacher_probs[b, t_idx].detach())
                        num_matched_common += 1
                if num_matched_common > 0:
                    loss_kd_score /= num_matched_common

                # Assign to non-key loss
                if self.training_strategy == 'kd_only':
                    loss_non_key = self.lambda_kd * loss_kd_feat + self.lambda_score * loss_kd_score
                else:
                    loss_non_key = loss_hungarian + self.lambda_kd * loss_kd_feat + self.lambda_score * loss_kd_score

                total_feat_mse += loss_kd_feat.item()
                total_score_mse += loss_kd_score.item() if isinstance(loss_kd_score, torch.Tensor) else loss_kd_score

            else:
                # Standard Hungarian loss (decoder_only, freeze_key, and joint)
                outputs_non_key = self.model.forward_non_key_frame(
                    img_non_key, target_non_key
                )
                criterion_kwargs = {}
                if self.reuse_match_indices:
                    if key_indices is None:
                        raise RuntimeError("Expected key matcher indices, but got None with --reuse_match_indices")
                    num_queries = outputs_non_key['pred_boxes'].shape[1]
                    criterion_kwargs['cached_indices'] = self._sanitize_cached_indices(
                        key_indices,
                        target_non_key,
                        num_queries,
                    )
                loss_dict_non_key = self.criterion(
                    outputs_non_key, target_non_key, **criterion_kwargs
                )
                # RTDETRCriterionv2 already applies weight_dict internally.
                loss_non_key = sum(loss_dict_non_key.values())

            loss_non_key_value = loss_non_key.item() if isinstance(loss_non_key, torch.Tensor) else loss_non_key

            # 3. ACCUMULATE & BACKPROP
            if loss is None:
                loss = loss_non_key
            else:
                loss = (1 - self.lambda_non_key) * loss + self.lambda_non_key * loss_non_key

            loss.backward()
            if self.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 
                    max_norm=self.clip_max_norm
                )
            self.optimizer.step()

            total_loss += loss.item() if loss is not None else 0.0
            total_loss_key += loss_key_value
            total_loss_non_key += loss_non_key_value

            # Logging
            if batch_idx % self.print_freq == 0:
                if self.training_strategy == 'kd_only':
                    print(f"Batch [{batch_idx}/{len(self.dataloader)}] "
                          f"Loss: {loss.item():.4f} (Feat MSE: {loss_kd_feat.item():.6f}, Score MSE: {loss_kd_score:.6f})")
                elif self.training_strategy == 'kd':
                    print(f"Batch [{batch_idx}/{len(self.dataloader)}] "
                          f"Loss: {loss.item():.4f} (Hungarian: {loss_hungarian.item():.4f}, Feat MSE: {loss_kd_feat.item():.6f}, Score MSE: {loss_kd_score:.6f})")
                elif self.training_strategy == 'joint':
                    print(f"Epoch [{epoch+1}] Batch [{batch_idx}/{len(self.dataloader)}] "
                          f"Loss: {loss.item():.4f} "
                          f"(Key: {loss_key_value:.4f}, Non-Key: {loss_non_key_value:.4f})")
                else:
                    print(f"Epoch [{epoch+1}] Batch [{batch_idx}/{len(self.dataloader)}] "
                          f"Loss: {loss.item():.4f} (Non-Key only)")

        avg_loss = total_loss / len(self.dataloader)
        avg_loss_key = total_loss_key / len(self.dataloader)
        avg_loss_non_key = total_loss_non_key / len(self.dataloader)

        avg_feat_mse = total_feat_mse / len(self.dataloader)
        avg_score_mse = total_score_mse / len(self.dataloader)

        return {
            'loss': avg_loss,
            'loss_key': avg_loss_key,
            'loss_non_key': avg_loss_non_key,
            'feat_mse': avg_feat_mse,
            'score_mse': avg_score_mse,
            'train_key': self.training_strategy == 'joint',
        }
        
    @torch.no_grad()
    def evaluate(self, val_dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Evaluate model on validation set using official COCO API.
        Computes mAP for both Key and Non-Key frames independently.
        """
        import gc
        self.model.eval()
        self.criterion.eval()
            
        # Ensure dataset has COCO object
        if not hasattr(val_dataloader.dataset, 'coco'):
            print("Error: Dataset missing .coco attribute. Cannot run COCO evaluation.")
            return {'mAP': 0.0}
        
        coco_gt = val_dataloader.dataset.coco
        
        print(f"\n{'='*80}")
        print(f"Evaluating Epoch {epoch}...")

        results_key = []
        results_non_key = []
        eval_img_ids_key = set()
        eval_img_ids_non_key = set()
        
        for batch_idx, (img_key, target_key, img_non_key, target_non_key) in enumerate(val_dataloader):
            if self.same_frame:
                img_non_key = img_key.clone()
                target_non_key = target_key

            img_key = img_key.to(self.device)
            # --- KEY FRAME ---
            outputs_key = self.model.forward_key_frame(img_key, None)
            orig_sizes_k = torch.stack([t["orig_size"] for t in target_key], dim=0).to(self.device)
            res_key = self.postprocessor(outputs_key, orig_sizes_k)
            self._accumulate(results_key, target_key, res_key, eval_img_ids_key)
            # --- NON-KEY FRAME ---
            if isinstance(img_non_key, torch.Tensor):
                img_non_key = img_non_key.to(self.device)
            
            outputs_nk = self.model.forward_non_key_frame(img_non_key, None)
            orig_sizes_nk = torch.stack([t["orig_size"] for t in target_non_key], dim=0).to(self.device)
            res_nk = self.postprocessor(outputs_nk, orig_sizes_nk)
            self._accumulate(results_non_key, target_non_key, res_nk, eval_img_ids_non_key)
            
            if batch_idx % 100 == 0:
                print(f"  Processed {batch_idx}/{len(val_dataloader)} batches")

        stats = {}
        print("\n>>> KEY FRAME RESULTS:")
        k_stats = self._run_coco_eval(coco_gt, results_key, eval_img_ids_key)
        stats.update({f'key_{k}': v for k, v in k_stats.items()})
        
        print("\n>>> NON-KEY FRAME RESULTS:")
        nk_stats = self._run_coco_eval(coco_gt, results_non_key, eval_img_ids_non_key)
        stats.update({f'non_key_{k}': v for k, v in nk_stats.items()})
        print(f"Gap (Key - NonKey) mAP: {k_stats['mAP'] - nk_stats['mAP']:.4f}")

        del results_key
        del results_non_key
        gc.collect()

        print(f"{'='*80}\n")
        return stats

    def _accumulate(self, results_list, targets, outputs, eval_img_ids=None):
        """Helper to convert tensor outputs to COCO list format"""
        for target, output in zip(targets, outputs):
            image_id = int(target['image_id'].item())
            if eval_img_ids is not None:
                if image_id in eval_img_ids:
                    continue # Skip duplicates to prevent mAP inflation
                eval_img_ids.add(image_id)
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            
            for i in range(len(scores)):
                x1, y1, x2, y2 = boxes[i]
                w, h = x2 - x1, y2 - y1
                results_list.append({
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(scores[i])
                })

    def _run_coco_eval(self, coco_gt, results_list, eval_img_ids=None):
        """Helper to run COCOeval on a list of results"""
        import gc
        if not results_list:
            print("  No predictions generated.")
            return {'mAP': 0.0, 'mAP@50': 0.0, 'mAP@75': 0.0}
        coco_dt = coco_gt.loadRes(results_list)
        
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        if eval_img_ids is not None:
            coco_eval.params.imgIds = sorted(eval_img_ids)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
      
        stats =  {
            'mAP': coco_eval.stats[0],
            'mAP@50': coco_eval.stats[1],
            'mAP@75': coco_eval.stats[2]
        }

        del coco_dt
        del coco_eval
        gc.collect()

        return stats
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'provenance': self.provenance,
        }
        
        map5095 = metrics.get('non_key_mAP', 0.0)
        map50 = metrics.get('non_key_mAP@50', 0.0)
        
        # Filename (e.g., checkpoint_epoch_10_050.pth)
        checkpoint_path = self.output_dir / f'{epoch:02d}_{int(map5095 * 1000):03d}_{int(map50 * 1000):03d}.pth'
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Also save as latest
        latest_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # 4. Save best model based on highest Non-Key mAP
        best_path = self.output_dir / 'best_model.pth'
        if not best_path.exists():
            torch.save(checkpoint, best_path)
        else:
            try:
                best_checkpoint = torch.load(best_path, weights_only=False)
                best_map = best_checkpoint['metrics'].get('non_key_mAP', 0.0)
                
                if map5095 > best_map:
                    torch.save(checkpoint, best_path)
                    print(f"✓ New best model saved! (Non-Key mAP improved from {best_map:.4f} to {map5095:.4f})")
            except Exception as e:
                print(f"Warning: Could not read previous best_model.pth to compare mAP. Overwriting it. Error: {e}")
                torch.save(checkpoint, best_path)

def build_model_from_config(config_path: str, device: torch.device):
    """
    Build TemporalRTDETR model from config
    """
    print(f"\nBuilding model from: {config_path}")
    cfg = YAMLConfig(config_path)
    base_model = cfg.model.to(device)
    
    # Extract components
    backbone = base_model.backbone
    encoder = base_model.encoder if hasattr(base_model, 'encoder') else None
    decoder = base_model.decoder if hasattr(base_model, 'decoder') else None
    
    if encoder is None or decoder is None:
        raise ValueError("Model must have encoder and decoder components")
    
    # Get model config
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
    
    # Get Phase 1 specific parameters
    use_lightweight_decoder = cfg.yaml_cfg.get('use_lightweight_decoder', False)
    reuse_position = cfg.yaml_cfg.get('reuse_position', 0)
    
    # Create temporal model
    temporal_model = TemporalRTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        num_classes=cfg.yaml_cfg.get('num_classes', 80),
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        use_lightweight_decoder=use_lightweight_decoder,
        reuse_position=reuse_position,
        enable_apg=cfg.yaml_cfg.get('enable_apg', False),
        apg_in_channels=cfg.yaml_cfg.get('apg_in_channels', 512),
        apg_hidden_channels=cfg.yaml_cfg.get('apg_hidden_channels', 64),
        apg_pool_size=cfg.yaml_cfg.get('apg_pool_size', 4),
    )
    
    return temporal_model, cfg


def load_pretrained_key_frame(model: TemporalRTDETR, pretrained_path: str, device: torch.device):
    """
    Load pretrained weights for key frame path
    """
    print(f"\nLoading pretrained key frame path from: {pretrained_path}")
    
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model']
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"  Missing keys (new components): {len(missing_keys)}")
    if unexpected_keys:
        print(f"  Unexpected keys: {len(unexpected_keys)}")
    
    print(f"  Success!")

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--training_strategy', '-s', type=str, default='freeze_key',
                        choices=['freeze_key', 'joint', 'kd', 'kd_only', 'decoder_only'],
                        help='freeze_key, joint, kd, kd_only, or decoder_only')
    parser.add_argument('--tuning', '-t', type=str, default=None,
                       help='Tuning from checkpoint (auto-detects key-only vs full temporal)')
    parser.add_argument('--eval_only', action='store_true',
                       help='Skip training and only run evaluation on the provided weights')
    parser.add_argument('--eval_before_train', action='store_true',
                       help='Run one validation pass before the first training epoch')
    parser.add_argument('--verify_pretrained_only', action='store_true',
                       help='Verify pretrained load integrity, run one evaluation pass, then exit')
    parser.add_argument('--same_frame', action='store_true',
                       help='Diagnostic: Force non-key frame to be identical to key frame')
    parser.add_argument('--init_weights', action='store_true', 
                       help='Warm start: Copy pretrained key decoder weights to non-key decoder')
    parser.add_argument('--reuse_match_indices', action='store_true',
                       help='Reuse key-frame matcher indices for non-key loss (default: disabled)')
    parser.add_argument('--lambda_kd', type=float, default=None,
                       help='Weighted scale for KD loss (only used in kd strategy)')
    parser.add_argument('--lambda_score', type=float, default=None,
                       help='Weighted scale for Score KD loss (only used in kd strategy)')

    # Optional
    parser.add_argument('--resume', '-r', type=str, default=None, 
                       help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, 
                       help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory (overrides config)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
     
    # Load config and build model
    try:
        model, cfg = build_model_from_config(args.config, device)

        is_temporal = False
        pretrained_metadata = {
            'pretrained_path': args.tuning,
            'checkpoint_type': 'none',
            'state_fingerprint': None,
            'load_integrity': None,
        }
        if args.tuning:
            print(f"\n=> Inspecting weights from: {args.tuning}")
            checkpoint = torch.load(args.tuning, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
            is_temporal = any('fusion_blocks' in k for k in state_dict.keys())
            source_fp = _state_fingerprint(state_dict, KEY_PATH_PREFIXES)
            print(f"   Source key-path fingerprint: {source_fp['sha256'][:16]} ({source_fp['matched_keys']} tensors)")
            
            if is_temporal:
                print("   [Auto-Detect] Full Temporal weights found. Loading strictly...")
                has_non_key_prediction_heads = any(k.startswith(NON_KEY_HEAD_PREFIXES) for k in state_dict.keys())
                if has_non_key_prediction_heads and hasattr(model, 'lightweight_decoder') and model.lightweight_decoder is not None:
                    model.decouple_non_key_prediction_heads()
                    print("   [Safety] Decoupled non-key prediction heads before strict temporal load.")
                model.load_state_dict(state_dict, strict=True)
                load_report = _compare_prefixed_state_dicts(
                    state_dict, model.state_dict(), KEY_PATH_PREFIXES
                )
                if load_report['missing_in_loaded'] or load_report['mismatched']:
                    details = load_report['mismatched'][:5]
                    raise RuntimeError(
                        "Key-path integrity check failed after strict temporal load. "
                        f"missing={len(load_report['missing_in_loaded'])}, "
                        f"mismatched={len(load_report['mismatched'])}, "
                        f"sample={details}"
                    )
                print("   [Integrity] Key-path tensors match source checkpoint.")
                pretrained_metadata['checkpoint_type'] = 'temporal'
            else:
                print("   [Auto-Detect] Standard Key-Frame weights found. Warm-starting...")
                model.load_state_dict(state_dict, strict=False)
                load_report = _compare_prefixed_state_dicts(
                    state_dict, model.state_dict(), KEY_PATH_PREFIXES
                )
                print(
                    "   [Integrity] Warm-start key-path report: "
                    f"missing={len(load_report['missing_in_loaded'])}, "
                    f"mismatched={len(load_report['mismatched'])}, "
                    f"max_abs_diff={load_report['max_abs_diff']:.6g}"
                )
                pretrained_metadata['checkpoint_type'] = 'key_only_or_partial'

            loaded_fp = _state_fingerprint(model.state_dict(), KEY_PATH_PREFIXES)
            pretrained_metadata['state_fingerprint'] = {
                'source_key_path': source_fp,
                'loaded_key_path': loaded_fp,
            }
            pretrained_metadata['load_integrity'] = {
                'missing_in_loaded': len(load_report['missing_in_loaded']),
                'mismatched': len(load_report['mismatched']),
                'max_abs_diff': load_report['max_abs_diff'],
            }
            print("   Success!")

        if args.init_weights:
            if is_temporal:
                print("\n=> [Skipping init_weights] Full temporal model detected. Preserving trained lightweight decoder weights.")
            else:
                print("\n=> Initializing Lightweight Decoder from Heavy Decoder...")
                if hasattr(model, 'lightweight_decoder') and model.lightweight_decoder is not None:
                    heavy_last_layer_state = model.decoder.decoder.layers[-1].state_dict()
                    model.lightweight_decoder.decoder.layers[0].load_state_dict(heavy_last_layer_state)
                    print("   ✅ Successfully copied perfectly trained heavy transformer layer!")
        
        # Get config values (with overrides)
        epochs = args.epochs if args.epochs is not None else getattr(cfg, 'epoches', 50)
        output_dir = args.output_dir if args.output_dir is not None else getattr(cfg, 'output_dir', 'output/phase1_virat')
        seed = args.seed if args.seed is not None else getattr(cfg, 'seed', 42)
        lambda_non_key = getattr(cfg, 'lambda_non_key', 0.5)
        print_freq = getattr(cfg, 'print_freq', 50)
        checkpoint_freq = getattr(cfg, 'checkpoint_freq', 5)
        clip_max_norm = getattr(cfg, 'clip_max_norm', 0.1)

        if args.lambda_kd is not None:
            cfg.lambda_kd = args.lambda_kd
        
        if args.lambda_score is not None:
            cfg.lambda_score = args.lambda_score

        summary_dir = getattr(cfg, 'summary_dir', os.path.join(output_dir, 'summary'))
        writer = SummaryWriter(log_dir=summary_dir)
        print(f"  Summary dir:      {summary_dir}")
        
        print(f"\nConfiguration:")
        print(f"  Epochs:           {epochs}")
        print(f"  Training strategy: {args.training_strategy}")
        print(f"  Reuse match idx:  {args.reuse_match_indices}")
        print(f"  Lambda (non-key): {lambda_non_key}")
        print(f"  Output dir:       {output_dir}")
        print(f"  Seed:             {seed}")
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get DataLoader from config
    print(f"\nLoading dataset from config")
    try:
        train_dataloader = cfg.train_dataloader
        print(f"  DataLoader loaded")
        print(f"  Dataset: {train_dataloader.dataset.__class__.__name__}")
        print(f"  Collate function: {train_dataloader.collate_fn.__class__.__name__}")
        print(f"  Batches/epoch: {len(train_dataloader)}")
    except Exception as e:
        print(f"Error loading dataloader: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get criterion from config
    criterion = cfg.criterion
    print("\n=> Preparing Temporal Model for Optimizer Registration...")

    if args.training_strategy in ['freeze_key', 'kd'] and not args.reuse_match_indices:
        model.decouple_non_key_prediction_heads()
        print("   Enabled decoupled non-key heads/query_pos for fresh-matcher training.")
    
    trainable_params = []
    
    for name, param in model.named_parameters():
        if args.training_strategy == 'kd_only':
            # Train fusion blocks ONLY. 
            if 'fusion_blocks' in name:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
                
        elif args.training_strategy == 'decoder_only':
            # Train lightweight transformer layers ONLY.
            if 'lightweight_decoder.decoder' in name:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
                
        elif args.training_strategy in ['freeze_key', 'kd']:
            # Train fusion blocks and lightweight decoder. Freeze heavy model.
            train_prediction_modules = (
                not args.reuse_match_indices and (
                    'lightweight_decoder.dec_score_head' in name
                    or 'lightweight_decoder.dec_bbox_head' in name
                    or 'lightweight_decoder.query_pos_head' in name
                )
            )
            if 'fusion_blocks' in name or 'lightweight_decoder.decoder' in name or train_prediction_modules:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
                
        elif args.training_strategy == 'joint':
            # Train heavy decoder, fusion blocks, and light decoder.
            if 'decoder.' in name or 'fusion_blocks' in name or 'lightweight_decoder.decoder' in name:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    print(f"   Registered {len(trainable_params)} parameter tensors to Optimizer.")
    
    # Resume
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)
    
    val_dataloader = cfg.val_dataloader if hasattr(cfg, 'val_dataloader') else None
    
    postprocessor = cfg.postprocessor

    # Trainer
    trainer = Phase1Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataloader=train_dataloader,
        postprocessor = postprocessor,
        cfg=cfg,
        device=device,
        lambda_non_key=lambda_non_key,
        output_dir=output_dir,
        print_freq=print_freq,
        clip_max_norm=clip_max_norm,
        training_strategy=args.training_strategy,
        same_frame=args.same_frame,
        reuse_match_indices=args.reuse_match_indices,
        provenance=pretrained_metadata,
    )

    if args.verify_pretrained_only:
        print("\n" + "="*80)
        print("Executing PRETRAINED VERIFICATION ONLY Mode...")
        print("="*80)
        if not args.tuning:
            print("❌ Error: --verify_pretrained_only requires --tuning.")
            sys.exit(1)
        if val_dataloader is None:
            print("⚠️ Warning: No validation dataloader found. Load integrity check already completed.")
            return
        verify_stats = trainer.evaluate(val_dataloader, epoch=0)
        print("\nVerification Evaluation Stats:")
        for k, v in verify_stats.items():
            print(f"  {k}: {v:.4f}")
        return
    
    if args.eval_only:
        print("\n" + "="*80)
        print("Executing EVALUATION ONLY Mode...")
        print("="*80)
        if val_dataloader is None:
            print("❌ Error: No validation dataloader found in config.")
            sys.exit(1)
            
        stats = trainer.evaluate(val_dataloader, epoch=0)
        
        print("\nFinal Evaluation Stats:")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}")
        return

    if args.eval_before_train:
        print("\n" + "="*80)
        print("Executing PRE-TRAINING EVALUATION...")
        print("="*80)
        if val_dataloader is None:
            print("⚠️ Warning: No validation dataloader found in config. Skipping pre-training evaluation.")
        else:
            pretrain_stats = trainer.evaluate(val_dataloader, epoch=start_epoch)
            print("\nPre-Training Evaluation Stats:")
            for k, v in pretrain_stats.items():
                print(f"  {k}: {v:.4f}")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    writer.add_scalar(f"PreTrainEval/{k}", v, start_epoch)
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80 + "\n")
    
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*80}")
        
        metrics = trainer.train_one_epoch(epoch)
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1} Training Summary:")
        print(f"  Total Loss:     {metrics['loss']:.4f}")
        if metrics['train_key']:
            print(f"  Key Frame:      {metrics['loss_key']:.4f}")
        print(f"  Non-Key Frame:  {metrics['loss_non_key']:.4f}")
        print(f"{'='*80}")
        
        if val_dataloader is not None:
            eval_metrics = trainer.evaluate(val_dataloader, epoch)
            metrics.update(eval_metrics)
        
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        # Log all numeric metrics (Losses and mAPs) to TensorBoard
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                writer.add_scalar(f"Metrics/{k}", v, epoch)
        
        writer.add_scalar("Train/Learning_Rate", current_lr, epoch)
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0 or epoch == epochs - 1:
            trainer.save_checkpoint(epoch, metrics)
    
    print("\n" + "="*80)
    print("Training Completed!")
    print(f"Checkpoints: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
