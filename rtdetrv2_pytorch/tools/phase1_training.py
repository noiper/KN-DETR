"""
Phase 1 Training Script for Temporal RT-DETR
Run with: python rtdetrv2_pytorch/tools/phase1_training.py -c rtdetrv2_pytorch/configs/rtdetrv2/phase1_virat_r18vd.yml --pretrained best.pth --training_strategy freeze_key
"""

import os 
from pyexpat import model
import sys


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from typing import Dict, List, Optional

from src.data import CocoEvaluator
from src.misc import MetricLogger
from src.core._config import BaseConfig

from torchmetrics.detection.mean_ap import MeanAveragePrecision


# Import to register classes BEFORE loading config
from src.zoo.temporal_rtdetr import TemporalRTDETR, ViratTemporalDataset
from src.core import YAMLConfig
from pycocotools.cocoeval import COCOeval

from torch.utils.tensorboard import SummaryWriter

class Phase1Trainer:
    """
    Training Strategies:
    - 'alternate': Alternate between key and non-key frame training (like Faster R-CNN)
    - 'freeze_key': Freeze key frame path, only train non-key path
    - 'joint': Train both paths together (dafault)
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
        alternate_interval: int = 1,
        same_frame: bool = False,
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
        self.alternate_interval = alternate_interval
        self.same_frame = same_frame
        
        print(f"\nTraining Strategy: {training_strategy}")
        if training_strategy == 'alternate':
            print(f"  Alternate interval: {alternate_interval} epoch(s)")
            print(f"  Note: Backbone/Encoder/Fusion shared, only decoder alternates")
        elif training_strategy == 'freeze_key':
            # Freeze key frame path
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            for param in self.model.decoder.parameters():
                param.requires_grad = False
            
            # Ensure fusion block and lightweight decoder are trainable
            for _, fusion_block in enumerate(self.model.fusion_blocks):
                for param in fusion_block.parameters():
                    param.requires_grad = True
            
            # Ensure lightweight decoder is trainable (if it exists)
            if hasattr(self.model, 'lightweight_decoder') and self.model.lightweight_decoder is not None:
                for param in self.model.lightweight_decoder.parameters():
                    param.requires_grad = True
            
            # Re-freeze the shared prediction heads.
            for param in self.model.lightweight_decoder.dec_score_head.parameters():
                param.requires_grad = False
            for param in self.model.lightweight_decoder.dec_bbox_head.parameters():
                param.requires_grad = False
            for param in self.model.lightweight_decoder.input_proj.parameters():
                param.requires_grad = False
            for param in self.model.lightweight_decoder.query_pos_head.parameters():
                param.requires_grad = False
    
        elif training_strategy == 'joint':
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            print(f"  Training both key and non-key paths jointly")
        else:
            raise ValueError(f"Invalid strategy: {training_strategy}")
        
    def _set_decoder_trainable(self, trainable: bool):
        """Enable/disable gradient for decoder only"""
        for param in self.model.decoder.parameters():
            param.requires_grad = trainable
    
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch with temporal frame pairs
        """
        self.model.train()
        
        # Determine training mode for this epoch
        if self.training_strategy == 'alternate':
            epoch_cycle = epoch // self.alternate_interval
            train_key = (epoch_cycle % 2 == 0)
            train_non_key = not train_key
            
            if train_key:
                print(f"  Training mode: Key frame (backbone/encoder/decoder)")
                self._set_decoder_trainable(True)
            else:
                print(f"  Training mode: Non-key frame (backbone/encoder, decoder frozen)")
                self._set_decoder_trainable(False)
                
        elif self.training_strategy == 'freeze_key':
            train_key = False
            train_non_key = True
            
        elif self.training_strategy == 'joint':
            # Train both paths together
            print(f"  Training mode: Joint training (both paths)")
            self._set_decoder_trainable(True)
            train_key = True
            train_non_key = True
        else:
            raise ValueError(f"Unknown training strategy: {self.training_strategy}")
        
        total_loss = 0.0
        total_loss_key = 0.0
        total_loss_non_key = 0.0
        
        for batch_idx, batch in enumerate(self.dataloader):
            img_key, target_key, img_non_key, target_non_key = batch

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
            
            if train_key:
                outputs_key = self.model.forward_key_frame(img_key, target_key)
                loss_dict_key, key_indices = self.criterion(
                    outputs_key, target_key, return_indices=True
                )

                # Remove all aux loss
                final_loss_dict = {}
                for k, v in loss_dict_key.items():
                    # 1. Drop Multi-layer / Aux losses (keys ending in a number like _0, _1, _2)
                    if k[-1].isdigit():
                        continue
                        
                    # 2. Drop Denoising (DN) losses (keys containing 'dn')
                    if 'dn' in k:
                        continue
                        
                    # 3. Keep ONLY the final layer's main outputs
                    # (Usually 'loss_vfl', 'loss_bbox', 'loss_giou')
                    final_loss_dict[k] = v
    
                loss_key = sum(final_loss_dict.values())
                loss = loss_key
                loss_key_value = loss_key.item()

            if train_non_key:
                if not train_key:
                    with torch.no_grad():
                        outputs_key = self.model.forward_key_frame(img_key, target_key)
                        _, key_indices = self.criterion(
                            outputs_key, target_key, return_indices=True
                        )

                # 1. THE TEACHER: Run non-key image through frozen heavy encoder
                with torch.no_grad():
                    teacher_backbone_feats = self.model.backbone(img_non_key)
                    teacher_c3, teacher_c4, teacher_c5 = teacher_backbone_feats[-3:]
                    teacher_ccff = self.model.encoder([teacher_c3, teacher_c4, teacher_c5])
                
                # 2. Get standard outputs and fused student features
                outputs_non_key, student_fused = self.model.forward_non_key_frame(
                    img_non_key, target_non_key, return_fused=True
                )
                
                # 3. Calculate standard bounding box loss
                loss_dict_non_key = self.criterion(outputs_non_key, target_non_key, cached_indices=key_indices)
                loss_non_key_bbox = sum(loss_dict_non_key.values())
                
                # 4. KNOWLEDGE DISTILLATION GATE
                # Only run KD if we are explicitly in freeze_key mode (Steps 3 & 4)
                if self.training_strategy == 'freeze_key' and not self.same_frame:
                    
                    # Run teacher and get perfect features
                    with torch.no_grad():
                        teacher_backbone_feats = self.model.backbone(img_non_key)
                        teacher_c3, teacher_c4, teacher_c5 = teacher_backbone_feats[-3:]
                        teacher_ccff = self.model.encoder([teacher_c3, teacher_c4, teacher_c5])
                    
                    # Calculate KD Loss
                    loss_kd = 0.0
                    for s_feat, t_feat in zip(student_fused, teacher_ccff):
                        loss_kd += torch.nn.functional.smooth_l1_loss(s_feat, t_feat)
                    
                    # Route the loss based on the phase
                    if self.kd_only:
                        # Step 2: 100% KD, ignore BBox
                        loss_non_key = loss_kd
                    else:
                        # Step 3: Mostly BBox, slight KD regularization
                        loss_non_key = loss_non_key_bbox + (0.1 * loss_kd)
                else:
                    # Step 1 (Same Frame / Joint): Pure BBox loss, no KD interference
                    loss_non_key = loss_non_key_bbox

                loss_non_key_value = loss_non_key.item()
                
                if loss is None:
                    loss = self.lambda_non_key * loss_non_key
                else:
                    loss = loss + self.lambda_non_key * loss_non_key
            
            if loss is not None:
                loss.backward()
                if self.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 
                        max_norm=self.clip_max_norm
                    )
                self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item() if loss is not None else 0.0
            total_loss_key += loss_key_value
            total_loss_non_key += loss_non_key_value
            
            # Logging
            if batch_idx % self.print_freq == 0:
                if train_key and train_non_key:
                    print(f"Epoch [{epoch+1}] Batch [{batch_idx}/{len(self.dataloader)}] "
                        f"Loss: {loss.item():.4f} "
                        f"(Key: {loss_key_value:.4f}, Non-Key: {loss_non_key_value:.4f})")
                elif train_key:
                    print(f"Epoch [{epoch+1}] Batch [{batch_idx}/{len(self.dataloader)}] "
                        f"Loss: {loss.item():.4f} (Key only)")
                else:
                    print(f"Epoch [{epoch+1}] Batch [{batch_idx}/{len(self.dataloader)}] "
                        f"Loss: {loss.item():.4f} (Non-Key only)")
        
        # Average losses
        avg_loss = total_loss / len(self.dataloader)
        avg_loss_key = total_loss_key / len(self.dataloader)
        avg_loss_non_key = total_loss_non_key / len(self.dataloader)
        
        return {
            'loss': avg_loss,
            'loss_key': avg_loss_key,
            'loss_non_key': avg_loss_non_key,
            'train_key': train_key,
            'train_non_key': train_non_key,
        }
    
    @torch.no_grad()
    def evaluate(self, val_dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Evaluate model on validation set using official COCO API.
        Computes mAP for both Key and Non-Key frames independently.
        """
        import gc
        self.model.eval()
        if self.criterion is not None:
            self.criterion.eval()
            
        # Ensure dataset has COCO object (as we fixed in phase1_dataset.py)
        if not hasattr(val_dataloader.dataset, 'coco'):
            print("Error: Dataset missing .coco attribute. Cannot run COCO evaluation.")
            return {'mAP': 0.0}
        
        coco_gt = val_dataloader.dataset.coco
        
        print(f"\n{'='*80}")
        print(f"Evaluating Epoch {epoch}...")

        results_key = []
        results_non_key = []
        
        for batch_idx, (img_key, target_key, img_non_key, target_non_key) in enumerate(val_dataloader):
            if self.same_frame:
                img_non_key = img_key.clone()
                target_non_key = target_key

            img_key = img_key.to(self.device)
            # --- KEY FRAME ---
            outputs_key = self.model.forward_key_frame(img_key, None)
            orig_sizes_k = torch.stack([t["orig_size"] for t in target_key], dim=0).to(self.device)
            res_key = self.postprocessor(outputs_key, orig_sizes_k)
            self._accumulate(results_key, target_key, res_key)
            # --- NON-KEY FRAME ---
            has_non_key = (img_non_key is not None) and (len(img_non_key) > 0)
            
            if has_non_key:
                if isinstance(img_non_key, torch.Tensor):
                    img_non_key = img_non_key.to(self.device)
                
                outputs_nk = self.model.forward_non_key_frame(img_non_key, None)
                orig_sizes_nk = torch.stack([t["orig_size"] for t in target_non_key], dim=0).to(self.device)
                res_nk = self.postprocessor(outputs_nk, orig_sizes_nk)
                self._accumulate(results_non_key, target_non_key, res_nk)
            
            if batch_idx % 100 == 0:
                print(f"  Processed {batch_idx}/{len(val_dataloader)} batches")

        stats = {}
        print("\n>>> KEY FRAME RESULTS:")
        k_stats = self._run_coco_eval(coco_gt, results_key)
        stats.update({f'key_{k}': v for k, v in k_stats.items()})
        
        if results_non_key:
            print("\n>>> NON-KEY FRAME RESULTS:")
            nk_stats = self._run_coco_eval(coco_gt, results_non_key)
            stats.update({f'non_key_{k}': v for k, v in nk_stats.items()})
            print(f"Gap (Key - NonKey) mAP@50: {k_stats['mAP@50'] - nk_stats['mAP@50']:.4f}")

        del results_key
        del results_non_key
        gc.collect()

        print(f"{'='*80}\n")
        return stats

    def _accumulate(self, results_list, targets, outputs):
        """Helper to convert tensor outputs to COCO list format"""
        for target, output in zip(targets, outputs):
            image_id = int(target['image_id'].item())
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

    def _run_coco_eval(self, coco_gt, results_list):
        """Helper to run COCOeval on a list of results"""
        import gc
        if not results_list:
            print("  No predictions generated.")
            return {'mAP': 0.0, 'mAP@50': 0.0, 'mAP@75': 0.0}
        coco_dt = coco_gt.loadRes(results_list)
        
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
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
        }
        
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Also save as latest
        latest_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best model based on total loss
        best_path = self.output_dir / 'best_model.pth'
        if not best_path.exists():
            torch.save(checkpoint, best_path)
        else:
            best_checkpoint = torch.load(best_path, weights_only=False)
            if metrics['loss'] < best_checkpoint['metrics']['loss']:
                torch.save(checkpoint, best_path)
                print(f"✓ New best model saved!")

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
    reuse_queries = cfg.yaml_cfg.get('reuse_queries', False)
    
    # Create temporal model
    temporal_model = TemporalRTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        num_classes=cfg.yaml_cfg.get('num_classes', 80),
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        use_lightweight_decoder=use_lightweight_decoder,
        reuse_queries=reuse_queries,
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


class DebugSubsetDataLoader:
    """Wrapper to limit dataloader to N batches for debugging"""
    def __init__(self, dataloader, max_batches):
        self.dataloader = dataloader
        self.max_batches = max_batches
        self.dataset = dataloader.dataset
        self.collate_fn = dataloader.collate_fn
        
    def __iter__(self):
        for i, batch in enumerate(self.dataloader):
            if i >= self.max_batches:
                break
            yield batch
    
    def __len__(self):
        return min(self.max_batches, len(self.dataloader))


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--training_strategy', type=str, default='joint',
                       choices=['alternate', 'freeze_key', 'joint'],
                       help='alternate, freeze_key, or joint')
    parser.add_argument('--alternate_interval', type=int, default=1,
                       help='Alternate strategy only: switch every N epochs')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to checkpoint (auto-detects key-only vs full temporal)')
    parser.add_argument('--eval_only', action='store_true',
                       help='Skip training and only run evaluation on the provided weights')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_batches', type=int, default=10,
                       help='Number of batches to use in debug mode')

    # Optional
    parser.add_argument('--resume', '-r', type=str, default=None, 
                       help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, 
                       help='Number of epochs (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory (overrides config)')
    
    parser.add_argument('--same_frame', action='store_true',
                       help='Diagnostic: Force non-key frame to be identical to key frame')
    
    parser.add_argument('--init_weights', action='store_true', 
                       help='Warm start: Copy pretrained key decoder weights to non-key decoder')
    
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

        if args.init_weights:
            print("\n=> Initializing Lightweight Decoder from Heavy Decoder...")
            if hasattr(model, 'lightweight_decoder') and model.lightweight_decoder is not None:
                # Get the state dict of the LAST layer of the heavy transformer
                heavy_last_layer_state = model.decoder.decoder.layers[-1].state_dict()
                
                # Force it into the FIRST (and only) layer of the lightweight transformer
                model.lightweight_decoder.decoder.layers[0].load_state_dict(heavy_last_layer_state)
                
                print("   ✅ Successfully copied pre-trained heavy transformer layer!")
        
        if args.pretrained:
            print(f"\n=> Inspecting weights from: {args.pretrained}")
            checkpoint = torch.load(args.pretrained, map_location=device, weights_only=False)
            
            # Handle different checkpoint dictionary structures
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
            
            # Auto-detect if these weights contain our temporal modules
            is_temporal = any('fusion_blocks' in k for k in state_dict.keys())
            
            if is_temporal:
                print("   [Auto-Detect] Full Temporal RT-DETR weights found. Loading strictly...")
                model.load_state_dict(state_dict, strict=True)
            else:
                print("   [Auto-Detect] Standard Key-Frame RT-DETR weights found. Warm-starting...")
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"   Missing keys (expected for new temporal modules): {len(missing_keys)}")
            
            print("   Success!")

        if args.init_weights:
            model.lightweight_decoder.load_state_dict(model.decoder.state_dict(), strict=False)
            print("Non-key decoder initialization complete.")
        
        # Get config values (with overrides)
        epochs = args.epochs if args.epochs is not None else getattr(cfg, 'epoches', 50)
        output_dir = args.output_dir if args.output_dir is not None else getattr(cfg, 'output_dir', 'output/phase1_virat')
        seed = args.seed if args.seed is not None else getattr(cfg, 'seed', 42)
        lambda_non_key = getattr(cfg, 'lambda_non_key', 0.5)
        print_freq = getattr(cfg, 'print_freq', 50)
        checkpoint_freq = getattr(cfg, 'checkpoint_freq', 5)
        clip_max_norm = getattr(cfg, 'clip_max_norm', 0.1)

        summary_dir = getattr(cfg, 'summary_dir', os.path.join(output_dir, 'summary'))
        writer = SummaryWriter(log_dir=summary_dir)
        print(f"  Summary dir:      {summary_dir}")
        
        # Debug mode overrides
        if args.debug:
            epochs = min(epochs, 3)
            print_freq = 5
            checkpoint_freq = 1
        
        print(f"\nConfiguration:")
        print(f"  Epochs:           {epochs}")
        print(f"  Training strategy: {args.training_strategy}")
        print(f"  Lambda (non-key): {lambda_non_key}")
        print(f"  Output dir:       {output_dir}")
        print(f"  Seed:             {seed}")
        if args.debug:
            print(f"  Debug mode:      {args.debug_batches} batches/epoch")
        
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
        if args.debug:
            train_dataloader = DebugSubsetDataLoader(train_dataloader, args.debug_batches)
            print(f"  DataLoader loaded (DEBUG MODE - {args.debug_batches} batches)")
        else:
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
    
    trainable_params = []
        
    # Iterate through every single parameter in the network by its string name
    for name, param in model.named_parameters():
        if 'fusion_blocks' in name:
                param.requires_grad = True
                trainable_params.append(param)
        else:
            param.requires_grad = False

    # 3. BUILD the custom optimizer
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
    if args.debug:
            val_dataloader = DebugSubsetDataLoader(val_dataloader, args.debug_batches)
    
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
        alternate_interval=args.alternate_interval,
        same_frame=args.same_frame
    )
    
    if args.eval_only:
        print("\n" + "="*80)
        print("Executing EVALUATION ONLY Mode...")
        print("="*80)
        if val_dataloader is None:
            print("❌ Error: No validation dataloader found in config.")
            sys.exit(1)
            
        # Run evaluation with a dummy epoch 0
        stats = trainer.evaluate(val_dataloader, epoch=0)
        
        print("\nFinal Evaluation Stats:")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}")
        return  # Exit the main function gracefully
    
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
        if metrics['train_non_key']:
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
