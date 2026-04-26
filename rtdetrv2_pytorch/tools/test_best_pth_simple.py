"""
Phase 1 Training Script for Temporal RT-DETR
Run with: python rtdetrv2_pytorch/tools/phase1_training.py -c rtdetrv2_pytorch/configs/rtdetrv2/phase1_virat_r18vd.yml --pretrained best.pth --training_strategy freeze_key
"""

import os 
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

# Import to register classes BEFORE loading config
from src.zoo.temporal_rtdetr import TemporalRTDETR, ViratTemporalDataset
from src.core import YAMLConfig

from pycocotools.cocoeval import COCOeval
import numpy as np
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
        postprocessor: nn.Module,
        device: torch.device,
    ):
        self.model = model
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.device = device

    @torch.no_grad()
    def evaluate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        self.criterion.eval()
        
        # 1. Setup: Get the COCO Ground Truth object directly
        #    (Relies on 'self.coco' existing in dataset, which we fixed earlier)
        coco_gt = val_dataloader.dataset.coco
        
        # List to store all predictions in standard COCO format
        results_list = []
        
        print("\nStarting evaluation...")

        for batch_idx, (img_key, target_key, _, _) in enumerate(val_dataloader):
            img_key = img_key.to(self.device)
            # We don't need target_key on GPU for eval, just for size info
            
            # Forward pass
            outputs_key = self.model.forward_key_frame(img_key, None)
            
            # Post-process (Rescale to 1920x1080)
            orig_target_sizes = torch.stack([t["orig_size"] for t in target_key], dim=0).to(self.device)
            results_key = self.postprocessor(outputs_key, orig_target_sizes)
            
            # Accumulate results
            for target, output in zip(target_key, results_key):
                image_id = target['image_id'].item()
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                for i in range(len(scores)):
                    x1, y1, x2, y2 = boxes[i]
                    w, h = x2 - x1, y2 - y1
                    
                    results_list.append({
                        "image_id": int(image_id),
                        "category_id": int(labels[i]),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(scores[i])
                    })

            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx}/{len(val_dataloader)} batches")

        # 2. Run Evaluation using official COCO API
        print(f"\nEvaluating {len(results_list)} predictions...")
        
        if len(results_list) == 0:
            print("Warning: No predictions generated!")
            return {}

        # Load results directly from memory
        coco_dt = coco_gt.loadRes(results_list)
        
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        return {}

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
    )
    
    return temporal_model, cfg


def load_pretrained_key_frame(model: TemporalRTDETR, pretrained_path: str, device: torch.device):
    """
    Load pretrained weights for key frame path
    """
    print(f"\nLoading pretrained key frame path from: {pretrained_path}")
    
    checkpoint = torch.load(pretrained_path, map_location=device)
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
    parser.add_argument('--pretrained', type=str, default="best.pth",
                       help='Path to pretrained key frame model')

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

        if args.pretrained:
            load_pretrained_key_frame(model, args.pretrained, device)
        
        # Get config values (with overrides)
        seed = 42

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    val_dataloader = cfg.val_dataloader if hasattr(cfg, 'val_dataloader') else None
    
    criterion = cfg.criterion
    postprocessor = cfg.postprocessor

    # Trainer
    trainer = Phase1Trainer(
        model=model,
        criterion=criterion,
        postprocessor = postprocessor,
        device=device,
    )

    trainer.evaluate(val_dataloader)

if __name__ == '__main__':
    main()
