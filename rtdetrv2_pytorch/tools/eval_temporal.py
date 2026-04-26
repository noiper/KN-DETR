"""
Standalone Temporal Inference & Evaluation Script
Executes K-NK-K-NK alternating pattern, reports COCO mAP, and tracks pure inference latency.
"""

import os
import sys
import time
import argparse
import torch
from tqdm import tqdm

# Ensure python path is correct when run from terminal
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.zoo.temporal_rtdetr import TemporalRTDETR
from pycocotools.cocoeval import COCOeval

def build_model_from_config(config_path: str, device: torch.device):
    """Build TemporalRTDETR model from config"""
    cfg = YAMLConfig(config_path)
    base_model = cfg.model.to(device)
    
    backbone = base_model.backbone
    encoder = base_model.encoder if hasattr(base_model, 'encoder') else None
    decoder = base_model.decoder if hasattr(base_model, 'decoder') else None
    
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
    
    use_lightweight_decoder = cfg.yaml_cfg.get('use_lightweight_decoder', False)
    reuse_position = cfg.yaml_cfg.get('reuse_position', 0)
    
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

def format_to_coco_results(targets, outputs, results_list):
    """Converts tensor outputs to the exact dictionary format required by COCOeval"""
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

def run_coco_eval(coco_gt, results_list, title):
    """Runs the pycocotools evaluation and prints the summary"""
    print(f"\n>>> EVALUATING: {title} <<<")
    if not results_list:
        print("    No predictions generated.")
        return 0.0
        
    coco_dt = coco_gt.loadRes(results_list)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval.stats[0] # Returns mAP@50:95

def main():
    parser = argparse.ArgumentParser(description="Evaluate Temporal RT-DETR")
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config yml')
    parser.add_argument('--weights', '-w', type=str, required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--warmup', type=int, default=10, help='Number of batches to ignore for latency metrics')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Build Model & Load Weights
    print(f"Building architecture from {args.config}...")
    model, cfg = build_model_from_config(args.config, device)
    
    print(f"Loading weights from {args.weights}...")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # 2. Get Validation Dataloader
    val_dataloader = cfg.val_dataloader
    postprocessor = cfg.postprocessor
    coco_gt = val_dataloader.dataset.coco
    
    print(f"Loaded validation set: {len(val_dataloader)} batches.")
    
    # 3. Setup Accumulators
    results_key = []
    results_non_key = []
    results_all = []
    
    # Latency tracking
    time_key = 0.0
    frames_key = 0
    time_non_key = 0.0
    frames_non_key = 0

    # 4. Alternating Inference Loop
    print("\nStarting K-NK-K-NK Temporal Inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Inference")):
            img_key, target_key, img_non_key, target_non_key = batch
            
            # --- PHASE A: KEY FRAME INFERENCE ---
            img_key = img_key.to(device)
            bs_key = img_key.shape[0]
            
            # Synchronize and measure pure forward pass latency
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            outputs_key = model.forward_key_frame(img_key, None)
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            if batch_idx >= args.warmup:
                time_key += (t1 - t0)
                frames_key += bs_key

            # Postprocessing (excluded from latency measurement)
            orig_sizes_k = torch.stack([t["orig_size"] for t in target_key], dim=0).to(device)
            res_key = postprocessor(outputs_key, orig_sizes_k)
            
            format_to_coco_results(target_key, res_key, results_key)
            format_to_coco_results(target_key, res_key, results_all)
            
            # --- PHASE B: NON-KEY FRAME INFERENCE ---
            has_non_key = (img_non_key is not None) and (len(img_non_key) > 0)
            if has_non_key:
                img_non_key = img_non_key.to(device)
                bs_non_key = img_non_key.shape[0]
                
                # Synchronize and measure pure forward pass latency
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t2 = time.perf_counter()
                
                outputs_nk = model.forward_non_key_frame(img_non_key, None)
                
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t3 = time.perf_counter()
                
                if batch_idx >= args.warmup:
                    time_non_key += (t3 - t2)
                    frames_non_key += bs_non_key

                # Postprocessing (excluded from latency measurement)
                orig_sizes_nk = torch.stack([t["orig_size"] for t in target_non_key], dim=0).to(device)
                res_nk = postprocessor(outputs_nk, orig_sizes_nk)
                
                format_to_coco_results(target_non_key, res_nk, results_non_key)
                format_to_coco_results(target_non_key, res_nk, results_all)

    # 5. Execute COCO Metrics
    print("\n" + "="*60)
    print("FINAL INFERENCE METRICS")
    print("="*60)
    
    map_key = run_coco_eval(coco_gt, results_key, "KEY FRAMES ONLY")
    map_nk = run_coco_eval(coco_gt, results_non_key, "NON-KEY FRAMES ONLY")
    map_all = run_coco_eval(coco_gt, results_all, "COMBINED OVERALL AVERAGE")
    
    # 6. Calculate Latency Metrics
    avg_latency_key = (time_key / frames_key) * 1000 if frames_key > 0 else 0.0
    avg_latency_nk = (time_non_key / frames_non_key) * 1000 if frames_non_key > 0 else 0.0
    speedup = avg_latency_key / avg_latency_nk if avg_latency_nk > 0 else 0.0
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  - Key Frame mAP:        {map_key:.4f}")
    print(f"  - Non-Key Frame mAP:    {map_nk:.4f}")
    print(f"  - Combined Avg mAP:     {map_all:.4f}")
    print("-" * 60)
    print(f"  - Key Frame Latency:    {avg_latency_key:.2f} ms/frame")
    print(f"  - Non-Key Latency:      {avg_latency_nk:.2f} ms/frame")
    print(f"  - Lightweight Speedup:  {speedup:.2f}x faster")
    print("="*60)

if __name__ == '__main__':
    main()
