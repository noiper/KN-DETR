"""
Real-Time Temporal Inference Simulator
Simulates a live K-NK-K-NK streaming environment with Batch Size = 1. 
Tracks Latency, Peak VRAM Memory Allocation, and Combined COCO mAP.
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

def evaluate_map(coco_gt, results, title):
    """Runs the pycocotools evaluation and prints the summary"""
    print(f"\n>>> EVALUATING: {title} <<<")
    if not results:
        print("    No predictions generated.")
        return 0.0
    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator.stats[0] # Returns mAP@50:95

def main():
    parser = argparse.ArgumentParser(description="Evaluate Temporal RT-DETR in Real-Time Simulation")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config yml')
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--warmup', type=int, default=10, help='Ignore first N batches for timing/memory')
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
        
        # Also enforce in the dataset wrapper if it exists there
        if 'dataset' in cfg.yaml_cfg['val_dataloader'] and hasattr(cfg, 'val_dataloader'):
            pass # Dictionary update is sufficient for YAMLConfig instantiation
    # ---------------------------------------------------------
    
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
        reuse_queries=cfg.yaml_cfg.get('reuse_queries', False),
    ).to(device)
    
    # 3. Load Weights
    print(f"Loading weights from {args.weights}...")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint)), strict=True)
    model.eval()
    
    # Because we modified cfg.yaml_cfg above, this will build a dataloader with batch_size=1
    val_dataloader = cfg.val_dataloader
    coco_gt = val_dataloader.dataset.coco
    postprocessor = cfg.postprocessor
    
    # Accumulator for Combined mAP only
    results_combined = []
    
    # Tracking Metrics
    metrics = {
        'k_time': 0.0, 'k_mem': 0.0, 'k_frames': 0,
        'nk_time': 0.0, 'nk_mem': 0.0, 'nk_frames': 0
    }

    print("\n--- INITIATING REAL-TIME STREAM SIMULATION ---")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc="Streaming Video (bs=1)")):
            img_key, target_key, img_non_key, target_non_key = batch
            
            # ==========================================
            # STEP 1: KEY FRAME ARRIVES (Heavy Inference)
            # ==========================================
            img_key = img_key.to(device)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            t0 = time.perf_counter()
            
            # Forward pass inherently updates the temporal memory cache
            out_k = model.forward_key_frame(img_key, None)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            if i >= args.warmup:
                metrics['k_time'] += (t1 - t0)
                metrics['k_frames'] += 1
                if torch.cuda.is_available():
                    metrics['k_mem'] += torch.cuda.max_memory_allocated() / (1024 ** 2)
                
            orig_sizes_k = torch.stack([t["orig_size"] for t in target_key], dim=0).to(device)
            format_coco(target_key, postprocessor(out_k, orig_sizes_k), results_combined)
            
            # ==========================================
            # STEP 2: NON-KEY FRAME ARRIVES (Light Inference)
            # ==========================================
            if img_non_key is not None and len(img_non_key) > 0:
                img_non_key = img_non_key.to(device)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                
                t2 = time.perf_counter()
                
                # Explicitly grabs the cache from the Key pass and computes the shortcut
                out_nk = model.forward_non_key_frame(img_non_key, None)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t3 = time.perf_counter()
                
                if i >= args.warmup:
                    metrics['nk_time'] += (t3 - t2)
                    metrics['nk_frames'] += 1
                    if torch.cuda.is_available():
                        metrics['nk_mem'] += torch.cuda.max_memory_allocated() / (1024 ** 2)
                    
                orig_sizes_nk = torch.stack([t["orig_size"] for t in target_non_key], dim=0).to(device)
                format_coco(target_non_key, postprocessor(out_nk, orig_sizes_nk), results_combined)

    # Calculate Averages
    avg_k_time = (metrics['k_time'] / metrics['k_frames']) * 1000 if metrics['k_frames'] else 0
    avg_nk_time = (metrics['nk_time'] / metrics['nk_frames']) * 1000 if metrics['nk_frames'] else 0
    
    avg_k_mem = (metrics['k_mem'] / metrics['k_frames']) if metrics['k_frames'] > 0 else 0
    avg_nk_mem = (metrics['nk_mem'] / metrics['nk_frames']) if metrics['nk_frames'] > 0 else 0

    print("\n" + "="*60)
    print("REAL-TIME PERFORMANCE REPORT")
    print("="*60)
    
    # Only report the true, combined mAP
    combined_map = evaluate_map(coco_gt, results_combined, "COMBINED OVERALL AVERAGE")
    
    print("\n" + "="*60)
    print("FINAL SUMMARY (Averaged per Frame):")
    print("="*60)
    print(f"  Combined System Accuracy (mAP):  {combined_map:.4f}\n")
    
    print(f"[ KEY FRAME - Heavy Pathway ]")
    print(f"  Latency:         {avg_k_time:.2f} ms")
    print(f"  Peak VRAM:       {avg_k_mem:.2f} MB\n")
    
    print(f"[ NON-KEY FRAME - Light Pathway ]")
    print(f"  Latency:         {avg_nk_time:.2f} ms")
    print(f"  Peak VRAM:       {avg_nk_mem:.2f} MB\n")
    
    if avg_nk_time > 0 and avg_nk_mem > 0:
        print(f"[ HARDWARE SAVINGS ]")
        print(f"  Speedup:         {avg_k_time / avg_nk_time:.2f}x Faster")
        print(f"  Memory Savings:  {100 - ((avg_nk_mem / avg_k_mem) * 100):.1f}% Less VRAM")
    print("="*60)

if __name__ == '__main__':
    main()