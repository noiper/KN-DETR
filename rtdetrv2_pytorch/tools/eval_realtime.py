"""
Real-Time Temporal Inference Simulator
Simulates a live continuous streaming environment with configurable K-NK ratios.
Tracks Latency, Peak VRAM Memory Allocation, and Combined COCO mAP.
"""

import os
import sys
import time
import argparse
import contextlib
import io
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

def propagate_key_results_to_non_key_targets(key_results, non_key_targets):
    """
    Reuse key-frame detections directly as non-key predictions.
    Keeps boxes/scores/labels and only changes destination image IDs via format_coco.
    """
    propagated = []
    for output in key_results:
        propagated.append({
            'boxes': output['boxes'].clone(),
            'scores': output['scores'].clone(),
            'labels': output['labels'].clone(),
        })
    if len(propagated) != len(non_key_targets):
        raise RuntimeError(
            f"Batch size mismatch for propagation: key={len(propagated)} vs non-key={len(non_key_targets)}"
        )
    return propagated

def evaluate_map(coco_gt, results, title):
    """Runs pycocotools evaluation and returns (mAP, mAP50)."""
    if not results:
        return 0.0, 0.0
    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # STRICTLY LIMIT EVALUATION TO THE IMAGES PREDICTED
    # Prevents artificial deflation when evaluating partial streams
    predicted_img_ids = sorted(list(set([res['image_id'] for res in results])))
    evaluator.params.imgIds = predicted_img_ids
    
    evaluator.evaluate()
    evaluator.accumulate()
    # COCOeval fills `stats` during summarize(); silence the default table output.
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.summarize()
    if len(evaluator.stats) < 2:
        return 0.0, 0.0
    return evaluator.stats[0], evaluator.stats[1]

def main():
    parser = argparse.ArgumentParser(description="Evaluate Temporal RT-DETR in Real-Time Simulation")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config yml')
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--warmup', type=int, default=10, help='Ignore first N batches for timing/memory')
    parser.add_argument('--nk_per_key', type=int, default=1, 
                        help='Number of Non-Key frames per Key frame. 1 = (K, NK), 2 = (K, NK, NK), etc.')
    parser.add_argument('--baseline', action='store_true',
                        help='Baseline: reuse key-frame detections directly for non-key frames')
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
        reuse_position=cfg.yaml_cfg.get('reuse_position', 0),
    ).to(device)
    
    # 3. Load Weights
    print(f"Loading weights from {args.weights}...")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint)), strict=True)
    model.eval()
    
    # --- PHYSICAL DATALOADER REBUILD FOR BATCH_SIZE=1 ---
    base_val_loader = cfg.val_dataloader
    from torch.utils.data import DataLoader
    
    print("Rebuilding validation dataloader to force batch_size=1...")
    val_dataloader = DataLoader(
        dataset=base_val_loader.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=base_val_loader.num_workers,
        collate_fn=base_val_loader.collate_fn,
        drop_last=False
    )
    # -----------------------------------------------------
    coco_gt = val_dataloader.dataset.coco
    postprocessor = cfg.postprocessor
    print(f"Non-key mode: {'baseline (reuse key detections)' if args.baseline else 'model forward'}")
    
    res_key = []
    res_nk = []
    latest_key_results = None
    
    metrics = {
        'k_time': 0.0, 'k_mem': 0.0, 'k_frames': 0,
        'nk_time': 0.0, 'nk_mem': 0.0, 'nk_frames': 0
    }

    # The length of one full cycle (e.g., K-NK-NK has a cycle length of 3)
    cycle_len = args.nk_per_key + 1

    print(f"\n--- INITIATING REAL-TIME STREAM SIMULATION (1 Key : {args.nk_per_key} Non-Key) ---")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc="Streaming Video")):
            img_key, target_key, img_non_key, target_non_key = batch
            
            # Determine where we are in the K-NK cycle
            step = i % cycle_len
            
            if step == args.nk_per_key:
                # SKIP BATCH: The img_non_key in this batch is the next cycle's Key frame.
                # If we evaluate it here, we double-count it.
                continue
            
            # ==========================================
            # KEY FRAME ARRIVES (Only on step 0)
            # ==========================================
            if step == 0:
                img_key = img_key.to(device)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                
                t0 = time.perf_counter()
                
                # Forward pass natively caches the features for upcoming Non-Key frames
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
                latest_key_results = postprocessor(out_k, orig_sizes_k)
                format_coco(target_key, latest_key_results, res_key)
            
            # ==========================================
            # NON-KEY FRAME ARRIVES (Every step except the skipped one)
            # ==========================================
            if img_non_key is not None and len(img_non_key) > 0:
                if args.baseline:
                    if latest_key_results is None:
                        raise RuntimeError("No cached key results available for non-key propagation")
                    t2 = time.perf_counter()
                    res_nk_batch = propagate_key_results_to_non_key_targets(latest_key_results, target_non_key)
                    t3 = time.perf_counter()
                else:
                    img_non_key = img_non_key.to(device)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    
                    t2 = time.perf_counter()
                    
                    # Relies on the cache stored during step == 0
                    out_nk = model.forward_non_key_frame(img_non_key, None)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    
                    orig_sizes_nk = torch.stack([t["orig_size"] for t in target_non_key], dim=0).to(device)
                    res_nk_batch = postprocessor(out_nk, orig_sizes_nk)
                
                if i >= args.warmup:
                    if not args.baseline:
                        metrics['nk_time'] += (t3 - t2)
                        metrics['nk_frames'] += 1
                        if torch.cuda.is_available():
                            metrics['nk_mem'] += torch.cuda.max_memory_allocated() / (1024 ** 2)
                
                format_coco(target_non_key, res_nk_batch, res_nk)

    # Combine results
    results_combined = res_key + res_nk

    # Calculate Averages
    avg_k_time = (metrics['k_time'] / metrics['k_frames']) * 1000 if metrics['k_frames'] else 0
    avg_nk_time = (metrics['nk_time'] / metrics['nk_frames']) * 1000 if metrics['nk_frames'] else 0
    
    avg_k_mem = (metrics['k_mem'] / metrics['k_frames']) if metrics['k_frames'] > 0 else 0
    avg_nk_mem = (metrics['nk_mem'] / metrics['nk_frames']) if metrics['nk_frames'] > 0 else 0

    map_k, map50_k = evaluate_map(coco_gt, res_key, "HEAVY KEY MODEL ONLY")
    map_nk, map50_nk = evaluate_map(coco_gt, res_nk, "LIGHTWEIGHT NON-KEY MODEL ONLY")
    combined_map, combined_map50 = evaluate_map(coco_gt, results_combined, "COMBINED OVERALL AVERAGE")

    print("\n" + "="*70)
    print(f"FINAL SUMMARY (Level {args.nk_per_key})")
    print("="*70)
    print(f"Key      mAP: {map_k:.4f} | mAP50: {map50_k:.4f}")
    print(f"Non-Key  mAP: {map_nk:.4f} | mAP50: {map50_nk:.4f}")
    print(f"Combined mAP: {combined_map:.4f} | mAP50: {combined_map50:.4f}")
    print("-"*70)
    print(f"Key Latency: {avg_k_time:.2f} ms | Key VRAM: {avg_k_mem:.2f} MB")
    if not args.baseline:
        print(f"Non-Key Latency: {avg_nk_time:.2f} ms | Non-Key VRAM: {avg_nk_mem:.2f} MB")
        if avg_nk_time > 0:
            print(f"Speedup (Key/Non-Key): {avg_k_time / avg_nk_time:.2f}x")
    print("="*70)

if __name__ == '__main__':
    main()
