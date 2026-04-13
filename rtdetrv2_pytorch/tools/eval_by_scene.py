"""
Real-Time Temporal Inference Simulator (Per-Scene Breakdown)
Simulates a live K-NK-K-NK streaming environment with Batch Size = 1. 
Tracks Latency, Peak VRAM, and COCO mAP grouped by individual Video Scenes.
"""

import os
import sys
import time
import argparse
import contextlib
import io
from pathlib import Path
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

def evaluate_map(coco_gt, results, img_ids=None, quiet=True):
    """Runs the pycocotools evaluation. Silences stdout if quiet=True to prevent terminal spam."""
    if not results:
        return 0.0
        
    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
        coco_dt = coco_gt.loadRes(results)
        evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
        if img_ids is not None:
            evaluator.params.imgIds = img_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        
    return evaluator.stats[0] # Returns mAP@50:95

def main():
    parser = argparse.ArgumentParser(description="Evaluate Temporal RT-DETR per Scene")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config yml')
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--warmup', type=int, default=10, help='Ignore first N batches for timing/memory')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Deployment Device: {device}")
    
    # 1. Load the raw config and force batch_size=1
    cfg = YAMLConfig(args.config)
    if 'val_dataloader' in cfg.yaml_cfg:
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
    
    val_dataloader = cfg.val_dataloader
    coco_gt = val_dataloader.dataset.coco
    postprocessor = cfg.postprocessor
    
    # 4. Map Image IDs to Video Scenes
    print("Mapping frames to Video Scenes...")
    img_id_to_scene = {}
    scene_to_img_ids = {}
    
    for img_id, img_info in coco_gt.imgs.items():
        file_name = img_info['file_name']
        # Extract the video folder name (e.g., 'VIRAT_S_000001')
        scene_name = Path(file_name).parts[0] if len(Path(file_name).parts) > 1 else "default_scene"
        
        img_id_to_scene[img_id] = scene_name
        if scene_name not in scene_to_img_ids:
            scene_to_img_ids[scene_name] = []
        scene_to_img_ids[scene_name].append(img_id)

    # Initialize Scene Stats Dictionary
    scene_stats = {
        scene: {
            'k_time': 0.0, 'k_mem': 0.0, 'k_frames': 0,
            'nk_time': 0.0, 'nk_mem': 0.0, 'nk_frames': 0,
            'res_key': [], 'res_nk': []
        } for scene in scene_to_img_ids.keys()
    }

    # 5. INFERENCE LOOP
    print(f"\n--- INITIATING STREAM SIMULATION OVER {len(scene_stats)} SCENES ---")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc="Streaming Video")):
            img_key, target_key, img_non_key, target_non_key = batch
            
            # Identify current scene
            current_img_id = int(target_key[0]['image_id'].item())
            scene = img_id_to_scene[current_img_id]
            stats = scene_stats[scene]
            
            # ==========================================
            # STEP 1: KEY FRAME (Heavy Inference)
            # ==========================================
            img_key = img_key.to(device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            
            out_k = model.forward_key_frame(img_key, None)
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            if i >= args.warmup:
                stats['k_time'] += (t1 - t0)
                stats['k_frames'] += 1
                if torch.cuda.is_available():
                    stats['k_mem'] += torch.cuda.max_memory_allocated() / (1024 ** 2)
                
            orig_sizes_k = torch.stack([t["orig_size"] for t in target_key], dim=0).to(device)
            format_coco(target_key, postprocessor(out_k, orig_sizes_k), stats['res_key'])
            
            # ==========================================
            # STEP 2: NON-KEY FRAME (Light Inference)
            # ==========================================
            if img_non_key is not None and len(img_non_key) > 0:
                img_non_key = img_non_key.to(device)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t2 = time.perf_counter()
                
                out_nk = model.forward_non_key_frame(img_non_key, None)
                
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t3 = time.perf_counter()
                
                if i >= args.warmup:
                    stats['nk_time'] += (t3 - t2)
                    stats['nk_frames'] += 1
                    if torch.cuda.is_available():
                        stats['nk_mem'] += torch.cuda.max_memory_allocated() / (1024 ** 2)
                    
                orig_sizes_nk = torch.stack([t["orig_size"] for t in target_non_key], dim=0).to(device)
                format_coco(target_non_key, postprocessor(out_nk, orig_sizes_nk), stats['res_nk'])

    # 6. PER-SCENE REPORT GENERATION
    print("\n" + "="*80)
    print("PER-SCENE PERFORMANCE REPORT")
    print("="*80)
    
    global_k_time, global_k_mem, global_k_frames = 0.0, 0.0, 0
    global_nk_time, global_nk_mem, global_nk_frames = 0.0, 0.0, 0
    global_res_key, global_res_nk = [], []

    for scene, stats in scene_stats.items():
        # Aggregate to globals
        global_k_time += stats['k_time']
        global_k_mem += stats['k_mem']
        global_k_frames += stats['k_frames']
        global_nk_time += stats['nk_time']
        global_nk_mem += stats['nk_mem']
        global_nk_frames += stats['nk_frames']
        global_res_key.extend(stats['res_key'])
        global_res_nk.extend(stats['res_nk'])
        
        # Calculate local averages
        avg_k_t = (stats['k_time'] / stats['k_frames']) * 1000 if stats['k_frames'] else 0
        avg_nk_t = (stats['nk_time'] / stats['nk_frames']) * 1000 if stats['nk_frames'] else 0
        avg_k_m = (stats['k_mem'] / stats['k_frames']) if stats['k_frames'] > 0 else 0
        avg_nk_m = (stats['nk_mem'] / stats['nk_frames']) if stats['nk_frames'] > 0 else 0
        speedup = avg_k_t / avg_nk_t if avg_nk_t > 0 else 0
        
        # Evaluate local mAP (Silently)
        img_ids = scene_to_img_ids[scene]
        map_k = evaluate_map(coco_gt, stats['res_key'], img_ids, quiet=True)
        map_nk = evaluate_map(coco_gt, stats['res_nk'], img_ids, quiet=True)
        map_comb = evaluate_map(coco_gt, stats['res_key'] + stats['res_nk'], img_ids, quiet=True)
        
        # Print Scene Block
        print(f"[ SCENE: {scene} ]")
        print(f"  Combined Stream mAP : {map_comb:.4f}")
        print(f"  Key Pathway         : mAP={map_k:.4f} | Latency={avg_k_t:.1f}ms | VRAM={avg_k_m:.0f}MB")
        print(f"  Non-Key Pathway     : mAP={map_nk:.4f} | Latency={avg_nk_t:.1f}ms | VRAM={avg_nk_m:.0f}MB")
        if speedup > 0:
            print(f"  Local Speedup       : {speedup:.2f}x Faster")
        print("-" * 80)

    # 7. GLOBAL SUMMARY REPORT
    avg_gk_t = (global_k_time / global_k_frames) * 1000 if global_k_frames else 0
    avg_gnk_t = (global_nk_time / global_nk_frames) * 1000 if global_nk_frames else 0
    avg_gk_m = (global_k_mem / global_k_frames) if global_k_frames > 0 else 0
    avg_gnk_m = (global_nk_mem / global_nk_frames) if global_nk_frames > 0 else 0
    
    print("\n" + "="*80)
    print("GLOBAL HARDWARE SUMMARY (Averaged Across All Scenes)")
    print("="*80)
    
    # Global mAP allows pycocotools to print its standard output blocks (quiet=False)
    print("\nCalculating Final Global mAP...")
    g_map_comb = evaluate_map(coco_gt, global_res_key + global_res_nk, quiet=False)
    
    print("\n[ AGGREGATE SYSTEM METRICS ]")
    print(f"  Global System Accuracy:  {g_map_comb:.4f} mAP")
    print(f"  Average Key Latency:     {avg_gk_t:.2f} ms")
    print(f"  Average Non-Key Latency: {avg_gnk_t:.2f} ms")
    
    if avg_gnk_t > 0 and avg_gnk_m > 0:
        print(f"\n[ HARDWARE SAVINGS ]")
        print(f"  Speedup:         {avg_gk_t / avg_gnk_t:.2f}x Faster")
        print(f"  Memory Savings:  {100 - ((avg_gnk_m / avg_gk_m) * 100):.1f}% Less VRAM")
    print("="*80)

if __name__ == '__main__':
    main()