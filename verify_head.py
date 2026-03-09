import torch
import argparse

def extract_state_dict(checkpoint):
    """Safely extract the state dict whether it's a custom save or RT-DETR save"""
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        return checkpoint['model']
    elif 'ema' in checkpoint:
        return checkpoint['ema']
    return checkpoint

def verify_internal_sharing(weights_path):
    print(f"Loading Checkpoint: {weights_path}")
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    sd = extract_state_dict(ckpt)

    # Find all lightweight keys
    lw_keys = {k: v for k, v in sd.items() if k.startswith('lightweight_decoder.')}
    if not lw_keys:
        print("\n❌ Error: Could not find 'lightweight_decoder' keys in this checkpoint.")
        return

    # Determine heavy decoder head indices
    heavy_score_keys = [k for k in sd.keys() if k.startswith('decoder.dec_score_head.') and 'weight' in k]
    last_head_idx = len(heavy_score_keys) - 1

    # Determine heavy transformer layer indices
    heavy_trans_keys = [k for k in sd.keys() if k.startswith('decoder.decoder.layers.') and 'weight' in k]
    if heavy_trans_keys:
        layer_indices = [int(k.split('decoder.decoder.layers.')[1].split('.')[0]) for k in heavy_trans_keys]
        last_trans_layer_idx = max(layer_indices)
    else:
        last_trans_layer_idx = 2

    print(f"\nFound {len(lw_keys)} total keys in lightweight decoder.")
    
    shared_identical = []
    shared_diverged = []
    
    trans_identical = []
    trans_diverged = []

    missing_keys = []

    for lw_key, lw_tensor in lw_keys.items():
        is_transformer = 'lightweight_decoder.decoder.layers' in lw_key
        
        # Route the mapping logic based on what component we are checking
        if is_transformer:
            heavy_key = lw_key.replace('lightweight_decoder.decoder.layers.0', f'decoder.decoder.layers.{last_trans_layer_idx}')
        else:
            heavy_key = lw_key.replace('lightweight_decoder.', 'decoder.')
            if 'dec_score_head.0' in heavy_key:
                heavy_key = heavy_key.replace('dec_score_head.0', f'dec_score_head.{last_head_idx}')
            elif 'dec_bbox_head.0' in heavy_key:
                heavy_key = heavy_key.replace('dec_bbox_head.0', f'dec_bbox_head.{last_head_idx}')

        if heavy_key not in sd:
            missing_keys.append(heavy_key)
            continue
            
        heavy_tensor = sd[heavy_key]
        
        # Check if the tensors are mathematically identical
        is_equal = torch.equal(lw_tensor, heavy_tensor)
        diff = 0.0 if is_equal else torch.max(torch.abs(lw_tensor - heavy_tensor)).item()
        
        if is_transformer:
            if is_equal:
                trans_identical.append(lw_key)
            else:
                trans_diverged.append((lw_key, diff))
        else:
            if is_equal:
                shared_identical.append(lw_key)
            else:
                shared_diverged.append((lw_key, diff))

    # --- PRINT RESULTS ---
    print("\n" + "="*70)
    print("1. SHARED COMPONENTS (Heads & Input Proj) - EXPECT PERFECT MATCH")
    print("="*70)
    print(f"🟢 Perfect Matches: {len(shared_identical)}")
    print(f"🔴 Diverged Tensors: {len(shared_diverged)}")
    if shared_diverged:
        for k, d in shared_diverged:
            print(f"  - {k}: diff = {d:.6f}")

    print("\n" + "="*70)
    print("2. DECOUPLED COMPONENTS (Transformer Layers) - EXPECT DIVERGED")
    print("="*70)
    print(f"🔴 Untrained/Frozen Tensors: {len(trans_identical)}")
    print(f"🟢 Trained/Updated Tensors: {len(trans_diverged)}")
    
    if trans_diverged:
        print("\nAll Weights that successfully learned/updated:")
        # Removed the [:5] slice so it prints everything
        for k, d in trans_diverged:
            print(f"  - {k}: max diff = {d:.6f}")

    if trans_identical:
        print("\nTransformer weights that remained identical (Untrained/Frozen/Buffers):")
        for k in trans_identical:
            print(f"  - {k}")

    print("\n" + "="*70)
    if len(shared_diverged) == 0 and len(trans_identical) <= 1: # Adjusted to allow 1 buffer
        print("🌟 FLAWLESS ARCHITECTURE STATE CONFIRMED!")
        print("The prediction heads are perfectly frozen, AND the transformer successfully learned!")
    elif len(shared_diverged) == 0:
        print("✅ Heads are safe, but check if the transformer updated.")
    else:
        print("❌ WARNING: The shared heads got mutated!")
    print("="*70 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, required=True, help="Path to checkpoint")
    args = parser.parse_args()
    
    verify_internal_sharing(args.weights)