import torch
import argparse

def extract_state_dict(checkpoint):
    """Safely extract the state dict whether it's a custom save or RT-DETR save"""
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint

def compare_models(path1, path2):
    print(f"Loading Model 1 (e.g., Warm Start): {path1}")
    # Force CPU loading so it doesn't touch your training GPU
    ckpt1 = torch.load(path1, map_location='cpu', weights_only=False)
    sd1 = extract_state_dict(ckpt1)

    print(f"Loading Model 2 (e.g., After Part 1): {path2}")
    ckpt2 = torch.load(path2, map_location='cpu', weights_only=False)
    sd2 = extract_state_dict(ckpt2)

    # Filter only the keys belonging to the lightweight decoder
    lw_keys1 = {k: v for k, v in sd1.items()}
    lw_keys2 = {k: v for k, v in sd2.items()}

    if not lw_keys1 or not lw_keys2:
        print("\n❌ Error: Could not find 'lightweight_decoder' keys in one or both models.")
        return

    print(f"\nFound {len(lw_keys1)} keys in lightweight decoder.")
    
    changed_keys = []
    identical_keys = []
    missing_keys = []

    for key in lw_keys1.keys():
        if key not in lw_keys2:
            missing_keys.append(key)
            continue
            
        tensor1 = lw_keys1[key]
        tensor2 = lw_keys2[key]
        
        # Check if the tensors are mathematically identical
        if torch.equal(tensor1, tensor2):
            identical_keys.append(key)
        else:
            # Calculate the maximum absolute difference
            max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
            changed_keys.append((key, max_diff))

    # --- PRINT RESULTS ---
    print("\n" + "="*50)
    print("VERIFICATION RESULTS")
    print("="*50)
    
    print(f"🟢 Identical Tensors: {len(identical_keys)}")
    print(f"🔴 Changed Tensors:   {len(changed_keys)}")
    if missing_keys:
        print(f"⚠️ Missing Tensors:   {len(missing_keys)}")

    if len(changed_keys) > 0:
        print("\nWeights that CHANGED (with max absolute difference):")
        for key, diff in changed_keys:
            print(f"  - {key}: diff = {diff:.6f}")
            
    if len(identical_keys) == len(lw_keys1):
        print("\n✅ CONFIRMED: The lightweight decoder was completely frozen. 0% of the weights changed.")
    elif len(changed_keys) > 0 and len(identical_keys) > 0:
        print("\n⚠️ MIXED: Some weights changed, while others were frozen. Check the printed lists.")
    else:
        print("\n❌ ALL CHANGED: The entire lightweight decoder was updated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--m1', type=str, required=True, help="Path to first checkpoint (e.g., best.pth)")
    parser.add_argument('--m2', type=str, required=True, help="Path to second checkpoint (e.g., part1_best.pth)")
    args = parser.parse_args()
    
    compare_models(args.m1, args.m2)
