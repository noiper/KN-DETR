"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn 
from src.core import YAMLConfig

def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        # Fixed: Added weights_only=False to allow loading older or numpy-based checkpoints
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) 
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
        
        # --- AUTO-DECOUPLE DETECTION ---
        # If the checkpoint contains decoupled non-key heads, we MUST decouple the model
        # before loading to avoid overwriting the heavy decoder's heads with the student's.
        is_decoupled = any('lightweight_decoder.dec_score_head' in k for k in state.keys())
        if is_decoupled:
            print("   [Auto-Detect] Decoupled prediction heads found in checkpoint. Decoupling model...")
            if hasattr(cfg.model, 'decouple_non_key_prediction_heads'):
                cfg.model.decouple_non_key_prediction_heads()
            else:
                print("   [Warning] Model does not support decoupling, skipping...")

        cfg.model.load_state_dict(state)
        print(f'Successfully loaded weights from {args.resume}')
    else:
        print('Not loading model.state_dict, using default init state dict...')

    # ==========================================
    # 1. KEY MODEL WRAPPER
    # ==========================================
    class KeyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            out_k = self.model.forward_key_frame(images, None)
            labels, boxes, scores = self.postprocessor(out_k, orig_target_sizes)
            
            # Explicitly extract the cache for Non-Key path
            ccff_s3, ccff_s4, ccff_s5 = self.model.cached_ccff
            content = self.model.cached_content
            points = self.model.cached_points_unact
            
            return labels, boxes, scores, ccff_s3, ccff_s4, ccff_s5, content, points

    # ==========================================
    # 2. NON-KEY MODEL WRAPPER
    # ==========================================
    class NonKeyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes, ccff_s3, ccff_s4, ccff_s5, content, points):
            # Pass the explicit cache tensors into the non-key forward
            out_nk = self.model.forward_non_key_frame(
                images, 
                cached_ccff=[ccff_s3, ccff_s4, ccff_s5],
                cached_content=content,
                cached_points_unact=points
            )
            labels, boxes, scores = self.postprocessor(out_nk, orig_target_sizes)
            return labels, boxes, scores

    # --- MOCK INPUTS ---
    input_size = args.input_size
    data = torch.rand(1, 3, input_size, input_size)
    size = torch.tensor([[input_size, input_size]])
    
    # Mock cache shapes based on Hidden Dim=256 and Strides=[8, 16, 32]
    # For 640x640 input: 80x80, 40x40, 20x20
    h_dim = 256
    num_q = 300
    mock_ccff_s3 = torch.rand(1, h_dim, input_size // 8, input_size // 8)
    mock_ccff_s4 = torch.rand(1, h_dim, input_size // 16, input_size // 16)
    mock_ccff_s5 = torch.rand(1, h_dim, input_size // 32, input_size // 32)
    mock_content = torch.rand(1, num_q, h_dim)
    mock_points = torch.rand(1, num_q, 4)

    # --- EXPORT KEY MODEL ---
    print("\nExporting Key Model...")
    key_model = KeyModel()
    torch.onnx.export(
        key_model, 
        (data, size), 
        "key_model.onnx",
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores', 'cache_s3', 'cache_s4', 'cache_s5', 'cache_content', 'cache_points'],
        dynamic_axes={
            'images': {0: 'N'}, 
            'orig_target_sizes': {0: 'N'},
            'cache_s3': {0: 'N'}, 'cache_s4': {0: 'N'}, 'cache_s5': {0: 'N'},
            'cache_content': {0: 'N'}, 'cache_points': {0: 'N'}
        },
        opset_version=16, 
        verbose=False,
        do_constant_folding=True,
    )

    # --- EXPORT NON-KEY MODEL ---
    print("\nExporting Non-Key Model...")
    nonkey_model = NonKeyModel()
    torch.onnx.export(
        nonkey_model, 
        (data, size, mock_ccff_s3, mock_ccff_s4, mock_ccff_s5, mock_content, mock_points), 
        "nonkey_model.onnx",
        input_names=['images', 'orig_target_sizes', 'cache_s3', 'cache_s4', 'cache_s5', 'cache_content', 'cache_points'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes={
            'images': {0: 'N'}, 
            'orig_target_sizes': {0: 'N'},
            'cache_s3': {0: 'N'}, 'cache_s4': {0: 'N'}, 'cache_s5': {0: 'N'},
            'cache_content': {0: 'N'}, 'cache_points': {0: 'N'}
        },
        opset_version=16, 
        verbose=False,
        do_constant_folding=True,
    )

    print("\nONNX Export Complete. Created: key_model.onnx, nonkey_model.onnx")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--resume', '-r', type=str, required=True)
    parser.add_argument('--input_size', '-s', type=int, default=640)
    args = parser.parse_args()
    main(args)
