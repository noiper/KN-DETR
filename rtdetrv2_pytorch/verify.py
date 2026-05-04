"""
Verification script to compare Phase1 model with original RT-DETR
"""
import torch
import numpy as np
import os
import sys

# Ensure python path is correct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.zoo.rtdetr import RTDETRv2
from src.zoo.temporal_rtdetr import TemporalRTDETR
from src.core import YAMLConfig

def verify_parameters(original_model, phase1_model):
    """Verify backbone/encoder parameters match"""
    print("\n" + "="*80)
    print("Parameter Verification")
    print("="*80)

    # Check backbone
    orig_backbone_params = {name: param for name, param in original_model.backbone.named_parameters()}
    phase1_backbone_params = {name: param for name, param in phase1_model.backbone.named_parameters()}

    backbone_match = True
    for name in orig_backbone_params:
        if name in phase1_backbone_params:
            if not torch.allclose(orig_backbone_params[name], phase1_backbone_params[name], atol=1e-6):
                print(f"❌ Backbone param mismatch: {name}")
                backbone_match = False

    print(f"✅ Backbone parameters {'MATCH' if backbone_match else 'MISMATCH'}")

    # Check encoder
    orig_encoder_params = {name: param for name, param in original_model.encoder.named_parameters()}
    phase1_encoder_params = {name: param for name, param in phase1_model.encoder.named_parameters()}

    encoder_match = True
    for name in orig_encoder_params:
        if name in phase1_encoder_params:
            if not torch.allclose(orig_encoder_params[name], phase1_encoder_params[name], atol=1e-6):
                print(f"❌ Encoder param mismatch: {name}")
                encoder_match = False

    print(f"✅ Encoder parameters {'MATCH' if encoder_match else 'MISMATCH'}")

    # Check decoder
    orig_decoder_params = {name: param for name, param in original_model.decoder.named_parameters()}
    phase1_decoder_params = {name: param for name, param in phase1_model.decoder.named_parameters()}

    decoder_match = True
    for name in orig_decoder_params:
        if name in phase1_decoder_params:
            if not torch.allclose(orig_decoder_params[name], phase1_decoder_params[name], atol=1e-6):
                print(f"❌ Decoder param mismatch: {name}")
                decoder_match = False

    print(f"✅ Decoder parameters {'MATCH' if decoder_match else 'MISMATCH'}")

    return backbone_match and encoder_match and decoder_match


def verify_outputs(original_model, phase1_model, dummy_input):
    """Verify outputs match for same input"""
    print("\n" + "="*80)
    print("Output Verification")
    print("="*80)

    original_model.eval()
    phase1_model.eval()

    with torch.no_grad():
        # Original model
        orig_output = original_model(dummy_input)

        # Phase1 model key frame path
        phase1_output = phase1_model.forward_key_frame(dummy_input, None)

        # Compare outputs
        print(f"\nOriginal model output keys: {orig_output.keys()}")
        print(f"Phase1 model output keys: {phase1_output.keys()}")

        if 'pred_boxes' in orig_output and 'pred_boxes' in phase1_output:
            boxes_diff = torch.abs(orig_output['pred_boxes'] - phase1_output['pred_boxes']).max()
            print(f"\nMax boxes difference: {boxes_diff.item():.6f}")
            print(f"✅ Boxes {'MATCH' if boxes_diff < 1e-4 else 'MISMATCH'}")

        if 'pred_logits' in orig_output and 'pred_logits' in phase1_output:
            logits_diff = torch.abs(orig_output['pred_logits'] - phase1_output['pred_logits']).max()
            print(f"Max logits difference: {logits_diff.item():.6f}")
            print(f"✅ Logits {'MATCH' if logits_diff < 1e-4 else 'MISMATCH'}")

    return boxes_diff < 1e-4 and logits_diff < 1e-4

def build_temporal_model(cfg, device):
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
        reuse_position=cfg.yaml_cfg.get('reuse_position', 0),
    ).to(device)
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_path = 'configs/rtdetrv2/phase1_mot17_r18vd.yml'

    # Load original model
    print("Loading original RT-DETR model...")
    orig_cfg = YAMLConfig(config_path)
    original_model = RTDETRv2(orig_cfg).to(device)

    # Load Phase1 model
    print("Loading Phase1 model...")
    phase1_cfg = YAMLConfig(config_path)
    phase1_model = build_temporal_model(phase1_cfg, device)

    # Create dummy input
    dummy_input = torch.randn(2, 3, 640, 640).to(device)

    # Run verifications
    params_match = verify_parameters(original_model, phase1_model)
    outputs_match = verify_outputs(original_model, phase1_model, dummy_input)

    print("\n" + "="*80)
    print("Final Verdict")
    print("="*80)
    if params_match and outputs_match:
        print("✅ All checks PASSED - Model is correctly initialized!")
    else:
        print("❌ Some checks FAILED - Debug required!")


if __name__ == '__main__':
    main()