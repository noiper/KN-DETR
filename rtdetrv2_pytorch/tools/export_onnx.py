"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""
import argparse
import inspect
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn

from src.core import YAMLConfig
from src.zoo.temporal_rtdetr import TemporalRTDETR


def _extract_model_state(ckpt):
    if isinstance(ckpt, dict):
        if 'ema' in ckpt and isinstance(ckpt['ema'], dict) and 'module' in ckpt['ema']:
            return ckpt['ema']['module']
        if 'model_state_dict' in ckpt:
            return ckpt['model_state_dict']
        if 'model' in ckpt:
            return ckpt['model']
    return ckpt


def _strip_module_prefix(state):
    if not isinstance(state, dict):
        return state
    if any(k.startswith('module.') for k in state.keys()):
        return {k[len('module.'):]: v for k, v in state.items()}
    return state


def _build_temporal_model(cfg):
    base_model = cfg.model
    encoder = getattr(base_model, 'encoder', None)
    decoder = getattr(base_model, 'decoder', None)
    if encoder is None or decoder is None:
        raise ValueError("Base model must have encoder and decoder to build TemporalRTDETR.")

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

    temporal_model = TemporalRTDETR(
        backbone=base_model.backbone,
        encoder=encoder,
        decoder=decoder,
        num_classes=cfg.yaml_cfg.get('num_classes', 80),
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        use_lightweight_decoder=cfg.yaml_cfg.get('use_lightweight_decoder', False),
        reuse_position=cfg.yaml_cfg.get('reuse_position', 0),
    )
    return temporal_model


def main(args):
    cfg = YAMLConfig(args.config)

    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    state = _strip_module_prefix(_extract_model_state(checkpoint))

    is_temporal_ckpt = isinstance(state, dict) and any(
        k.startswith('fusion_blocks.') or k.startswith('lightweight_decoder.')
        for k in state.keys()
    )
    if not is_temporal_ckpt:
        raise RuntimeError(
            "Checkpoint does not look like a full temporal checkpoint (missing fusion/lightweight decoder keys). "
            "Please pass a phase1 temporal checkpoint, e.g. output/phase1_*/best_model.pth."
        )

    model = _build_temporal_model(cfg)

    # If the checkpoint has decoupled non-key heads, decouple model before loading.
    if isinstance(state, dict) and any('lightweight_decoder.dec_score_head' in k for k in state.keys()):
        print("   [Auto-Detect] Decoupled non-key heads found. Decoupling model...")
        if hasattr(model, 'decouple_non_key_prediction_heads'):
            model.decouple_non_key_prediction_heads()
        else:
            print("   [Warning] Model does not support decoupling, skipping...")

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load checkpoint into TemporalRTDETR. "
            f"This usually means config/checkpoint mismatch.\nConfig: {args.config}\nCheckpoint: {args.resume}\n{e}"
        ) from e
    print(f'Successfully loaded weights from {args.resume}')

    model = model.deploy().eval()
    postprocessor = cfg.postprocessor.deploy().eval()

    class KeyModel(nn.Module):
        def __init__(self, temporal_model, deployed_postprocessor):
            super().__init__()
            self.model = temporal_model
            self.postprocessor = deployed_postprocessor

        def forward(self, images, orig_target_sizes):
            out_k = self.model.forward_key_frame(images, None)
            labels, boxes, scores = self.postprocessor(out_k, orig_target_sizes)

            if self.model.cached_ccff is None:
                raise RuntimeError("Key path did not populate cached CCFF features.")
            if self.model.cached_content is None or self.model.cached_points_unact is None:
                raise RuntimeError("Key path did not populate cached decoder states.")

            ccff_0, ccff_1, ccff_2 = self.model.cached_ccff
            return (
                labels,
                boxes,
                scores,
                ccff_0,
                ccff_1,
                ccff_2,
                self.model.cached_content,
                self.model.cached_points_unact,
            )

    class NonKeyModel(nn.Module):
        def __init__(self, temporal_model, deployed_postprocessor):
            super().__init__()
            self.model = temporal_model
            self.postprocessor = deployed_postprocessor

        def forward(self, images, orig_target_sizes, cache_ccff_0, cache_ccff_1, cache_ccff_2, content, points):
            out_nk = self.model.forward_non_key_frame(
                images,
                cached_ccff=[cache_ccff_0, cache_ccff_1, cache_ccff_2],
                cached_content=content,
                cached_points_unact=points,
            )
            labels, boxes, scores = self.postprocessor(out_nk, orig_target_sizes)
            return labels, boxes, scores

    input_h = args.input_h if args.input_h is not None else args.input_size
    input_w = args.input_w if args.input_w is not None else args.input_size
    data = torch.rand(1, 3, input_h, input_w, dtype=torch.float32)
    # Postprocessor expects [width, height].
    size = torch.tensor([[input_w, input_h]], dtype=torch.int64)

    hidden_dim = int(getattr(model, 'hidden_dim', 256))
    num_queries = int(getattr(model, 'num_queries', 300))
    mock_ccff_0 = torch.rand(1, hidden_dim, input_h // 8, input_w // 8, dtype=torch.float32)
    mock_ccff_1 = torch.rand(1, hidden_dim, input_h // 16, input_w // 16, dtype=torch.float32)
    mock_ccff_2 = torch.rand(1, hidden_dim, input_h // 32, input_w // 32, dtype=torch.float32)
    mock_content = torch.rand(1, num_queries, hidden_dim, dtype=torch.float32)
    mock_points = torch.rand(1, num_queries, 4, dtype=torch.float32)

    key_model = KeyModel(model, postprocessor).eval()
    nonkey_model = NonKeyModel(model, postprocessor).eval()

    key_dynamic_axes = {
        'images': {0: 'N'},
        'orig_target_sizes': {0: 'N'},
        'labels': {0: 'N'},
        'boxes': {0: 'N'},
        'scores': {0: 'N'},
        'cache_ccff_0': {0: 'N'},
        'cache_ccff_1': {0: 'N'},
        'cache_ccff_2': {0: 'N'},
        'cache_content': {0: 'N'},
        'cache_points': {0: 'N'},
    }
    nonkey_dynamic_axes = {
        'images': {0: 'N'},
        'orig_target_sizes': {0: 'N'},
        'cache_ccff_0': {0: 'N'},
        'cache_ccff_1': {0: 'N'},
        'cache_ccff_2': {0: 'N'},
        'cache_content': {0: 'N'},
        'cache_points': {0: 'N'},
        'labels': {0: 'N'},
        'boxes': {0: 'N'},
        'scores': {0: 'N'},
    }

    export_fn_params = inspect.signature(torch.onnx.export).parameters
    export_supports_dynamo = 'dynamo' in export_fn_params

    def _export_onnx(model_obj, model_inputs, save_path, input_names, output_names, dynamic_axes):
        export_kwargs = dict(
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            verbose=False,
            do_constant_folding=True,
        )
        # Default to legacy exporter to avoid hard dependency on onnxscript.
        if export_supports_dynamo:
            export_kwargs['dynamo'] = args.dynamo
        torch.onnx.export(model_obj, model_inputs, save_path, **export_kwargs)

    with torch.no_grad():
        print(f"\nExporting Key Model -> {args.key_onnx}")
        _export_onnx(
            key_model,
            (data, size),
            args.key_onnx,
            input_names=['images', 'orig_target_sizes'],
            output_names=[
                'labels',
                'boxes',
                'scores',
                'cache_ccff_0',
                'cache_ccff_1',
                'cache_ccff_2',
                'cache_content',
                'cache_points',
            ],
            dynamic_axes=key_dynamic_axes,
        )

        print(f"\nExporting Non-Key Model -> {args.nonkey_onnx}")
        _export_onnx(
            nonkey_model,
            (data, size, mock_ccff_0, mock_ccff_1, mock_ccff_2, mock_content, mock_points),
            args.nonkey_onnx,
            input_names=[
                'images',
                'orig_target_sizes',
                'cache_ccff_0',
                'cache_ccff_1',
                'cache_ccff_2',
                'cache_content',
                'cache_points',
            ],
            output_names=['labels', 'boxes', 'scores'],
            dynamic_axes=nonkey_dynamic_axes,
        )

    print(f"\nONNX export complete. Created: {args.key_onnx}, {args.nonkey_onnx}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--resume', '-r', type=str, required=True)
    parser.add_argument('--input_size', '-s', type=int, default=640)
    parser.add_argument('--input_h', type=int, default=None)
    parser.add_argument('--input_w', type=int, default=None)
    parser.add_argument('--opset', type=int, default=16)
    parser.add_argument('--dynamo', action='store_true',
                        help='Use the new torch.export-based ONNX path (requires onnxscript in newer PyTorch).')
    parser.add_argument('--key_onnx', type=str, default='key_model.onnx')
    parser.add_argument('--nonkey_onnx', type=str, default='nonkey_model.onnx')
    args = parser.parse_args()
    main(args)
