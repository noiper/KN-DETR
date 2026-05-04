import os
import argparse
import tensorrt as trt

def main(onnx_path, engine_path, model_type, max_batchsize, opt_batchsize, min_batchsize, use_fp16=True, verbose=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    print(f"[INFO] Loading ONNX file from {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file")

    config = builder.create_builder_config()
    config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True)
    config.max_workspace_size = 1 << 30  # 1GB
    
    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] FP16 optimization enabled.")

    profile = builder.create_optimization_profile()
    profile.set_shape("images", min=(min_batchsize, 3, 640, 640), opt=(opt_batchsize, 3, 640, 640), max=(max_batchsize, 3, 640, 640))
    profile.set_shape("orig_target_sizes", min=(min_batchsize, 2), opt=(opt_batchsize, 2), max=(max_batchsize, 2))
    
    # --- ADD THE CACHE PROFILE FOR NON-KEY MODEL ---
    if model_type == "nonkey":
        # FIXME 3: Update this tuple to match the exact shape of your temporal cache
        cache_shape = (min_batchsize, 256, 20, 20)
        profile.set_shape("cache", min=cache_shape, opt=cache_shape, max=cache_shape)

    config.add_optimization_profile(profile)

    print("[INFO] Building TensorRT engine...")
    engine = builder.build_engine(network, config)

    if engine is None:
        raise RuntimeError("Failed to build the engine. Check unsupported nodes or Deformable Attention plugins.")

    print(f"[INFO] Saving engine to {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    print("[INFO] Engine export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT Engine")
    parser.add_argument("--onnx", "-i", type=str, required=True, help="Path to input ONNX model file")
    parser.add_argument("--saveEngine", "-o", type=str, required=True, help="Path to output TensorRT engine file")
    parser.add_argument("--model_type", "-m", type=str, choices=['key', 'nonkey'], required=True, help="Which engine is being built?")
    parser.add_argument("--maxBatchSize", type=int, default=1)
    parser.add_argument("--optBatchSize", type=int, default=1)
    parser.add_argument("--minBatchSize", type=int, default=1)
    parser.add_argument("--fp16", default=True, action="store_true")
    
    args = parser.parse_args()
    main(args.onnx, args.saveEngine, args.model_type, args.maxBatchSize, args.optBatchSize, args.minBatchSize, args.fp16)