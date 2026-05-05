import argparse
import tensorrt as trt


def _set_workspace_size(config, workspace_mb: int):
    workspace_bytes = int(workspace_mb) << 20
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes


def _build_shape_for_batch(name, shape, batch_size, image_h, image_w):
    resolved = []
    for axis, dim in enumerate(shape):
        d = int(dim)
        if axis == 0:
            resolved.append(batch_size if d == -1 else d)
            continue
        if d != -1:
            resolved.append(d)
            continue
        if name == 'images' and axis == 2:
            resolved.append(image_h)
        elif name == 'images' and axis == 3:
            resolved.append(image_w)
        else:
            raise ValueError(
                f"Unsupported dynamic dimension in input '{name}' at axis {axis}. "
                "Export ONNX with fixed non-batch dims, or extend this script."
            )
    return tuple(resolved)


def _validate_temporal_inputs(network, model_type):
    input_names = {network.get_input(i).name for i in range(network.num_inputs)}
    if model_type == 'nonkey':
        required = {'images', 'orig_target_sizes', 'cache_ccff_0', 'cache_ccff_1', 'cache_ccff_2', 'cache_content', 'cache_points'}
        missing = sorted(required - input_names)
        if missing:
            raise RuntimeError(
                f"Non-key ONNX is missing required inputs: {missing}. "
                "Expected inputs: images, orig_target_sizes, and all cache_* tensors."
            )
    elif model_type == 'key':
        required = {'images', 'orig_target_sizes'}
        missing = sorted(required - input_names)
        if missing:
            raise RuntimeError(f"Key ONNX is missing required inputs: {missing}")


def main(
    onnx_path,
    engine_path,
    model_type,
    max_batchsize,
    opt_batchsize,
    min_batchsize,
    use_fp16=True,
    verbose=False,
    workspace_mb=1024,
    image_h=640,
    image_w=640,
):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')

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
    _validate_temporal_inputs(network, model_type)

    config = builder.create_builder_config()
    if hasattr(config, 'set_preview_feature') and hasattr(trt, 'PreviewFeature') and \
            hasattr(trt.PreviewFeature, 'FASTER_DYNAMIC_SHAPES_0805'):
        config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True)
    _set_workspace_size(config, workspace_mb)

    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] FP16 optimization enabled.")
    elif use_fp16:
        print("[INFO] FP16 requested, but platform does not support fast FP16. Falling back to FP32.")

    profile = builder.create_optimization_profile()
    print("[INFO] Applying optimization profile:")
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        shape = tuple(tensor.shape)
        min_shape = _build_shape_for_batch(tensor.name, shape, min_batchsize, image_h, image_w)
        opt_shape = _build_shape_for_batch(tensor.name, shape, opt_batchsize, image_h, image_w)
        max_shape = _build_shape_for_batch(tensor.name, shape, max_batchsize, image_h, image_w)
        profile.set_shape(tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
        print(f"  - {tensor.name}: min={min_shape}, opt={opt_shape}, max={max_shape}")

    config.add_optimization_profile(profile)

    print("[INFO] Building TensorRT engine...")
    if hasattr(builder, 'build_serialized_network'):
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build serialized TensorRT engine. Check unsupported nodes/plugins.")
        with open(engine_path, "wb") as f:
            f.write(bytes(serialized_engine))
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine. Check unsupported nodes/plugins.")
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
    parser.add_argument("--workspaceMB", type=int, default=1024, help="TensorRT workspace size in MB")
    parser.add_argument("--inputH", type=int, default=640, help="Fallback input image height for dynamic ONNX dims")
    parser.add_argument("--inputW", type=int, default=640, help="Fallback input image width for dynamic ONNX dims")
    parser.add_argument("--fp16", dest="fp16", action="store_true", help="Enable FP16 (default)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16")
    parser.add_argument("--verbose", action="store_true", help="Enable TensorRT verbose logs")
    parser.set_defaults(fp16=True)

    args = parser.parse_args()
    main(
        args.onnx,
        args.saveEngine,
        args.model_type,
        args.maxBatchSize,
        args.optBatchSize,
        args.minBatchSize,
        args.fp16,
        args.verbose,
        args.workspaceMB,
        args.inputH,
        args.inputW,
    )
