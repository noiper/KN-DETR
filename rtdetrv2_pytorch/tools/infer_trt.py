#!/usr/bin/env python3
"""Jetson-oriented batch-1 TensorRT temporal inference on frame sequences.

Example (MOT17 image directory):
python tools/trt_sequence_infer.py \
  --frames_dir ~/Desktop/Projects_2025/dataset/mot17/val/MOT17-02-FRCNN/img1 \
  --key_engine key.engine \
  --nonkey_engine nonkey.engine \
  --mode knk \
  --nk_per_key 3 \
  --num_frames 300 \
  --power
"""

import argparse
import csv
import json
import re
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorrt as trt
import torch
from PIL import Image


KEY_CACHE_NAMES = (
    "cache_ccff_0",
    "cache_ccff_1",
    "cache_ccff_2",
    "cache_content",
    "cache_points",
)


class TensorRTInference:
    def __init__(self, engine_path: str, device: str = "cuda:0", verbose: bool = False):
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [
            name for name in self.tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name for name in self.tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]

        self._signature: Optional[Tuple[Tuple[str, Tuple[int, ...], torch.dtype], ...]] = None
        self._buffers: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._binding_addrs: OrderedDict[str, int] = OrderedDict()

    def _load_engine(self, path: str):
        with open(path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {path}")
        return engine

    def validate_bindings(self, required_inputs: set, required_outputs: set, tag: str):
        missing_inputs = sorted(required_inputs - set(self.input_names))
        missing_outputs = sorted(required_outputs - set(self.output_names))
        if missing_inputs or missing_outputs:
            raise RuntimeError(
                f"{tag} engine binding mismatch. "
                f"Missing inputs={missing_inputs}, missing outputs={missing_outputs}. "
                f"Found inputs={self.input_names}, outputs={self.output_names}"
            )

    def _ensure_buffers(self, blob: Dict[str, torch.Tensor]):
        missing = [name for name in self.input_names if name not in blob]
        if missing:
            raise RuntimeError(f"Missing input tensors for inference: {missing}")

        signature: List[Tuple[str, Tuple[int, ...], torch.dtype]] = []
        for name in self.input_names:
            tensor = blob[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Input '{name}' must be torch.Tensor, got {type(tensor)}")
            if tensor.device != self.device:
                raise RuntimeError(
                    f"Input '{name}' is on {tensor.device}, expected {self.device}. "
                    "Move all tensors to the inference device before calling infer()."
                )
            signature.append((name, tuple(tensor.shape), tensor.dtype))

        sig_tuple = tuple(signature)
        if sig_tuple == self._signature:
            return

        self._buffers.clear()
        self._binding_addrs.clear()

        for name in self.input_names:
            tensor = blob[name]
            self.context.set_input_shape(name, tuple(tensor.shape))
            self._buffers[name] = torch.empty_like(tensor, device=self.device)
            self._binding_addrs[name] = self._buffers[name].data_ptr()

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f"Output '{name}' has unresolved dynamic shape {shape}. "
                    "Please provide compatible static/dynamic profile inputs."
                )
            np_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = torch.from_numpy(np.empty([], dtype=np_dtype)).dtype
            self._buffers[name] = torch.empty(shape, dtype=torch_dtype, device=self.device)
            self._binding_addrs[name] = self._buffers[name].data_ptr()

        self._signature = sig_tuple

    def infer(self, blob: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self._ensure_buffers(blob)

        for name in self.input_names:
            self._buffers[name].copy_(blob[name])

        bindings = [int(self._binding_addrs[name]) for name in self.tensor_names]
        ok = self.context.execute_v2(bindings)
        if not ok:
            raise RuntimeError("TensorRT execution failed")

        return {name: self._buffers[name] for name in self.output_names}


class TegrastatsMonitor:
    def __init__(self, interval_ms: int = 200):
        self.interval_ms = int(interval_ms)
        self.samples_w: List[float] = []
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.available = False

    @staticmethod
    def _extract_power_w(line: str) -> Optional[float]:
        vdd = re.search(r"VDD_IN\s+(\d+)(mW|W)?", line)
        if vdd:
            value = float(vdd.group(1))
            unit = vdd.group(2) or "mW"
            return value if unit == "W" else value / 1000.0

        pom = re.search(r"POM_5V_IN\s+(\d+)(mW|W)?", line)
        if pom:
            value = float(pom.group(1))
            unit = pom.group(2) or "mW"
            return value if unit == "W" else value / 1000.0
        return None

    def _reader(self):
        assert self._proc is not None
        for line in self._proc.stdout:
            if self._stop_event.is_set():
                break
            power_w = self._extract_power_w(line)
            if power_w is not None:
                self.samples_w.append(power_w)

    def start(self):
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except (FileNotFoundError, PermissionError):
            self.available = False
            return

        self.available = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2.0)

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "avg_ms": float(arr.mean()),
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
    }


def _list_frames(frames_dir: Path, num_frames: int) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    if not files:
        raise RuntimeError(f"No image files found in {frames_dir}")
    return files[:num_frames]


def _preprocess_frame(
    frame_path: Path,
    input_h: int,
    input_w: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with Image.open(frame_path) as img:
        rgb = img.convert("RGB")
        orig_w, orig_h = rgb.size
        resized = rgb.resize((input_w, input_h))
        arr = np.asarray(resized, dtype=np.float32) / 255.0

    chw = np.ascontiguousarray(arr.transpose(2, 0, 1))
    # images: [B, C, H, W] -> [1, 3, input_h, input_w]
    images = torch.from_numpy(chw).unsqueeze(0).to(device=device, non_blocking=True)
    # orig_target_sizes: [B, 2] in [width, height]
    orig_target_sizes = torch.tensor([[orig_w, orig_h]], dtype=torch.int64, device=device)
    return images, orig_target_sizes


def _count_dets(scores: torch.Tensor, score_thr: float) -> int:
    # Keep postprocessing on CPU so we do not depend on PyTorch CUDA kernels.
    scores_np = scores.detach().cpu().numpy()
    return int(np.count_nonzero(scores_np > score_thr))


def _timed_infer(engine: TensorRTInference, blob: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, torch.Tensor]]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = engine.infer(blob)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, outputs


def _write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-1 TensorRT temporal inference on frame sequence")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory of ordered frame images")
    parser.add_argument("--key_engine", type=str, required=True, help="Path to key.engine")
    parser.add_argument("--nonkey_engine", type=str, default=None, help="Path to nonkey.engine")
    parser.add_argument(
        "--mode",
        type=str,
        default="knk",
        choices=["all_key", "knk", "baseline"],
        help="all_key: every frame key; knk: key + non-key; baseline: key + prediction reuse",
    )
    parser.add_argument("--nk_per_key", type=int, default=1, help="Non-key frames after each key frame")
    parser.add_argument("--num_frames", type=int, default=300, help="Number of sorted frames to process")
    parser.add_argument("--warmup", type=int, default=10, help="Exclude first N frames from metrics")
    parser.add_argument("--input_h", type=int, default=640)
    parser.add_argument("--input_w", type=int, default=640)
    parser.add_argument("--score_thr", type=float, default=0.5, help="Threshold for detection count reporting")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--power", action="store_true", help="Measure power using tegrastats")
    parser.add_argument("--tegrastats_interval_ms", type=int, default=200)
    parser.add_argument("--print_every", type=int, default=50, help="Progress print interval in frames")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional per-frame metrics CSV path")
    parser.add_argument("--save_json", type=str, default=None, help="Optional summary JSON path")
    parser.add_argument("--verbose_trt", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for TensorRT inference.")
    if args.mode == "knk" and not args.nonkey_engine:
        raise SystemExit("--nonkey_engine is required for --mode knk")
    if args.nk_per_key < 1 and args.mode in {"knk", "baseline"}:
        raise SystemExit("--nk_per_key must be >= 1 for knk/baseline modes")
    if args.num_frames <= 0:
        raise SystemExit("--num_frames must be > 0")

    frames_dir = Path(args.frames_dir).expanduser().resolve()
    if not frames_dir.exists():
        raise SystemExit(f"frames_dir does not exist: {frames_dir}")
    frame_paths = _list_frames(frames_dir, args.num_frames)

    print(f"[INFO] Mode={args.mode}, frames={len(frame_paths)}, warmup={args.warmup}, nk_per_key={args.nk_per_key}")
    print(f"[INFO] Frames dir: {frames_dir}")

    key_engine = TensorRTInference(args.key_engine, device=args.device, verbose=args.verbose_trt)
    key_engine.validate_bindings(
        required_inputs={"images", "orig_target_sizes"},
        required_outputs={"labels", "boxes", "scores", *KEY_CACHE_NAMES},
        tag="Key",
    )

    nonkey_engine = None
    if args.mode == "knk":
        nonkey_engine = TensorRTInference(args.nonkey_engine, device=args.device, verbose=args.verbose_trt)
        nonkey_engine.validate_bindings(
            required_inputs={"images", "orig_target_sizes", *KEY_CACHE_NAMES},
            required_outputs={"labels", "boxes", "scores"},
            tag="Non-key",
        )

    power_monitor = TegrastatsMonitor(args.tegrastats_interval_ms) if args.power else None
    if power_monitor is not None:
        power_monitor.start()
        if not power_monitor.available:
            print("[WARN] tegrastats unavailable; running latency-only metrics.")

    frame_latency_ms: List[float] = []
    key_latency_ms: List[float] = []
    nonkey_latency_ms: List[float] = []
    rows: List[Dict] = []

    latest_cache: Optional[Dict[str, torch.Tensor]] = None
    latest_key_preds: Optional[Dict[str, torch.Tensor]] = None
    cycle_len = args.nk_per_key + 1

    try:
        for i, frame_path in enumerate(frame_paths):
            images, orig_target_sizes = _preprocess_frame(
                frame_path, args.input_h, args.input_w, key_engine.device
            )
            measured = i >= args.warmup
            step = i % cycle_len

            role = "key"
            infer_ms = 0.0
            det_count = 0

            run_key = args.mode == "all_key" or step == 0
            if run_key:
                key_blob = {
                    "images": images,
                    "orig_target_sizes": orig_target_sizes,
                }
                infer_ms, key_out = _timed_infer(key_engine, key_blob)
                latest_cache = {name: key_out[name].clone() for name in KEY_CACHE_NAMES}
                latest_key_preds = {name: key_out[name].clone() for name in ("labels", "boxes", "scores")}
                det_count = _count_dets(key_out["scores"][0], args.score_thr)
                role = "key"
                if measured:
                    key_latency_ms.append(infer_ms)
                    frame_latency_ms.append(infer_ms)
            elif args.mode == "knk":
                if nonkey_engine is None:
                    raise RuntimeError("Non-key engine is not initialized.")
                if latest_cache is None:
                    raise RuntimeError("No cached key tensors available for non-key inference.")
                nonkey_blob = {
                    "images": images,
                    "orig_target_sizes": orig_target_sizes,
                    "cache_ccff_0": latest_cache["cache_ccff_0"],
                    "cache_ccff_1": latest_cache["cache_ccff_1"],
                    "cache_ccff_2": latest_cache["cache_ccff_2"],
                    "cache_content": latest_cache["cache_content"],
                    "cache_points": latest_cache["cache_points"],
                }
                infer_ms, nonkey_out = _timed_infer(nonkey_engine, nonkey_blob)
                det_count = _count_dets(nonkey_out["scores"][0], args.score_thr)
                role = "nonkey"
                if measured:
                    nonkey_latency_ms.append(infer_ms)
                    frame_latency_ms.append(infer_ms)
            else:
                if latest_key_preds is None:
                    raise RuntimeError("No key predictions available for baseline reuse mode.")
                role = "reuse"
                infer_ms = 0.0
                det_count = _count_dets(latest_key_preds["scores"][0], args.score_thr)
                if measured:
                    frame_latency_ms.append(infer_ms)

            rows.append(
                {
                    "frame_idx": i,
                    "frame_name": frame_path.name,
                    "role": role,
                    "inference_ms": round(infer_ms, 6),
                    "detections_over_thr": det_count,
                    "is_warmup": int(not measured),
                }
            )
            if args.print_every > 0 and ((i + 1) % args.print_every == 0 or (i + 1) == len(frame_paths)):
                print(f"[INFO] Processed {i + 1}/{len(frame_paths)} frames")
    finally:
        if power_monitor is not None:
            power_monitor.stop()

    measured_frames = max(0, len(frame_paths) - args.warmup)
    total_infer_s = float(sum(frame_latency_ms) / 1000.0)
    fps = (measured_frames / total_infer_s) if total_infer_s > 0 else 0.0

    frame_stats = _stats(frame_latency_ms)
    key_stats = _stats(key_latency_ms)
    nonkey_stats = _stats(nonkey_latency_ms)

    avg_power_w = 0.0
    energy_per_frame_j = 0.0
    power_samples = 0
    if power_monitor is not None and power_monitor.available and power_monitor.samples_w:
        avg_power_w = float(np.mean(np.asarray(power_monitor.samples_w, dtype=np.float64)))
        power_samples = len(power_monitor.samples_w)
        energy_per_frame_j = (avg_power_w * total_infer_s / measured_frames) if measured_frames > 0 else 0.0

    summary = {
        "mode": args.mode,
        "frames_total": len(frame_paths),
        "frames_measured": measured_frames,
        "warmup": args.warmup,
        "nk_per_key": args.nk_per_key,
        "input_h": args.input_h,
        "input_w": args.input_w,
        "frame_latency_ms": frame_stats,
        "key_latency_ms": key_stats,
        "nonkey_latency_ms": nonkey_stats,
        "fps_inference_only": fps,
        "power_avg_w": avg_power_w,
        "power_samples": power_samples,
        "energy_per_frame_j": energy_per_frame_j,
    }

    print("\n================ SUMMARY ================")
    print(f"Mode: {args.mode}")
    print(f"Frames: total={len(frame_paths)}, measured={measured_frames}, warmup={args.warmup}")
    print(
        f"Frame latency (ms): avg={frame_stats['avg_ms']:.3f}, "
        f"p50={frame_stats['p50_ms']:.3f}, p95={frame_stats['p95_ms']:.3f}"
    )
    print(
        f"Key latency   (ms): avg={key_stats['avg_ms']:.3f}, "
        f"p50={key_stats['p50_ms']:.3f}, p95={key_stats['p95_ms']:.3f}"
    )
    if args.mode == "knk":
        print(
            f"Non-key lat.  (ms): avg={nonkey_stats['avg_ms']:.3f}, "
            f"p50={nonkey_stats['p50_ms']:.3f}, p95={nonkey_stats['p95_ms']:.3f}"
        )
    print(f"Inference-only FPS: {fps:.3f}")
    if args.power:
        if power_samples > 0:
            print(
                f"Power avg (W): {avg_power_w:.3f} (samples={power_samples}), "
                f"Energy/frame (J): {energy_per_frame_j:.5f}"
            )
        else:
            print("Power avg (W): unavailable (tegrastats missing or no samples parsed)")
    print("=========================================\n")

    if args.save_csv:
        csv_path = Path(args.save_csv).expanduser().resolve()
        _write_csv(csv_path, rows)
        print(f"[INFO] Wrote per-frame CSV: {csv_path}")
    if args.save_json:
        json_path = Path(args.save_json).expanduser().resolve()
        _write_json(json_path, summary)
        print(f"[INFO] Wrote summary JSON: {json_path}")


if __name__ == "__main__":
    main()
