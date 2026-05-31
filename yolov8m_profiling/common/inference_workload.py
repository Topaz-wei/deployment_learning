#!/usr/bin/env python3
"""共享推理负载脚本 — 供 Nsight Systems / Nsight Compute 作为目标进程。

用法:
    python3 inference_workload.py \
      --engine weights/engines/yolov8m_fp16.engine \
      --data dummy \
      --warmup 10 \
      --iters 100
"""
import argparse
import os
import sys
import time
import numpy as np

# 必须: 系统 tensorrt 优先于项目 tensorrt/ 目录
sys.path.insert(0, '/usr/lib/python3.8/dist-packages')
import tensorrt as trt  # noqa: E402 — 缓存为系统 tensorrt

# 添加项目路径以导入 TrtEngine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tensorrt'))
from trt_engine import TrtEngine  # noqa: E402

# 添加当前路径以导入 data_source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.data_source import DummySource, CameraSource  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='YOLOv8m TRT engine inference workload')
    parser.add_argument('--engine', required=True, help='Path to .engine file')
    parser.add_argument('--data', default='dummy', choices=['dummy', 'camera'])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()

    # 选择数据源
    if args.data == 'dummy':
        source = DummySource()
    else:
        source = CameraSource()

    # 加载 engine
    print(f'[Workload] Loading engine: {args.engine}')
    engine = TrtEngine(args.engine)
    print(f'[Workload] Inputs: {engine.input_names}, Outputs: {engine.output_names}')

    inputs = source.get_input()
    metadata = source.get_metadata()
    print(f'[Workload] Data source: {metadata}')

    # 预热
    print(f'[Workload] Warming up ({args.warmup} iters)...')
    for _ in range(args.warmup):
        engine.infer(inputs)

    # 计时推理
    print(f'[Workload] Running inference ({args.iters} iters)...')
    times = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        engine.infer(inputs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed_ms)

    times = np.array(times)
    print(f'\n[Workload] === Inference Statistics ===')
    print(f'  Latency (ms): mean={times.mean():.3f}, min={times.min():.3f}, '
          f'max={times.max():.3f}, p50={np.median(times):.3f}, std={times.std():.3f}')
    print(f'  FPS: {1000.0 / times.mean():.2f}')
    print(f'  Iters: {args.iters}, Warmup: {args.warmup}')


if __name__ == '__main__':
    main()
