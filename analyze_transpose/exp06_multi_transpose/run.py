"""实验 6: 多层 Transpose 累积效应测试。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import onnxruntime as ort
from common.benchmark import (
    benchmark_model, print_stats, save_results,
    generate_dummy_input, compare_models
)

MODEL_ORIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8m.onnx')
MODEL_MOD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modified_models', 'yolov8m_multi_transpose.onnx')
WARMUP, ITERS = 10, 100

def run():
    print("=" * 60)
    print("实验 6: 多层 Transpose 累积效应")
    print("=" * 60)
    base = benchmark_model(MODEL_ORIG, generate_dummy_input,
                           graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                           warmup=WARMUP, iters=ITERS)
    print_stats('Baseline', base)
    mod = benchmark_model(MODEL_MOD, generate_dummy_input,
                          graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                          warmup=WARMUP, iters=ITERS)
    print_stats('With 4 Stage Transpose Pairs', mod)
    cmp = compare_models(base, mod, 'Multi Transpose Pairs')
    per_pair = cmp['delta_ms'] / 4.0
    per_single = cmp['delta_ms'] / 8.0
    all_results = {
        'experiment': 'exp06_multi_transpose',
        'baseline': base, 'modified': mod, 'comparison': cmp,
        'per_roundtrip_pair_ms': per_pair,
        'per_single_transpose_ms': per_single,
        'note': '4 round-trip pairs (8 Transposes) at backbone stage Conv outputs',
    }
    save_results(all_results, 'exp06_multi_transpose.json')
    print(f"  每对 (2T) 边际延迟: {per_pair:.2f} ms")
    print(f"  每个 T 边际延迟: {per_single:.2f} ms")
    print("实验 6 完成")

if __name__ == '__main__':
    run()
