"""实验 5: 中间层 Transpose 开销测试。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import onnxruntime as ort
from common.benchmark import (
    benchmark_model, print_stats, save_results,
    generate_dummy_input, compare_models
)

MODEL_ORIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8m.onnx')
MODEL_MOD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modified_models', 'yolov8m_mid_transpose.onnx')
WARMUP, ITERS = 10, 100

def run():
    print("=" * 60)
    print("实验 5: 中间层 Transpose 开销")
    print("=" * 60)
    base = benchmark_model(MODEL_ORIG, generate_dummy_input,
                           graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                           warmup=WARMUP, iters=ITERS)
    print_stats('Baseline', base)
    mod = benchmark_model(MODEL_MOD, generate_dummy_input,
                          graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                          warmup=WARMUP, iters=ITERS)
    print_stats('With Mid Transpose Pair', mod)
    cmp = compare_models(base, mod, 'Mid Transpose Pair')
    single_est = cmp['delta_ms'] / 2.0
    all_results = {
        'experiment': 'exp05_mid_transpose',
        'baseline': base, 'modified': mod, 'comparison': cmp,
        'single_transpose_estimate_ms': single_est,
        'note': 'Round-trip pair after 20th Conv; single = delta/2',
    }
    save_results(all_results, 'exp05_mid_transpose.json')
    print("实验 5 完成")

if __name__ == '__main__':
    run()
