"""实验 2: 输入层 Transpose 开销测试。"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import onnxruntime as ort
from common.benchmark import (
    benchmark_model, print_stats, save_results,
    generate_dummy_input, compare_models
)

MODEL_ORIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8m.onnx')
MODEL_MOD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modified_models', 'yolov8m_input_transpose.onnx')

WARMUP, ITERS = 10, 100

def run():
    print("=" * 60)
    print("实验 2: 输入层 Transpose 开销")
    print("=" * 60)

    base = benchmark_model(MODEL_ORIG, generate_dummy_input,
                           graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                           warmup=WARMUP, iters=ITERS)
    print_stats('Baseline', base)

    mod = benchmark_model(MODEL_MOD, generate_dummy_input,
                          graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                          warmup=WARMUP, iters=ITERS)
    print_stats('With Input Transpose Pair', mod)

    cmp = compare_models(base, mod, 'Input Transpose Pair')
    single_estimate = cmp['delta_ms'] / 2.0
    print(f"  Estimated single Transpose cost: {single_estimate:.2f} ms")

    all_results = {
        'experiment': 'exp02_input_transpose',
        'baseline': base,
        'modified': mod,
        'comparison': cmp,
        'single_transpose_estimate_ms': single_estimate,
        'note': 'round-trip Transpose pair (NCHW→NHWC→NCHW) at input; single = delta/2',
    }
    save_results(all_results, 'exp02_input_transpose.json')
    print("\n实验 2 完成")


if __name__ == '__main__':
    run()
