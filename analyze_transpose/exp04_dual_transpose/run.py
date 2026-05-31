"""实验 4: 输入+输出双 Transpose 开销测试。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import onnxruntime as ort
from common.benchmark import (
    benchmark_model, print_stats, save_results,
    generate_dummy_input, compare_models
)

MODEL_ORIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8m.onnx')
MODEL_MOD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modified_models', 'yolov8m_dual_transpose.onnx')
WARMUP, ITERS = 10, 100

def run():
    print("=" * 60)
    print("实验 4: 输入+输出双 Transpose 开销")
    print("=" * 60)
    base = benchmark_model(MODEL_ORIG, generate_dummy_input,
                           graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                           warmup=WARMUP, iters=ITERS)
    print_stats('Baseline', base)
    mod = benchmark_model(MODEL_MOD, generate_dummy_input,
                          graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                          warmup=WARMUP, iters=ITERS)
    print_stats('With Dual Transpose', mod)
    cmp = compare_models(base, mod, 'Dual Transpose')
    all_results = {
        'experiment': 'exp04_dual_transpose',
        'baseline': base, 'modified': mod, 'comparison': cmp,
        'note': 'input round-trip pair (2T) + output single (1T) = 3 Transposes total',
    }
    save_results(all_results, 'exp04_dual_transpose.json')
    print("实验 4 完成")

if __name__ == '__main__':
    run()
