"""实验 1: Baseline 基准测试 — 原始 YOLOv8m ONNX 无修改。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import onnxruntime as ort
from common.benchmark import (
    benchmark_model, print_stats, save_results,
    generate_dummy_input, compare_models
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8m.onnx')

OPT_LEVELS = {
    'DISABLE_ALL': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    'BASIC': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    'EXTENDED': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    'ALL': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}

WARMUP, ITERS = 10, 100

def run():
    print("=" * 60)
    print("实验 1: Baseline 基准测试")
    print("=" * 60)

    all_results = {'experiment': 'exp01_baseline', 'model': MODEL_PATH}
    baseline = None

    for opt_name, opt_val in OPT_LEVELS.items():
        print(f"\n--- ORT Optimization: {opt_name} ---")
        stats = benchmark_model(
            MODEL_PATH, generate_dummy_input,
            graph_optimization_level=opt_val,
            warmup=WARMUP, iters=ITERS
        )
        stats['optimization_level'] = opt_name
        print_stats(f'Baseline ({opt_name})', stats)
        all_results[opt_name] = stats
        if opt_name == 'ALL':
            baseline = stats

    save_results(all_results, 'exp01_baseline.json')
    print("\n实验 1 完成")
    return baseline


if __name__ == '__main__':
    run()
