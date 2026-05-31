"""实验 7: ORT 图优化对 Transpose 的消除。对比不同优化级别下原始模型和含 Transpose 模型的节点数+延迟。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import json
import numpy as np
import onnxruntime as ort
from common.benchmark import (
    benchmark_model, save_results, generate_dummy_input
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ORIG = os.path.join(HERE, '..', 'yolov8m.onnx')
MODEL_MOD = os.path.join(HERE, '..', 'modified_models', 'yolov8m_input_transpose.onnx')

OPT_LEVELS = {
    'DISABLE_ALL': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    'BASIC': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    'EXTENDED': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    'ALL': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}

WARMUP, ITERS = 10, 100

def run():
    print("=" * 60)
    print("实验 7: ORT 图优化对 Transpose 的消除")
    print("=" * 60)
    all_results = {'experiment': 'exp07_ort_opt_transpose'}
    print(f"\n{'Optimization':<16} {'Model':<12} {'Nodes':<8} {'Transpose':<12} {'Mean(ms)':<12} {'FPS':<10}")
    print("-" * 75)

    for opt_name, opt_val in OPT_LEVELS.items():
        for label, model_path in [('Original', MODEL_ORIG), ('Modified', MODEL_MOD)]:
            stats = benchmark_model(model_path, generate_dummy_input,
                                    graph_optimization_level=opt_val, warmup=WARMUP, iters=ITERS)
            nc = stats['node_count']
            t_count = nc['by_type'].get('Transpose', 0)
            print(f"  {opt_name:<16} {label:<12} {nc['total']:<8} {t_count:<12} {stats['mean_ms']:<12.2f} {stats['fps']:<10.2f}")
            all_results[f'{opt_name}_{label}'] = {
                'optimization': opt_name, 'model': label,
                'total_nodes': nc['total'], 'transpose_nodes': t_count,
                'mean_ms': stats['mean_ms'], 'fps': stats['fps'],
            }
    save_results(all_results, 'exp07_ort_opt_transpose.json')
    print("\n实验 7 完成")

if __name__ == '__main__':
    run()
