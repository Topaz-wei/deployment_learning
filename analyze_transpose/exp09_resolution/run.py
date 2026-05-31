"""实验 9: 不同输入分辨率下的 Transpose 开销。
注：ONNX 模型输入 shape 固定为 1x3x640x640，仅测试 640x640 分辨率。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import onnxruntime as ort
from common.benchmark import (
    benchmark_model, save_results
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ORIG = os.path.join(HERE, '..', 'yolov8m.onnx')
MODEL_MOD = os.path.join(HERE, '..', 'modified_models', 'yolov8m_input_transpose.onnx')

RESOLUTIONS = [(640, 640)]
WARMUP, ITERS = 10, 100

def run():
    print("=" * 60)
    print("实验 9: 不同分辨率下的 Transpose 开销")
    print("=" * 60)
    all_results = {'experiment': 'exp09_resolution', 'results': []}

    print(f"\n{'Resolution':<16} {'Model':<12} {'Mean(ms)':<12} {'FPS':<10} {'Pixels':<12} {'ns/pixel':<12}")
    print("-" * 80)

    for h, w in RESOLUTIONS:
        def input_fn():
            return {'images': np.random.randn(1, 3, h, w).astype(np.float32)}

        for label, model_path in [('Original', MODEL_ORIG), ('Modified', MODEL_MOD)]:
            stats = benchmark_model(model_path, input_fn,
                                    graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                                    warmup=WARMUP, iters=ITERS)
            n_pixels = 3 * h * w
            ns_per_pixel = stats['mean_ms'] * 1e6 / n_pixels
            print(f"  {h}x{w:<10} {label:<12} {stats['mean_ms']:<12.2f} {stats['fps']:<10.2f} {n_pixels:<12} {ns_per_pixel:<12.3f}")
            all_results['results'].append({
                'resolution': f'{h}x{w}', 'model': label,
                'mean_ms': stats['mean_ms'], 'fps': stats['fps'],
                'pixels': n_pixels, 'ns_per_pixel': ns_per_pixel,
            })

    save_results(all_results, 'exp09_resolution.json')
    print("\n实验 9 完成")

if __name__ == '__main__':
    run()
