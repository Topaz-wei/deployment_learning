"""统一基准测试框架: ONNX Runtime CPU。"""
import time
import json
import os
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(HERE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_session(model_path, provider='CPUExecutionProvider',
                 graph_optimization_level=None,
                 enable_profiling=False):
    """创建 ORT session，支持自定义优化级别。"""
    sess_opts = ort.SessionOptions()
    sess_opts.enable_profiling = enable_profiling
    if graph_optimization_level is not None:
        sess_opts.graph_optimization_level = graph_optimization_level
    return ort.InferenceSession(model_path, sess_opts, providers=[provider])


def benchmark_ort(model_path, input_dict, provider='CPUExecutionProvider',
                  graph_optimization_level=None,
                  warmup=10, iters=100, input_name=None):
    """运行 ORT 基准测试，返回统计 dict。"""
    sess = make_session(model_path, provider, graph_optimization_level)

    # 如果指定了 input_name，重映射输入
    if input_name and input_name not in input_dict:
        for key in list(input_dict.keys()):
            input_dict[input_name] = input_dict.pop(key)
            break

    actual_input = {sess.get_inputs()[0].name: list(input_dict.values())[0]}

    # 预热
    for _ in range(warmup):
        sess.run(None, actual_input)

    # 测量
    times = []
    for _ in range(iters):
        t0 = time.time()
        sess.run(None, actual_input)
        times.append((time.time() - t0) * 1000)

    times = np.array(times)
    return {
        'mean_ms': float(times.mean()),
        'min_ms': float(times.min()),
        'max_ms': float(times.max()),
        'std_ms': float(times.std()),
        'p50_ms': float(np.median(times)),
        'fps': float(1000.0 / times.mean()),
        'iters': iters,
        'warmup': warmup,
        'raw_times_ms': [float(t) for t in times],
    }


def benchmark_model(model_path, dummy_input_fn,
                    provider='CPUExecutionProvider',
                    graph_optimization_level=None,
                    warmup=10, iters=100):
    """便捷 wrapper: dummy_input_fn() 生成输入 → benchmark_ort。"""
    inputs = dummy_input_fn()
    result = benchmark_ort(model_path, inputs, provider,
                           graph_optimization_level, warmup, iters)
    # 记录节点数
    import onnx
    model = onnx.load(model_path)
    from common.model_utils import count_nodes
    result['node_count'] = count_nodes(model)
    return result


def print_stats(label, stats):
    """打印格式化统计信息"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Latency (ms): mean={stats['mean_ms']:.2f}, min={stats['min_ms']:.2f}, "
          f"max={stats['max_ms']:.2f}, p50={stats['p50_ms']:.2f}, std={stats['std_ms']:.2f}")
    print(f"  FPS: {stats['fps']:.2f}")
    if 'node_count' in stats:
        nc = stats['node_count']
        print(f"  Nodes: total={nc['total']}, Conv={nc['by_type'].get('Conv', 0)}, "
              f"Transpose={nc['by_type'].get('Transpose', 0)}")


def save_results(results, filename):
    """保存结果到 JSON"""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {path}")
    return path


def generate_dummy_input(batch=1, channels=3, height=640, width=640):
    """生成随机 float32 输入"""
    return {'images': np.random.randn(batch, channels, height, width).astype(np.float32)}


def compare_models(baseline_stats, modified_stats, label='Modified'):
    """对比 baseline 和修改模型的延迟，打印增量"""
    delta = modified_stats['mean_ms'] - baseline_stats['mean_ms']
    ratio = modified_stats['mean_ms'] / baseline_stats['mean_ms']
    print(f"\n  --- {label} vs Baseline ---")
    print(f"  Baseline:  {baseline_stats['mean_ms']:.2f} ms")
    print(f"  {label}: {modified_stats['mean_ms']:.2f} ms")
    print(f"  Delta: {delta:+.2f} ms ({ratio:.4f}x)")
    return {'delta_ms': delta, 'ratio': ratio}
