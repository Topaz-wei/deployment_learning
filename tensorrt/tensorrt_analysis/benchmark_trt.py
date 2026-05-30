"""TensorRT 4 模型精度+速度基准测试

测量每个 TensorRT engine 的单次推理耗时，
并与 ONNX Runtime FP32 比对输出精度。

用法:
    conda activate py38
    cd /home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/tensorrt_analysis
    python3 benchmark_trt.py 2>&1 | tee benchmark_results.txt
"""

import os, sys, time, ctypes, json
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, '/home/ssd/code/vh3/src/py_algorithm')

# 预加载 torch 库
CONDA_LIB = '/home/ssd/anaconda3/envs/py38/lib'
for _lib in ['libopenblas.so.0', 'libgomp.so.1']:
    try:
        ctypes.CDLL(f'{CONDA_LIB}/{_lib}', mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

from trt_engine import TrtEngine

WEIGHTS = os.path.join(HERE, '..', '..', 'data', 'weights')
ENGINES = os.path.join(WEIGHTS, 'engines')

# ===== 测试参数 =====
WARMUP_ITERS = 10
BENCH_ITERS = 100


def bench_engine(name, engine_path, dummy_input_fn, onnx_path=None):
    """对单个 engine 做基准测试"""
    print(f"\n{'='*60}")
    print(f"[{name}]")
    print(f"  Engine: {engine_path}")
    if onnx_path:
        print(f"  ONNX:   {onnx_path}")

    # 加载 TRT engine
    t0 = time.time()
    engine = TrtEngine(engine_path)
    load_time = (time.time() - t0) * 1000
    print(f"  加载耗时: {load_time:.1f} ms")

    # 准备输入
    inputs = dummy_input_fn()
    input_info = {k: f'shape={v.shape}, dtype={v.dtype}' for k, v in inputs.items()}
    print(f"  输入: {input_info}")

    # 预热
    for _ in range(WARMUP_ITERS):
        engine.infer(inputs)

    # 测速
    times = []
    for _ in range(BENCH_ITERS):
        t0 = time.time()
        engine.infer(inputs)
        times.append((time.time() - t0) * 1000)

    times = np.array(times)
    trt_output = engine.infer(inputs)
    output_info = {k: f'shape={v.shape}, dtype={v.dtype}, range=[{v.min():.4f}, {v.max():.4f}]'
                   for k, v in trt_output.items()}
    print(f"  输出: {output_info}")
    print(f"  推理耗时 (ms): mean={times.mean():.3f}, min={times.min():.3f}, "
          f"max={times.max():.3f}, std={times.std():.3f}, p50={np.median(times):.3f}")

    # ONNX Runtime FP32 精度对比（ONNX Runtime 在此环境下仅支持 CPU，因此仅用于精度对比）
    onnx_output = None
    onnx_times = None
    if onnx_path and os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            # 构建与 TRT 输入匹配的 ONNX 输入
            onnx_inputs = {}
            for inp in sess.get_inputs():
                name = inp.name
                for k, v in inputs.items():
                    if k == name or name.startswith(k):
                        onnx_inputs[name] = v.astype(np.float32)
                        break
                else:
                    onnx_inputs[name] = inputs[list(inputs.keys())[0]].astype(np.float32)

            # 仅运行一次用于精度对比（CPU ONNX 不做速度对比）
            onnx_output = sess.run(None, onnx_inputs)
            onnx_output = {sess.get_outputs()[i].name: onnx_output[i]
                          for i in range(len(onnx_output))}
            print(f"  ONNX FP32 (CPU) 推理完成，仅用于精度对比")

            # 精度对比
            for key in trt_output:
                if key in onnx_output:
                    trt_val = trt_output[key].astype(np.float64).ravel()
                    onnx_val = onnx_output[key].astype(np.float64).ravel()
                    min_len = min(len(trt_val), len(onnx_val))
                    trt_val = trt_val[:min_len]
                    onnx_val = onnx_val[:min_len]
                    diff = trt_val - onnx_val
                    abs_diff = np.abs(diff)
                    rel_diff = np.abs(diff / (np.abs(onnx_val) + 1e-8))
                    cos_sim = np.dot(trt_val, onnx_val) / (
                        np.linalg.norm(trt_val) * np.linalg.norm(onnx_val) + 1e-8)
                    print(f"  精度对比 [{key}]:")
                    print(f"    TRT(FP16)   range: [{trt_output[key].min():.6f}, {trt_output[key].max():.6f}]")
                    print(f"    ONNX(FP32)  range: [{onnx_output[key].min():.6f}, {onnx_output[key].max():.6f}]")
                    print(f"    max_abs_diff: {abs_diff.max():.6f}")
                    print(f"    mean_abs_diff: {abs_diff.mean():.6f}")
                    print(f"    max_rel_diff: {rel_diff.max():.6f}")
                    print(f"    mean_rel_diff: {rel_diff.mean():.6f}")
                    print(f"    cosine_sim:   {cos_sim:.8f}")
        except Exception as e:
            print(f"  ONNX 对比失败: {e}")

    del engine
    return {
        'name': name,
        'engine_path': engine_path,
        'load_time_ms': load_time,
        'infer_time_ms_mean': float(times.mean()),
        'infer_time_ms_min': float(times.min()),
        'infer_time_ms_max': float(times.max()),
        'infer_time_ms_std': float(times.std()),
        'infer_time_ms_p50': float(np.median(times)),
        'trt_fps': 1000.0 / times.mean(),
        'onnx_time_ms_mean': float(onnx_times.mean()) if onnx_times is not None else None,
        'onnx_fps': 1000.0 / onnx_times.mean() if onnx_times is not None else None,
        'speedup': (float(onnx_times.mean()) / float(times.mean())) if onnx_times is not None else None,
    }


# ===== 各模型 dummy input =====

def yolo_input():
    """YOLOv8m: 640x640 RGB [0,1] NCHW"""
    img = np.random.rand(1, 3, 640, 640).astype(np.float32)
    return {'images': img}

def sixdrepnet_input():
    """SixDRepNet: 192x192 RGB normalized NCHW"""
    img = np.random.rand(1, 3, 192, 192).astype(np.float32)
    # 模拟 normalize 后的值 (~N(0,1))
    img = (img - 0.45) / 0.23
    return {'input': img}

def sface_input():
    """SFace: 112x112 RGB normalized [-1,1] NCHW"""
    img = np.random.rand(1, 3, 112, 112).astype(np.float32)
    img = (img - 0.5) * 2.0  # ~[-1, 1]
    return {'data': img}

def repvit_input():
    """RepViT: 224x224 RGB normalized NCHW"""
    img = np.random.rand(1, 3, 224, 224).astype(np.float32)
    img = (img - 0.45) / 0.23
    return {'input': img}


# ===== 运行测试 =====
if __name__ == '__main__':
    print("TensorRT 基准测试开始")
    print(f"Engine 目录: {ENGINES}")
    print(f"预热: {WARMUP_ITERS}, 测量: {BENCH_ITERS} iterations\n")

    results = []

    tests = [
        ('YOLOv8m', f'{ENGINES}/yolov8m_fp16.engine', yolo_input, f'{WEIGHTS}/yolov8m.onnx'),
        ('SixDRepNet', f'{ENGINES}/sixdrepnet_fp16.engine', sixdrepnet_input, f'{WEIGHTS}/sixdrepnet.onnx'),
        ('SFace', f'{ENGINES}/sface_fp16.engine', sface_input, f'{WEIGHTS}/sface_cleaned.onnx'),
        ('RepViT', f'{ENGINES}/repvit_fp16.engine', repvit_input, f'{WEIGHTS}/upperbody_repvit.onnx'),
    ]

    for name, engine_path, input_fn, onnx_path in tests:
        if not os.path.exists(engine_path):
            print(f"[{name}] SKIP: engine 文件不存在: {engine_path}")
            continue
        r = bench_engine(name, engine_path, input_fn, onnx_path=onnx_path)
        results.append(r)

    # ===== 汇总 =====
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)
    print(f"{'模型':<15} {'TRT(ms)':>9} {'ONNX(ms)':>9} {'加速比':>7} {'TRT_FPS':>8}")
    print("-" * 55)
    for r in results:
        trt = r['infer_time_ms_mean']
        onnx = r.get('onnx_time_ms_mean')
        speedup = r.get('speedup')
        fps = r['trt_fps']
        onnx_str = f"{onnx:.3f}" if onnx else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
        fps_str = f"{fps:.1f}"
        print(f"{r['name']:<15} {trt:>9.3f} {onnx_str:>9} {speedup_str:>7} {fps_str:>8}")

    # 保存 JSON
    output_path = HERE + '/benchmark_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_path}")
