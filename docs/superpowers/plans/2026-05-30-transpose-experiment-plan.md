# Transpose 实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `analyze_transpose/` 下实现 9 个 Transpose 插入实验，用 ONNX Runtime 测量 YOLOv8m 在不同位置/数量/分辨率下 Transpose 节点的推理开销。

**Architecture:** `common/` 提供 ONNX 图操作工具 (`model_utils.py`) 和统一基准测试 (`benchmark.py`)。每个实验在独立子目录下，`modify_model.py` 生成修改版 ONNX 到 `modified_models/`，`run.py` 加载并对比测试，结果输出到 `results/`。关键策略: 布局修改型 Transpose 会破坏下游 Conv (weight 通道数不匹配)，使用 round-trip Transpose pair (NCHW→NHWC→NCHW) 保持兼容；输出端为 3D tensor 使用单次 Transpose。

**Tech Stack:** Python 3.8 (conda py38), onnx 1.17, onnxruntime 1.19 (CPU, Jetson Orin ARM64), numpy, YOLOv8m ONNX

**Graph analysis:**
- 输入: `images` [1,3,640,640] → 首层 Conv: `/model.0/conv/Conv` (index 0)
- 输出: `/model.22/Concat_5` (index 324) → `output0` [1,84,8400] 3D
- Backbone stage 输出 Conv: [0] `/model.0/conv/Conv`, [25] `/model.2/cv2/conv/Conv`, [64] `/model.4/cv2/conv/Conv`, [103] `/model.6/cv2/conv/Conv`
- 第 20 个 Conv: `/model.2/m.1/cv2/conv/Conv` (index 20)

**Validated approaches:**
- Input round-trip Transpose pair: OK (model runs, output correct shape)
- Output single Transpose (perm=[0,2,1]): OK (output changes to [1,8400,84])
- Mid-model round-trip Transpose pair: OK
- Multi round-trip (4 pairs = 8 Transposes): OK

---

### Task 1: 公共基础设施 — model_utils.py

**Files:** Create `analyze_transpose/common/__init__.py`, `analyze_transpose/common/model_utils.py`

- [ ] **Step 1: 创建目录和空 __init__.py**

```bash
mkdir -p analyze_transpose/common analyze_transpose/modified_models analyze_transpose/results
touch analyze_transpose/common/__init__.py
```

- [ ] **Step 2: 编写 model_utils.py**

```python
"""ONNX 图操作工具: 加载、修改、保存模型。"""
import onnx
from onnx import helper, TensorProto
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODIFIED_DIR = os.path.join(HERE, 'modified_models')
os.makedirs(MODIFIED_DIR, exist_ok=True)


def load_model(path):
    """加载 ONNX 模型"""
    return onnx.load(path)


def save_model(model, path, check=False):
    """保存 ONNX 模型，可选 onnx.checker 验证"""
    if check:
        onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"  [save] {path}")
    return path


def make_transpose_node(name, input_name, output_name, perm):
    """创建 Transpose 节点"""
    return helper.make_node('Transpose', [input_name], [output_name], name=name, perm=perm)


def make_transpose_pair(base_name, input_name, output_name, fwd_perm, rev_perm):
    """创建 round-trip Transpose 节点对。返回 (t1, t2, intermediate_name)"""
    mid_name = f'{input_name}_{base_name}_mid'
    t1 = make_transpose_node(f'{base_name}_fwd', input_name, mid_name, fwd_perm)
    t2 = make_transpose_node(f'{base_name}_rev', mid_name, output_name, rev_perm)
    return t1, t2, mid_name


def insert_nodes_after(graph, target_node_output_name, new_nodes):
    """在产生 target_node_output_name 的节点之后插入 new_nodes，
    并将所有下游消费者的对应输入重定向到 new_nodes 最后一个的输出。"""
    new_output = new_nodes[-1].output[0]
    new_names = {n.name for n in new_nodes}
    rewired = 0
    for n in graph.node:
        for i, inp in enumerate(n.input):
            if inp == target_node_output_name and n.name not in new_names:
                n.input[i] = new_output
                rewired += 1
    # 找到产生 target_node_output_name 的节点
    insert_pos = 0
    for idx, n in enumerate(graph.node):
        if target_node_output_name in n.output:
            insert_pos = idx
            break
    for n in reversed(new_nodes):
        graph.node.insert(insert_pos + 1, n)
    return rewired


def insert_nodes_at_graph_input(graph, input_name, new_nodes):
    """在 graph 输入 input_name 之后插入 new_nodes，
    将原 consumer 的输入重定向到 new_nodes 最后一个的输出。"""
    new_output = new_nodes[-1].output[0]
    new_names = {n.name for n in new_nodes}
    rewired = 0
    for n in graph.node:
        for i, inp in enumerate(n.input):
            if inp == input_name and n.name not in new_names:
                n.input[i] = new_output
                rewired += 1
    for n in reversed(new_nodes):
        graph.node.insert(0, n)
    return rewired


def insert_transpose_at_graph_output(graph, output_name, perm, transpose_name='output_transpose'):
    """在 graph 输出 output_name 之后插入单次 Transpose 并重定向 graph.output。
    仅适用于输出为 3D tensor 的场景。"""
    new_output = f'{output_name}_transposed'
    t_node = make_transpose_node(transpose_name, output_name, new_output, perm)
    graph.node.append(t_node)
    for out in graph.output:
        if out.name == output_name:
            graph.output.remove(out)
            break
    graph.output.append(helper.make_tensor_value_info(new_output, TensorProto.FLOAT, None))
    return new_output


def find_node_by_name(graph, name):
    """按名称查找节点，返回 (index, node) 或 (None, None)"""
    for i, n in enumerate(graph.node):
        if n.name == name:
            return i, n
    return None, None


def find_nodes_by_op_type(graph, op_type):
    """返回所有匹配 op_type 的 (index, node) 列表"""
    return [(i, n) for i, n in enumerate(graph.node) if n.op_type == op_type]


def count_nodes(graph):
    """统计节点数（按类型）"""
    from collections import Counter
    cnt = Counter(n.op_type for n in graph.node)
    return {'total': len(graph.node), 'by_type': dict(cnt)}


def print_graph_summary(graph, label=''):
    """打印图摘要"""
    counts = count_nodes(graph)
    print(f"  [{label}] 总节点: {counts['total']}")
    transpose_count = counts['by_type'].get('Transpose', 0)
    conv_count = counts['by_type'].get('Conv', 0)
    print(f"  [{label}] Conv: {conv_count}, Transpose: {transpose_count}")
```

- [ ] **Step 3: 验证 model_utils.py 可导入**

```bash
conda run -n py38 python3 -c "import sys; sys.path.insert(0, 'analyze_transpose'); from common.model_utils import *; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add analyze_transpose/common/
git commit -m "feat: add common model_utils for ONNX graph manipulation

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: 公共基础设施 — benchmark.py

**Files:** Modify `analyze_transpose/common/benchmark.py`

- [ ] **Step 1: 编写 benchmark.py**

```python
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
```

- [ ] **Step 2: 验证 benchmark.py 可导入**

```bash
conda run -n py38 python3 -c "import sys; sys.path.insert(0, 'analyze_transpose'); from common.benchmark import *; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add analyze_transpose/common/benchmark.py
git commit -m "feat: add common benchmark framework for ORT testing

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: 实验 1 — Baseline 基准测试

**Files:** Create `analyze_transpose/exp01_baseline/run.py`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p analyze_transpose/exp01_baseline
```

- [ ] **Step 2: 编写 run.py**

```python
"""实验 1: Baseline 基准测试 — 原始 YOLOv8m ONNX 无修改。
测量 ORT 不同优化级别 (DISABLE_ALL / BASIC / ALL) 下的推理延迟。
"""
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
```

- [ ] **Step 3: 运行验证**

```bash
conda run -n py38 python3 analyze_transpose/exp01_baseline/run.py
```

Expected: 输出 4 种优化级别的延迟统计，保存 `results/exp01_baseline.json`。

- [ ] **Step 4: Commit**

```bash
git add analyze_transpose/exp01_baseline/ analyze_transpose/results/exp01_baseline.json
git commit -m "feat: add exp01 baseline benchmark for YOLOv8m ONNX

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: 实验 2 — 输入层 Transpose

**Files:** Create `analyze_transpose/exp02_input_transpose/modify_model.py`, `run.py`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p analyze_transpose/exp02_input_transpose
```

- [ ] **Step 2: 编写 modify_model.py**

```python
"""实验 2: 输入层 Transpose — 在 images 后插入 NCHW→NHWC→NCHW round-trip Transpose pair。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.model_utils import (
    load_model, save_model, make_transpose_pair,
    insert_nodes_at_graph_input, print_graph_summary
)

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8m.onnx')
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modified_models', 'yolov8m_input_transpose.onnx')

def modify():
    model = load_model(SRC)
    graph = model.graph
    print_graph_summary(graph, 'before')

    t1, t2, _ = make_transpose_pair(
        'input', 'images', 'images_transposed_back',
        fwd_perm=[0, 2, 3, 1],   # NCHW → NHWC
        rev_perm=[0, 3, 1, 2],   # NHWC → NCHW
    )
    insert_nodes_at_graph_input(graph, 'images', [t1, t2])

    print_graph_summary(graph, 'after')
    save_model(model, DST)
    print(f"  Modified model: {DST}")
    print(f"  Note: round-trip pair — 单次 Transpose 开销 = (delta from baseline) / 2")


if __name__ == '__main__':
    modify()
```

- [ ] **Step 3: 编写 run.py**

```python
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

    # Baseline
    base = benchmark_model(MODEL_ORIG, generate_dummy_input,
                           graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                           warmup=WARMUP, iters=ITERS)
    print_stats('Baseline', base)

    # Modified
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
```

- [ ] **Step 4: 生成修改模型并运行**

```bash
conda run -n py38 python3 analyze_transpose/exp02_input_transpose/modify_model.py
conda run -n py38 python3 analyze_transpose/exp02_input_transpose/run.py
```

Expected: 输出 baseline vs modified 延迟对比，估计单个 Transpose 开销。

- [ ] **Step 5: Commit**

```bash
git add analyze_transpose/exp02_input_transpose/ analyze_transpose/modified_models/ analyze_transpose/results/exp02_input_transpose.json
git commit -m "feat: add exp02 input transpose overhead experiment

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: 实验 3 — 输出层 Transpose

**Files:** Create `analyze_transpose/exp03_output_transpose/modify_model.py`, `run.py`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p analyze_transpose/exp03_output_transpose
```

- [ ] **Step 2: 编写 modify_model.py**

```python
"""实验 3: 输出层 Transpose — 在 output0 之后插入单次 Transpose(perm=[0,2,1])。
输出从 [1,84,8400] 变为 [1,8400,84]。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.model_utils import (
    load_model, save_model, insert_transpose_at_graph_output, print_graph_summary
)

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8m.onnx')
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modified_models', 'yolov8m_output_transpose.onnx')

def modify():
    model = load_model(SRC)
    graph = model.graph
    print_graph_summary(graph, 'before')
    new_out = insert_transpose_at_graph_output(graph, 'output0', perm=[0, 2, 1])
    print(f"  Graph output redirected: output0 -> {new_out}")
    print_graph_summary(graph, 'after')
    save_model(model, DST)


if __name__ == '__main__':
    modify()
```

- [ ] **Step 3: 编写 run.py**

```python
"""实验 3: 输出层 Transpose 开销测试。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import onnxruntime as ort
from common.benchmark import (
    benchmark_model, print_stats, save_results,
    generate_dummy_input, compare_models
)

MODEL_ORIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8m.onnx')
MODEL_MOD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modified_models', 'yolov8m_output_transpose.onnx')

WARMUP, ITERS = 10, 100

def run():
    print("=" * 60)
    print("实验 3: 输出层 Transpose 开销")
    print("=" * 60)

    base = benchmark_model(MODEL_ORIG, generate_dummy_input,
                           graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                           warmup=WARMUP, iters=ITERS)
    print_stats('Baseline', base)

    mod = benchmark_model(MODEL_MOD, generate_dummy_input,
                          graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                          warmup=WARMUP, iters=ITERS)
    print_stats('With Output Transpose', mod)

    cmp = compare_models(base, mod, 'Output Transpose')

    all_results = {
        'experiment': 'exp03_output_transpose',
        'baseline': base,
        'modified': mod,
        'comparison': cmp,
        'note': 'single Transpose(perm=[0,2,1]) at output, 3D tensor [1,84,8400]->[1,8400,84]',
    }
    save_results(all_results, 'exp03_output_transpose.json')
    print("\n实验 3 完成")


if __name__ == '__main__':
    run()
```

- [ ] **Step 4: 生成修改模型并运行**

```bash
conda run -n py38 python3 analyze_transpose/exp03_output_transpose/modify_model.py
conda run -n py38 python3 analyze_transpose/exp03_output_transpose/run.py
```

- [ ] **Step 5: Commit**

```bash
git add analyze_transpose/exp03_output_transpose/ analyze_transpose/modified_models/yolov8m_output_transpose.onnx analyze_transpose/results/exp03_output_transpose.json
git commit -m "feat: add exp03 output transpose overhead experiment

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: 实验 4 — 输入+输出双 Transpose

**Files:** Create `analyze_transpose/exp04_dual_transpose/modify_model.py`, `run.py`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p analyze_transpose/exp04_dual_transpose
```

- [ ] **Step 2: 编写 modify_model.py**

```python
"""实验 4: 输入+输出双 Transpose — 同时插入输入 round-trip pair 和输出单次 Transpose。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.model_utils import (
    load_model, save_model, make_transpose_pair,
    insert_nodes_at_graph_input, insert_transpose_at_graph_output,
    print_graph_summary
)

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8m.onnx')
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modified_models', 'yolov8m_dual_transpose.onnx')

def modify():
    model = load_model(SRC)
    graph = model.graph
    print_graph_summary(graph, 'before')

    # Input: round-trip Transpose pair
    t1, t2, _ = make_transpose_pair(
        'input', 'images', 'images_transposed_back',
        fwd_perm=[0, 2, 3, 1], rev_perm=[0, 3, 1, 2],
    )
    insert_nodes_at_graph_input(graph, 'images', [t1, t2])

    # Output: single Transpose
    insert_transpose_at_graph_output(graph, 'output0', perm=[0, 2, 1], transpose_name='output_dual_t')

    print_graph_summary(graph, 'after')
    save_model(model, DST)
    print(f"  Input: round-trip pair (NCHW→NHWC→NCHW), Output: single Transpose([0,2,1])")


if __name__ == '__main__':
    modify()
```

- [ ] **Step 3: 编写 run.py**

```python
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
    print_stats('With Input+Output Transpose', mod)

    cmp = compare_models(base, mod, 'Dual Transpose')

    all_results = {
        'experiment': 'exp04_dual_transpose',
        'baseline': base,
        'modified': mod,
        'comparison': cmp,
        'note': 'input round-trip pair (2 Transposes) + output single Transpose (1 Transpose) = 3 total',
    }
    save_results(all_results, 'exp04_dual_transpose.json')
    print("\n实验 4 完成")


if __name__ == '__main__':
    run()
```

- [ ] **Step 4: 生成修改模型并运行**

```bash
conda run -n py38 python3 analyze_transpose/exp04_dual_transpose/modify_model.py
conda run -n py38 python3 analyze_transpose/exp04_dual_transpose/run.py
```

- [ ] **Step 5: Commit**

```bash
git add analyze_transpose/exp04_dual_transpose/ analyze_transpose/modified_models/yolov8m_dual_transpose.onnx analyze_transpose/results/exp04_dual_transpose.json
git commit -m "feat: add exp04 dual transpose experiment

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 7: 实验 5 — 中间层 Transpose

**Files:** Create `analyze_transpose/exp05_mid_transpose/modify_model.py`, `run.py`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p analyze_transpose/exp05_mid_transpose
```

- [ ] **Step 2: 编写 modify_model.py**

```python
"""实验 5: 中间层 Transpose — 在第 20 个 Conv 输出后插入 round-trip Transpose pair。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.model_utils import (
    load_model, save_model, make_transpose_pair,
    insert_nodes_after, print_graph_summary
)

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8m.onnx')
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modified_models', 'yolov8m_mid_transpose.onnx')

TARGET_CONV_IDX = 20  # 第 20 个 Conv (0-indexed)

def modify():
    model = load_model(SRC)
    graph = model.graph
    print_graph_summary(graph, 'before')

    # 找到第 20 个 Conv 节点
    conv_count = 0
    target = None
    for n in graph.node:
        if n.op_type == 'Conv':
            if conv_count == TARGET_CONV_IDX:
                target = n
                break
            conv_count += 1

    if target is None:
        raise RuntimeError(f'Could not find Conv #{TARGET_CONV_IDX}')

    print(f"  Target Conv: [{conv_count}] {target.name}, output={list(target.output)}")
    assert target.name == '/model.2/m.1/cv2/conv/Conv', f'Unexpected target: {target.name}'

    orig_out = target.output[0]
    t1, t2, _ = make_transpose_pair(
        'mid', orig_out, f'{orig_out}_back',
        fwd_perm=[0, 2, 3, 1], rev_perm=[0, 3, 1, 2],
    )
    rewired = insert_nodes_after(graph, orig_out, [t1, t2])
    print(f"  Rewired {rewired} downstream consumers")

    print_graph_summary(graph, 'after')
    save_model(model, DST)
    print(f"  Note: round-trip pair at mid-network Conv — 单次 Transpose = delta/2")


if __name__ == '__main__':
    modify()
```

- [ ] **Step 3: 编写 run.py**

```python
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
        'baseline': base,
        'modified': mod,
        'comparison': cmp,
        'single_transpose_estimate_ms': single_est,
        'note': f'Round-trip pair after 20th Conv (node: /model.2/m.1/cv2/conv/Conv); single = delta/2',
    }
    save_results(all_results, 'exp05_mid_transpose.json')
    print("\n实验 5 完成")


if __name__ == '__main__':
    run()
```

- [ ] **Step 4: 生成修改模型并运行**

```bash
conda run -n py38 python3 analyze_transpose/exp05_mid_transpose/modify_model.py
conda run -n py38 python3 analyze_transpose/exp05_mid_transpose/run.py
```

- [ ] **Step 5: Commit**

```bash
git add analyze_transpose/exp05_mid_transpose/ analyze_transpose/modified_models/yolov8m_mid_transpose.onnx analyze_transpose/results/exp05_mid_transpose.json
git commit -m "feat: add exp05 mid-network transpose experiment

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 8: 实验 6 — 多层 Transpose 累积效应

**Files:** Create `analyze_transpose/exp06_multi_transpose/modify_model.py`, `run.py`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p analyze_transpose/exp06_multi_transpose
```

- [ ] **Step 2: 编写 modify_model.py**

```python
"""实验 6: 多层 Transpose 累积 — 在 Backbone 4 个 Stage 输出后各插入 round-trip Transpose pair。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.model_utils import (
    load_model, save_model, make_transpose_pair,
    insert_nodes_after, print_graph_summary
)

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8m.onnx')
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modified_models', 'yolov8m_multi_transpose.onnx')

# Backbone 4 个 Stage 输出 Conv 名称
STAGE_CONV_NAMES = [
    '/model.0/conv/Conv',
    '/model.2/cv2/conv/Conv',
    '/model.4/cv2/conv/Conv',
    '/model.6/cv2/conv/Conv',
]

def modify():
    model = load_model(SRC)
    graph = model.graph
    print_graph_summary(graph, 'before')

    # 构建 name→node 索引
    name_to_node = {n.name: n for n in graph.node}

    total_inserted = 0
    for i, name in enumerate(STAGE_CONV_NAMES):
        target = name_to_node.get(name)
        if target is None:
            print(f"  WARNING: node {name} not found, skipping")
            continue

        orig_out = target.output[0]
        t1, t2, _ = make_transpose_pair(
            f'stage{i}', orig_out, f'{orig_out}_back',
            fwd_perm=[0, 2, 3, 1], rev_perm=[0, 3, 1, 2],
        )
        rewired = insert_nodes_after(graph, orig_out, [t1, t2])
        print(f"  Stage {i} [{name}]: rewired {rewired} consumers")
        total_inserted += 2

    print(f"  Total Transpose nodes inserted: {total_inserted}")
    print_graph_summary(graph, 'after')
    save_model(model, DST)
    print(f"  Note: 4 round-trip pairs = 8 Transposes total")


if __name__ == '__main__':
    modify()
```

- [ ] **Step 3: 编写 run.py**

```python
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
        'baseline': base,
        'modified': mod,
        'comparison': cmp,
        'per_roundtrip_pair_ms': per_pair,
        'per_single_transpose_ms': per_single,
        'note': '4 round-trip pairs (8 Transposes) at backbone stage Conv outputs',
    }
    save_results(all_results, 'exp06_multi_transpose.json')
    print(f"\n  每对 (2 Transpose) 边际延迟: {per_pair:.2f} ms")
    print(f"  每个 Transpose 边际延迟: {per_single:.2f} ms")
    print("\n实验 6 完成")


if __name__ == '__main__':
    run()
```

- [ ] **Step 4: 生成修改模型并运行**

```bash
conda run -n py38 python3 analyze_transpose/exp06_multi_transpose/modify_model.py
conda run -n py38 python3 analyze_transpose/exp06_multi_transpose/run.py
```

- [ ] **Step 5: Commit**

```bash
git add analyze_transpose/exp06_multi_transpose/ analyze_transpose/modified_models/yolov8m_multi_transpose.onnx analyze_transpose/results/exp06_multi_transpose.json
git commit -m "feat: add exp06 multi-stage transpose cumulative effect experiment

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 9: 实验 7 — ORT 图优化对 Transpose 的消除

**Files:** Create `analyze_transpose/exp07_ort_opt_transpose/run.py`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p analyze_transpose/exp07_ort_opt_transpose
```

- [ ] **Step 2: 编写 run.py**

```python
"""实验 7: ORT 图优化对 Transpose 的消除 — 对比不同优化级别下的节点数和延迟。

对原始模型和含 Transpose 的修改模型分别测试 4 种优化级别，
观察 Transpose 节点是否被常量折叠/算子融合消除。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import json
import numpy as np
import onnxruntime as ort
from common.benchmark import (
    benchmark_model, print_stats, save_results,
    generate_dummy_input, make_session
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ORIG = os.path.join(HERE, '..', 'yolov8m.onnx')
# 使用 exp2 的修改模型 (含输入 round-trip Transpose pair)
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
            stats = benchmark_model(
                model_path, generate_dummy_input,
                graph_optimization_level=opt_val,
                warmup=WARMUP, iters=ITERS
            )
            nc = stats['node_count']
            t_count = nc['by_type'].get('Transpose', 0)
            print(f"  {opt_name:<16} {label:<12} {nc['total']:<8} {t_count:<12} {stats['mean_ms']:<12.2f} {stats['fps']:<10.2f}")
            all_results[f'{opt_name}_{label}'] = {
                'optimization': opt_name,
                'model': label,
                'total_nodes': nc['total'],
                'transpose_nodes': t_count,
                'mean_ms': stats['mean_ms'],
                'fps': stats['fps'],
            }

    save_results(all_results, 'exp07_ort_opt_transpose.json')
    print("\n实验 7 完成")


if __name__ == '__main__':
    run()
```

- [ ] **Step 3: 运行**

```bash
conda run -n py38 python3 analyze_transpose/exp07_ort_opt_transpose/run.py
```

Expected: 表格显示不同优化级别下节点数变化，判断 Transpose 是否被优化。

- [ ] **Step 4: Commit**

```bash
git add analyze_transpose/exp07_ort_opt_transpose/ analyze_transpose/results/exp07_ort_opt_transpose.json
git commit -m "feat: add exp07 ORT graph optimization effect on Transpose

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 10: 实验 8 — Transpose 与 Conv 融合边界

**Files:** Create `analyze_transpose/exp08_transpose_conv_fusion/run.py`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p analyze_transpose/exp08_transpose_conv_fusion
```

- [ ] **Step 2: 编写 run.py**

```python
"""实验 8: Transpose 与 Conv 融合边界 — 测试 ORT 是否融合相邻 Transpose+Conv。

构建 3 种微模型:
1. Conv only (baseline)
2. Transpose → Conv (紧邻，perm=[0,2,3,1])
3. Transpose → Conv → Transpose (Conv 被两个 Transpose 包夹)

用不同 ORT 优化级别运行，观察节点数变化判断是否融合。
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
from common.benchmark import print_stats, save_results

HERE = os.path.dirname(os.path.abspath(__file__))
WARMUP, ITERS = 10, 100

def make_simple_model(name, input_shape=(1, 64, 32, 32), nodes_fn=None):
    """创建精简测试 ONNX 模型。"""
    C_in, H, W = input_shape[1], input_shape[2], input_shape[3]
    C_out = 64
    inputs = [helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info('output', TensorProto.FLOAT, None)]

    weight = np.random.randn(C_out, C_in, 3, 3).astype(np.float32)
    bias = np.random.randn(C_out).astype(np.float32)

    init = [
        helper.make_tensor('w', TensorProto.FLOAT, weight.shape, weight.tobytes(), raw=True),
        helper.make_tensor('b', TensorProto.FLOAT, bias.shape, bias.tobytes(), raw=True),
    ]

    nodes = nodes_fn() if nodes_fn else [
        helper.make_node('Conv', ['input', 'w', 'b'], ['output'], name='conv1', kernel_shape=[3, 3]),
    ]

    graph = helper.make_graph(nodes, name, inputs, outputs, init)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 12)])
    return model


def run_benchmark(model, model_name, opt_level, opt_name):
    """运行单个模型基准。返回 stats dict。"""
    path = f'/tmp/exp08_{model_name}_{opt_name}.onnx'
    onnx.save(model, path)

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = opt_level
    sess = ort.InferenceSession(path, sess_opts, providers=['CPUExecutionProvider'])

    inp = np.random.randn(1, 64, 32, 32).astype(np.float32)

    for _ in range(WARMUP): sess.run(None, {'input': inp})
    times = []
    for _ in range(ITERS):
        t0 = time.time(); sess.run(None, {'input': inp}); times.append((time.time()-t0)*1000)
    times = np.array(times)

    # Count nodes in saved model
    loaded = onnx.load(path)
    t_count = sum(1 for n in loaded.graph.node if n.op_type == 'Transpose')
    conv_count = sum(1 for n in loaded.graph.node if n.op_type == 'Conv')

    return {
        'model': model_name, 'optimization': opt_name,
        'nodes_total': len(loaded.graph.node),
        'transpose_nodes': t_count, 'conv_nodes': conv_count,
        'mean_ms': float(times.mean()), 'fps': float(1000.0 / times.mean()),
    }


def build_nodes_conv_only():
    return [helper.make_node('Conv', ['input', 'w', 'b'], ['output'], name='conv1', kernel_shape=[3, 3])]

def build_nodes_transpose_conv():
    return [
        helper.make_node('Transpose', ['input'], ['t_out'], name='t1', perm=[0, 2, 3, 1]),
        helper.make_node('Transpose', ['t_out'], ['t2_out'], name='t2', perm=[0, 3, 1, 2]),
        helper.make_node('Conv', ['t2_out', 'w', 'b'], ['output'], name='conv1', kernel_shape=[3, 3]),
    ]

def build_nodes_transpose_conv_transpose():
    return [
        helper.make_node('Transpose', ['input'], ['t1_out'], name='t1', perm=[0, 2, 3, 1]),
        helper.make_node('Transpose', ['t1_out'], ['t2_out'], name='t2', perm=[0, 3, 1, 2]),
        helper.make_node('Conv', ['t2_out', 'w', 'b'], ['c_out'], name='conv1', kernel_shape=[3, 3]),
        helper.make_node('Transpose', ['c_out'], ['t3_out'], name='t3', perm=[0, 2, 3, 1]),
        helper.make_node('Transpose', ['t3_out'], ['output'], name='t4', perm=[0, 3, 1, 2]),
    ]


def run():
    print("=" * 60)
    print("实验 8: Transpose 与 Conv 融合边界")
    print("=" * 60)

    models = [
        ('Conv Only', build_nodes_conv_only),
        ('Transpose→Conv', build_nodes_transpose_conv),
        ('T→Conv→T', build_nodes_transpose_conv_transpose),
    ]

    opt_levels = [
        ('DISABLE_ALL', ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ('ALL', ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ]

    results = []

    for model_name, nodes_fn in models:
        model = make_simple_model(model_name.replace('→', '_'), nodes_fn=nodes_fn)
        for opt_name, opt_val in opt_levels:
            r = run_benchmark(model, model_name.replace('→', '_'), opt_val, opt_name)
            results.append(r)
            print(f"  {r['model']:<25} {r['optimization']:<14} "
                  f"nodes={r['nodes_total']:<4} T={r['transpose_nodes']} "
                  f"Conv={r['conv_nodes']} mean={r['mean_ms']:.3f}ms FPS={r['fps']:.1f}")

    all_results = {'experiment': 'exp08_transpose_conv_fusion', 'results': results}
    save_results(all_results, 'exp08_transpose_conv_fusion.json')

    # 分析融合
    print("\n--- 融合分析 ---")
    for model_name, _ in models:
        base = [r for r in results if r['model'] == model_name.replace('→','_') and r['optimization'] == 'DISABLE_ALL'][0]
        opt = [r for r in results if r['model'] == model_name.replace('→','_') and r['optimization'] == 'ALL'][0]
        nodes_delta = base['nodes_total'] - opt['nodes_total']
        print(f"  {model_name}: DISABLE_ALL→ALL 节点减少 {nodes_delta} "
              f"(T: {base['transpose_nodes']}→{opt['transpose_nodes']}, "
              f"延迟: {base['mean_ms']:.3f}→{opt['mean_ms']:.3f}ms)")
    print("\n实验 8 完成")
    print("  Hint: 用 Netron 打开修改后的模型可视觉确认融合效果。")


if __name__ == '__main__':
    run()
```

- [ ] **Step 3: 运行**

```bash
conda run -n py38 python3 analyze_transpose/exp08_transpose_conv_fusion/run.py
```

- [ ] **Step 4: Commit**

```bash
git add analyze_transpose/exp08_transpose_conv_fusion/ analyze_transpose/results/exp08_transpose_conv_fusion.json
git commit -m "feat: add exp08 transpose-conv fusion boundary experiment

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 11: 实验 9 — 不同输入分辨率

**Files:** Create `analyze_transpose/exp09_resolution/run.py`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p analyze_transpose/exp09_resolution
```

- [ ] **Step 2: 编写 run.py**

```python
"""实验 9: 不同输入分辨率下的 Transpose 开销。
测试 320x320, 640x640, 1280x1280 三种分辨率。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import onnxruntime as ort
from common.benchmark import (
    benchmark_ort, benchmark_model, print_stats, save_results,
    generate_dummy_input
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ORIG = os.path.join(HERE, '..', 'yolov8m.onnx')
MODEL_MOD = os.path.join(HERE, '..', 'modified_models', 'yolov8m_input_transpose.onnx')

RESOLUTIONS = [
    (320, 320),
    (640, 640),
    (1280, 1280),
]

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
            stats = benchmark_model(
                model_path, input_fn,
                graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                warmup=WARMUP, iters=ITERS
            )
            n_pixels = 3 * h * w
            ns_per_pixel = stats['mean_ms'] * 1e6 / n_pixels
            print(f"  {h}x{w:<10} {label:<12} {stats['mean_ms']:<12.2f} {stats['fps']:<10.2f} "
                  f"{n_pixels:<12} {ns_per_pixel:<12.3f}")

            all_results['results'].append({
                'resolution': f'{h}x{w}',
                'model': label,
                'mean_ms': stats['mean_ms'],
                'fps': stats['fps'],
                'pixels': n_pixels,
                'ns_per_pixel': ns_per_pixel,
            })

    # 分析缩放效率
    print("\n--- 分辨率缩放分析 ---")
    for label in ['Original', 'Modified']:
        entries = [r for r in all_results['results'] if r['model'] == label]
        if len(entries) >= 2:
            t320 = entries[0]['mean_ms']
            t1280 = entries[-1]['mean_ms']
            pixel_ratio = entries[-1]['pixels'] / entries[0]['pixels']
            time_ratio = t1280 / t320
            print(f"  {label}: 1280/320 pixel ratio={pixel_ratio:.1f}x, time ratio={time_ratio:.2f}x")

    save_results(all_results, 'exp09_resolution.json')
    print("\n实验 9 完成")


if __name__ == '__main__':
    run()
```

- [ ] **Step 3: 运行**

```bash
conda run -n py38 python3 analyze_transpose/exp09_resolution/run.py
```

- [ ] **Step 4: Commit**

```bash
git add analyze_transpose/exp09_resolution/ analyze_transpose/results/exp09_resolution.json
git commit -m "feat: add exp09 resolution scaling Transpose overhead

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 12: run_all.sh 汇总脚本

**Files:** Create `analyze_transpose/run_all.sh`

- [ ] **Step 1: 编写 run_all.sh**

```bash
#!/bin/bash
# 一键运行全部 9 个 Transpose 实验
set -e
cd "$(dirname "$0")"

echo "=============================================="
echo "Transpose 实验套件 — 全部 9 个实验"
echo "=============================================="

# 检查 conda 环境
if [[ "$CONDA_DEFAULT_ENV" != "py38" ]]; then
    echo "请先激活 py38 conda 环境: conda activate py38"
    exit 1
fi

# 1. 生成修改模型
echo ""
echo ">>> 第 1 步: 生成修改模型 <<<"
for exp in exp02_input_transpose exp03_output_transpose exp04_dual_transpose exp05_mid_transpose exp06_multi_transpose; do
    if [ -f "${exp}/modify_model.py" ]; then
        echo "--- ${exp} ---"
        python3 "${exp}/modify_model.py"
    fi
done

# 2. 运行所有实验
echo ""
echo ">>> 第 2 步: 运行实验 <<<"
for exp in exp01_baseline exp02_input_transpose exp03_output_transpose exp04_dual_transpose exp05_mid_transpose exp06_multi_transpose exp07_ort_opt_transpose exp08_transpose_conv_fusion exp09_resolution; do
    echo ""
    echo "=============================================="
    echo "  运行: ${exp}"
    echo "=============================================="
    python3 "${exp}/run.py"
done

echo ""
echo "=============================================="
echo "全部实验完成! 结果在 results/ 目录"
echo "=============================================="
ls -la results/
```

- [ ] **Step 2: 添加执行权限**

```bash
chmod +x analyze_transpose/run_all.sh
```

- [ ] **Step 3: Commit**

```bash
git add analyze_transpose/run_all.sh
git commit -m "feat: add run_all.sh for one-click experiment execution

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 13: 最终验证

- [ ] **Step 1: 运行全部实验**

```bash
conda activate py38
bash analyze_transpose/run_all.sh
```

- [ ] **Step 2: 检查所有 results JSON 文件存在**

```bash
ls -la analyze_transpose/results/*.json
```

Expected: 9 个 JSON 文件 (exp01 到 exp09)。

- [ ] **Step 3: 快速检查 JSON 内容有效性**

```bash
conda run -n py38 python3 -c "
import json, os
results_dir = 'analyze_transpose/results'
for f in sorted(os.listdir(results_dir)):
    if f.endswith('.json'):
        with open(os.path.join(results_dir, f)) as fp:
            data = json.load(fp)
            exp = data.get('experiment', 'N/A')
            print(f'  {f}: experiment={exp}, keys={list(data.keys())[:5]}...')
"
```

- [ ] **Step 4: Commit 最终结果**

```bash
git add analyze_transpose/results/
git commit -m "feat: add all experiment results

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## 文件清单

| 文件 | 职责 |
|------|------|
| `analyze_transpose/common/__init__.py` | 空文件，标记 Python 包 |
| `analyze_transpose/common/model_utils.py` | ONNX 图操作: 插入 Transpose, 重定向边, 保存 |
| `analyze_transpose/common/benchmark.py` | ORT 基准: 预热, 测量, 统计, 保存 JSON |
| `analyze_transpose/exp01_baseline/run.py` | Baseline 4 种优化级别 |
| `analyze_transpose/exp02_input_transpose/modify_model.py` | 输入 round-trip Transpose pair |
| `analyze_transpose/exp02_input_transpose/run.py` | 对比测试 |
| `analyze_transpose/exp03_output_transpose/modify_model.py` | 输出单次 Transpose |
| `analyze_transpose/exp03_output_transpose/run.py` | 对比测试 |
| `analyze_transpose/exp04_dual_transpose/modify_model.py` | 输入 pair + 输出 single |
| `analyze_transpose/exp04_dual_transpose/run.py` | 对比测试 |
| `analyze_transpose/exp05_mid_transpose/modify_model.py` | 第 20 个 Conv 后 pair |
| `analyze_transpose/exp05_mid_transpose/run.py` | 对比测试 |
| `analyze_transpose/exp06_multi_transpose/modify_model.py` | 4 个 backbone stage pair |
| `analyze_transpose/exp06_multi_transpose/run.py` | 对比测试 |
| `analyze_transpose/exp07_ort_opt_transpose/run.py` | ORT 优化级别对比 (原始+修改模型) |
| `analyze_transpose/exp08_transpose_conv_fusion/run.py` | 微模型 Transpose-Conv 融合测试 |
| `analyze_transpose/exp09_resolution/run.py` | 320/640/1280 分辨率对比 |
| `analyze_transpose/run_all.sh` | 一键运行全部实验 |
| `analyze_transpose/modified_models/` | 生成的修改版 ONNX (gitignore 建议) |
| `analyze_transpose/results/` | 9 个 JSON 结果文件 |

## 注意事项

1. **Round-trip 策略**: exp2/4/5/6 使用 Transpose pair 而非单个 Transpose，因为单个布局修改型 Transpose 会导致下游 Conv 的 weight 通道数不匹配，ORT 无法运行。每个 pair 的延迟增量 / 2 = 单个 Transpose 的开销估算。

2. **输出端例外**: exp3/4 的输出 Transpose 使用单次 perm=[0,2,1]（3D tensor），因为它不改变 channel 维度且是 graph 最后一个操作，无下游约束。

3. **exp8**: 使用独立微模型（64 通道 Conv + Transpose），避免 YOLOv8m 图优化器的干扰。

4. **所有模型输入**: 使用 `np.random.randn` 模拟图像，无摄像头依赖。
