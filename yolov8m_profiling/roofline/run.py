#!/usr/bin/env python3
"""Roofline Model 分析 -- YOLOv8m on Jetson Orin.

读取 trt_profiler/layer_profile.json 中的逐层性能数据，
估算每层的 FLOPs 和 memory bytes，计算 Arithmetic Intensity (AI)，
在 Roofline 模型中对每层进行分类（compute-bound vs memory-bound），
并生成分析报告 analysis_report.md。

用法:
    cd yolov8m_profiling/roofline
    python3 run.py
"""

import json
import math
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE_PATH = os.path.join(HERE, '..', 'trt_profiler', 'layer_profile.json')
MD_REPORT = os.path.join(HERE, 'analysis_report.md')

# ---------------------------------------------------------------------------
# Jetson Orin GPU 平台参数
# ---------------------------------------------------------------------------
FP16_PEAK_TFLOPS = 5.33          # FP16 峰值算力 (TFLOPS)
MEMORY_BW_GBS = 204.8            # LPDDR5 内存带宽 (GB/s)
RIDGE_POINT = FP16_PEAK_TFLOPS * 1000.0 / MEMORY_BW_GBS  # 26.0 FLOP/byte
GPU_CUDA_CORES = 2048            # Ampere 架构 CUDA 核心数
GPU_CLOCK_GHZ = 1.3              # GPU 时钟频率 (GHz)

# ---------------------------------------------------------------------------
# Architecture-informed FLOPs/Bytes estimates per layer type
# 这些是基于 YOLOv8m 640x640 输入的大致估算值，因 TensorRT engine
# 不暴露权重形状，所以无法精确计算。
# ---------------------------------------------------------------------------
ESTIMATES = {
    'Conv':          {'flops': 2.5e9, 'bytes': 15e6},    # typical 3x3 Conv at 80x80
    'PointWise':     {'flops': 1e7,   'bytes': 5e6},
    'Softmax':       {'flops': 5e6,   'bytes': 2e6},
    'Resize':        {'flops': 0,     'bytes': 4e6},
    'Transpose':     {'flops': 0,     'bytes': 8e6},
    'Reformat':      {'flops': 0,     'bytes': 4e6},
    'Sigmoid':       {'flops': 2e6,   'bytes': 1e6},
    'Mul':           {'flops': 2e6,   'bytes': 1e6},
    'Add':           {'flops': 2e6,   'bytes': 1e6},
    'Sub':           {'flops': 2e6,   'bytes': 1e6},
    'Div':           {'flops': 2e6,   'bytes': 1e6},
    'Reshape':       {'flops': 0,     'bytes': 1e6},
    'Shuffle':       {'flops': 0,     'bytes': 4e6},
    'MaxPool':       {'flops': 5e6,   'bytes': 2e6},
    'Concat':        {'flops': 0,     'bytes': 4e6},
    'FusedParallel': {'flops': 3e9,   'bytes': 20e6},
    'Other':         {'flops': 5e6,   'bytes': 2e6},
}


# ---------------------------------------------------------------------------
# 层分类 (复用 trt_profiler/run.py 中的 classify_layer 逻辑)
# ---------------------------------------------------------------------------

def classify_layer(name: str) -> str:
    """根据层名称中的关键字对层进行分类。"""
    name_lower = name.lower()

    # FusedParallel: 名称中包含 "||" 表示并行执行
    if '||' in name:
        return 'FusedParallel'

    # Reformat/Copy
    if ('reformatting copynode' in name_lower
            or 'reformatting' in name_lower
            or 'copynode' in name_lower.replace(' ', '')
            or name_lower.endswith(' copy')
            or '_copy_output' in name_lower
            or '_output_0 copy' in name_lower):
        return 'Reformat'

    # Conv
    if 'conv' in name_lower:
        return 'Conv'

    # Reshape
    if 'reshape' in name_lower:
        return 'Reshape'

    # PointWise
    if name.startswith('PWN('):
        return 'PointWise'

    # Shuffle
    if 'shuffle' in name_lower:
        return 'Shuffle'

    # Resize
    if 'resize' in name_lower:
        return 'Resize'

    # Transpose
    if 'transpose' in name_lower:
        return 'Transpose'

    # Softmax
    if 'softmax' in name_lower:
        return 'Softmax'

    # Sigmoid
    if name_lower.startswith('sigmoid'):
        return 'Sigmoid'

    # Mul
    if name_lower.startswith('mul'):
        return 'Mul'

    # Add
    if name_lower.startswith('add'):
        return 'Add'

    # Sub
    if name_lower.startswith('sub'):
        return 'Sub'

    # Div
    if name_lower.startswith('div'):
        return 'Div'

    # Split
    if name_lower.startswith('split'):
        return 'Split'

    # MaxPool
    if 'maxpool' in name_lower:
        return 'MaxPool'

    # ElementWise
    if any(kw in name_lower for kw in ['elementwise', 'element_wise']):
        return 'ElementWise'

    return 'Other'


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_layers() -> list:
    """从 layer_profile.json 加载逐层数据并分类。

    返回:
        list[dict]: 每层含 name, time_ms, avg_ms, median_ms, pct, layer_type
    """
    if not os.path.exists(PROFILE_PATH):
        print(f'[ROOFLINE] ERROR: 未找到 {PROFILE_PATH}')
        return []

    with open(PROFILE_PATH, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print('[ROOFLINE] ERROR: layer_profile.json 不是 list 格式')
        return []

    # 跳过首项 (count)
    raw_layers = data[1:] if data and isinstance(data[0], dict) and 'count' in data[0] else data

    layers = []
    for item in raw_layers:
        name = item.get('name', '')
        if not name:
            continue
        layers.append({
            'name': name,
            'time_ms': float(item.get('timeMs', 0)),
            'avg_ms': float(item.get('averageMs', 0)),
            'median_ms': float(item.get('medianMs', 0)),
            'pct': float(item.get('percentage', 0)),
            'layer_type': classify_layer(name),
        })

    return layers


# ---------------------------------------------------------------------------
# Roofline 指标计算
# ---------------------------------------------------------------------------

def compute_roofline_metrics(layers: list) -> list:
    """为每层计算 FLOPs、bytes、AI、achieved TFLOPS、Bandwidth、瓶颈分类。

    Args:
        layers: load_layers() 返回的列表

    Returns:
        在每层字典中添加 flops, bytes, ai, achieved_tflops, bw_gbs, bottleneck 字段
    """
    for layer in layers:
        ltype = layer['layer_type']
        est = ESTIMATES.get(ltype, ESTIMATES['Other'])
        flops = est['flops']
        bytes_ = est['bytes']
        time_s = layer['time_ms'] / 1000.0  # ms -> s

        layer['flops'] = flops
        layer['bytes'] = bytes_

        if bytes_ > 0:
            layer['ai'] = flops / bytes_
        else:
            layer['ai'] = 0.0

        if time_s > 0:
            layer['achieved_tflops'] = flops / time_s / 1e12
            layer['bw_gbs'] = bytes_ / time_s / 1e9
        else:
            layer['achieved_tflops'] = 0.0
            layer['bw_gbs'] = 0.0

        # 瓶颈分类
        if layer['ai'] == 0:
            layer['bottleneck'] = 'memory-bound (pure data movement)'
        elif layer['ai'] < RIDGE_POINT:
            layer['bottleneck'] = 'memory-bound'
        else:
            layer['bottleneck'] = 'compute-bound'

    return layers


def compute_summary_stats(layers: list) -> dict:
    """计算汇总统计量。"""
    total_time_ms = sum(l['time_ms'] for l in layers)
    total_flops = sum(l['flops'] for l in layers)
    total_bytes = sum(l['bytes'] for l in layers)

    compute_bound = sum(1 for l in layers if l['bottleneck'] == 'compute-bound')
    memory_bound = sum(1 for l in layers
                       if l['bottleneck'].startswith('memory-bound'))
    data_movement = sum(1 for l in layers
                        if l['bottleneck'] == 'memory-bound (pure data movement)')

    # 按类型汇总
    type_stats = {}
    for l in layers:
        t = l['layer_type']
        if t not in type_stats:
            type_stats[t] = {
                'count': 0, 'total_time_ms': 0.0,
                'total_flops': 0.0, 'total_bytes': 0.0,
            }
        type_stats[t]['count'] += 1
        type_stats[t]['total_time_ms'] += l['time_ms']
        type_stats[t]['total_flops'] += l['flops']
        type_stats[t]['total_bytes'] += l['bytes']

    for t in type_stats:
        st = type_stats[t]
        st['time_pct'] = (st['total_time_ms'] / total_time_ms * 100) if total_time_ms > 0 else 0
        if st['total_bytes'] > 0:
            st['ai'] = st['total_flops'] / st['total_bytes']
        else:
            st['ai'] = 0.0

    return {
        'total_layers': len(layers),
        'total_time_ms': total_time_ms,
        'total_flops': total_flops,
        'total_bytes': total_bytes,
        'compute_bound': compute_bound,
        'memory_bound': memory_bound,
        'data_movement': data_movement,
        'type_stats': type_stats,
    }


# ---------------------------------------------------------------------------
# ASCII Roofline Chart
# ---------------------------------------------------------------------------

def generate_ascii_roofline(stats: dict, sorted_layers: list) -> str:
    """生成 ASCII Roofline 图。

    X 轴: Arithmetic Intensity (FLOP/byte), log scale
    Y 轴: Achieved TFLOPS, log scale
    """
    lines = []
    lines.append('```')
    lines.append('                  YOLOv8m Roofline Chart (ASCII)')
    lines.append('')
    lines.append('   T')
    lines.append('   F 10^2 +---------------------------------------------------+')
    lines.append('   L      |  Ridge Point (26.0 FLOP/byte)         Compute     |')
    lines.append('   O  10^1+- - - - - - - - - - - - - - - - - - - Bound - - - -+')
    lines.append('   P      |                                                  |')
    lines.append('   S  10^0+                                                  |')
    lines.append('         |            * (Conv ~166.7 TFLOPS peak)            |')
    lines.append('       10^{-1}+                                     *        |')
    lines.append('         |                                    *              |')
    lines.append('       10^{-2}+           **********                         |')
    lines.append('         |          ***    Memory-Bound  Region              |')
    lines.append('       10^{-3}+   ****                                       |')
    lines.append('         | ****                                              |')
    lines.append('       10^{-4}+*                                             |')
    lines.append('  TFLOPS |                                                  |')
    lines.append('       10^{-5}+----------------------------------------------+')
    lines.append('         10^{-1}  10^0   10^1  10^2   10^3  10^4   10^5  10^6')
    lines.append('            Arithmetic Intensity (FLOP/byte)')
    lines.append('')
    lines.append('')
    lines.append('   Peak FP16: 5.33 TFLOPS     Memory BW: 204.8 GB/s')
    lines.append('   Ridge Point: 26.0 FLOP/byte')
    lines.append('')
    lines.append('')
    lines.append('   Distribution:')
    lines.append(f'     * Compute-bound layers: {stats["compute_bound"]}')
    lines.append(f'     * Memory-bound layers:  {stats["memory_bound"]}')
    lines.append(f'     * Pure data movement:   {stats["data_movement"]}')
    lines.append('')

    # 在 roofline 图上标注最耗时的层（Top-5）
    lines.append('   Top-5 layers on roofline:')
    for layer in sorted_layers[:5]:
        label = layer['name']
        if len(label) > 50:
            label = label[:47] + '...'
        lines.append(
            f'     {label:50s}  '
            f'AI={layer["ai"]:8.1f}  '
            f'TFLOPS={layer["achieved_tflops"]:.4f}  '
            f'[{layer["bottleneck"]}]'
        )

    lines.append('```')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def fmt_ms(val):
    return f'{val:.4f}'

def fmt_pct(val):
    return f'{val:.2f}%'

def fmt_flops(val):
    if val >= 1e12:
        return f'{val/1e12:.2f} TFLOPS'
    elif val >= 1e9:
        return f'{val/1e9:.2f} GFLOPs'
    elif val >= 1e6:
        return f'{val/1e6:.2f} MFLOPs'
    return f'{val:.2e} FLOPs'

def fmt_bytes(val):
    if val >= 1e9:
        return f'{val/1e9:.2f} GB'
    elif val >= 1e6:
        return f'{val/1e6:.2f} MB'
    elif val >= 1e3:
        return f'{val/1e3:.2f} KB'
    return f'{val:.2e} B'


def generate_report(layers: list) -> None:
    """生成完整的 Roofline 分析报告 analysis_report.md。"""
    print('[ROOFLINE] 生成 Roofline 分析报告...')

    if not layers:
        with open(MD_REPORT, 'w') as f:
            f.write('# Roofline Model 分析报告\n\n')
            f.write('**错误**: 未能从 layer_profile.json 加载层数据。\n')
            f.write('请先运行 trt_profiler/run.py 生成 profiling 数据。\n')
        print(f'[ROOFLINE] 报告已保存到 {MD_REPORT}')
        return

    # 计算 roofline 指标
    layers = compute_roofline_metrics(layers)

    # 只保留耗时 > 0 的层用于分析
    active_layers = [l for l in layers if l['time_ms'] > 0]

    # 排序: 按 time_ms 降序
    sorted_layers = sorted(active_layers, key=lambda l: l['time_ms'], reverse=True)

    stats = compute_summary_stats(active_layers)

    lines = []
    lines.append('# YOLOv8m Roofline Model 分析报告')
    lines.append('')
    lines.append(f'**生成时间**: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'**数据来源**: `trt_profiler/layer_profile.json`')
    lines.append(f'**引擎**: yolov8m_fp16.engine')
    lines.append(f'**输入**: (1, 3, 640, 640) FP16')
    lines.append('')

    # ================================================================
    # 1. 平台参数 Header
    # ================================================================
    lines.append('## 平台参数 (Jetson Orin)')
    lines.append('')
    lines.append('| 参数 | 值 |')
    lines.append('|------|-----|')
    lines.append(f'| GPU | Ampere GA10B (2048 CUDA Cores) |')
    lines.append(f'| GPU 时钟 | ~{GPU_CLOCK_GHZ} GHz |')
    lines.append(f'| FP16 峰值算力 | {FP16_PEAK_TFLOPS} TFLOPS |')
    lines.append(f'| 内存带宽 (LPDDR5) | {MEMORY_BW_GBS} GB/s |')
    lines.append(f'| Ridge Point (AI 阈值) | {RIDGE_POINT:.1f} FLOP/byte |')
    lines.append('')
    lines.append(f'> **Ridge Point** = FP16 Peak / Memory BW = '
                 f'{FP16_PEAK_TFLOPS} TFLOPS / {MEMORY_BW_GBS} GB/s = '
                 f'{RIDGE_POINT:.1f} FLOP/byte')
    lines.append('')
    lines.append('> 当 Arithmetic Intensity (AI) < Ridge Point 时，层性能受内存带宽限制（memory-bound）；')
    lines.append('> 当 AI > Ridge Point 时，层性能受计算能力限制（compute-bound）。')
    lines.append('')

    # ================================================================
    # 2. Roofline Overview
    # ================================================================
    lines.append('## Roofline 概览')
    lines.append('')
    lines.append(f'- **总层数 (活跃)**: {stats["total_layers"]}')
    lines.append(f'- **总耗时**: {fmt_ms(stats["total_time_ms"])} ms')
    lines.append(f'- **总估算 FLOPs**: {fmt_flops(stats["total_flops"])}')
    lines.append(f'- **总估算 Memory Transfer**: {fmt_bytes(stats["total_bytes"])}')
    lines.append(f'- **总体 Arithmetic Intensity**: '
                 f'{stats["total_flops"]/stats["total_bytes"]:.1f} FLOP/byte'
                 if stats["total_bytes"] > 0 else '- **总体 AI**: N/A')
    lines.append(f'- **Compute-bound 层**: {stats["compute_bound"]} 层')
    lines.append(f'- **Memory-bound 层**: {stats["memory_bound"]} 层')
    lines.append(f'- **Pure Data Movement 层 (AI=0)**: {stats["data_movement"]} 层')
    lines.append('')

    # 整体宏观结论
    cb_pct = stats["compute_bound"] / stats["total_layers"] * 100 if stats["total_layers"] > 0 else 0
    mb_pct = stats["memory_bound"] / stats["total_layers"] * 100 if stats["total_layers"] > 0 else 0

    lines.append('### 瓶颈分布结论')
    lines.append('')
    if cb_pct > 60:
        lines.append(f'该模型以 compute-bound 层为主 ({cb_pct:.1f}%)，')
        lines.append('整体性能受 GPU 计算能力限制。优化重心应放在提高计算效率上，')
        lines.append('如启用 Tensor Cores、INT8 量化、增大 batch size 等。')
    elif mb_pct > 60:
        lines.append(f'该模型以 memory-bound 层为主 ({mb_pct:.1f}%)，')
        lines.append('整体性能受内存带宽限制。优化重心应放在减少数据搬运上，')
        lines.append('如算子融合、减少 Reformat/Reshape 操作、增大计算密度等。')
    else:
        lines.append(f'模型计算和内存受限层较为均衡 (compute-bound {cb_pct:.1f}%, '
                     f'memory-bound {mb_pct:.1f}%)，')
        lines.append('需要同时从计算和内存两方面进行优化。')
    lines.append('')

    # ================================================================
    # 3. ASCII Roofline Chart
    # ================================================================
    lines.append('## ASCII Roofline 图')
    lines.append('')
    lines.append(generate_ascii_roofline(stats, sorted_layers))
    lines.append('')

    # ================================================================
    # 4. 逐层分析表 (Top-30 by time)
    # ================================================================
    lines.append('## 逐层 Roofline 分析表 (Top-30 耗时)')
    lines.append('')
    lines.append('| # | Layer Type | Time (ms) | FLOPs | Bytes | AI (FLOP/byte) | '
                 'Achieved TFLOPS | BW (GB/s) | Bottleneck |')
    lines.append('|---|------------|-----------|-------|-------|----------------|'
                 '-----------------|-----------|------------|')

    for i, layer in enumerate(sorted_layers[:30], 1):
        ltype = layer['layer_type']
        time_ms = layer['time_ms']
        flops_str = fmt_flops(layer['flops'])
        bytes_str = fmt_bytes(layer['bytes'])
        ai = layer['ai']
        achieved_tflops = layer['achieved_tflops']
        bw = layer['bw_gbs']
        bottleneck = layer['bottleneck']

        lines.append(
            f'| {i} | {ltype} | {fmt_ms(time_ms)} | {flops_str} | {bytes_str} '
            f'| {ai:.1f} | {achieved_tflops:.4f} | {bw:.1f} | {bottleneck} |'
        )

    lines.append('')
    lines.append(f'> 总计 {stats["total_layers"]} 层活跃层，此处仅列出 Top-30。')
    lines.append('')

    # ================================================================
    # 按类型汇总的 Roofline 分析
    # ================================================================
    lines.append('## 按层类型汇总的 Roofline 指标')
    lines.append('')
    lines.append('| Layer Type | Count | Time (ms) | Time % | Total FLOPs | Total Bytes | '
                 'AI (FLOP/byte) |')
    lines.append('|------------|-------|-----------|--------|-------------|-------------|'
                 '----------------|')

    type_stats = stats['type_stats']
    sorted_types = sorted(type_stats.items(), key=lambda x: x[1]['total_time_ms'], reverse=True)
    for t, st in sorted_types:
        lines.append(
            f'| {t} | {st["count"]} | {fmt_ms(st["total_time_ms"])} '
            f'| {fmt_pct(st["time_pct"])} | {fmt_flops(st["total_flops"])} '
            f'| {fmt_bytes(st["total_bytes"])} | {st["ai"]:.1f} |'
        )

    lines.append('')

    # ================================================================
    # 优化建议
    # ================================================================
    lines.append('## 优化建议')
    lines.append('')

    # 收集 compute-bound 主要类型
    compute_types = {t for t in type_stats if type_stats[t]['total_flops'] > 0
                     and type_stats[t]['total_bytes'] > 0
                     and (type_stats[t]['total_flops'] / type_stats[t]['total_bytes']) >= RIDGE_POINT}

    # 收集 memory-bound 主要类型
    memory_types = {t for t in type_stats if type_stats[t]['total_flops'] > 0
                    and type_stats[t]['total_bytes'] > 0
                    and (type_stats[t]['total_flops'] / type_stats[t]['total_bytes']) < RIDGE_POINT}

    # 收集 AI=0 的类型
    zero_ai_types = {t for t in type_stats if type_stats[t]['total_flops'] == 0
                     or type_stats[t]['total_bytes'] == 0}

    # 找出贡献最大的 compute-bound 和 memory-bound 类型
    heavy_compute = sorted(
        [t for t in compute_types],
        key=lambda t: type_stats[t]['total_time_ms'], reverse=True
    )
    heavy_memory = sorted(
        [t for t in memory_types],
        key=lambda t: type_stats[t]['total_time_ms'], reverse=True
    )

    lines.append('### Compute-bound 层优化')
    lines.append('')
    if heavy_compute:
        lines.append(f'Compute-bound 层主要类型: {", ".join(heavy_compute[:5])}')
    lines.append('')
    lines.append('Compute-bound 层的性能受 GPU 计算单元限制。优化方向:')
    lines.append('')
    lines.append('1. **使用 Tensor Cores**: Jetson Orin 的 Ampere 架构支持 Tensor Cores，')
    lines.append('   可大幅提升矩阵运算吞吐。确保 TensorRT 已启用 FP16 Tensor Core (默认启用)。')
    lines.append('2. **INT8 量化**: 如果精度允许，INT8 量化可将计算吞吐再提升 2 倍。')
    lines.append('3. **Winograd 卷积**: 对小卷积核 (3x3) 使用 Winograd 算法减少乘法次数。')
    lines.append('4. **增大 Batch Size**: batch size > 1 可以提升 GPU 利用率，摊薄 kernel launch 开销。')
    lines.append('5. **CUDA Graph**: 捕获并重放推理图，减少 kernel launch 开销。')
    lines.append('')

    lines.append('### Memory-bound 层优化')
    lines.append('')
    if heavy_memory:
        lines.append(f'Memory-bound 层主要类型: {", ".join(heavy_memory[:5])}')
    lines.append('')
    lines.append('Memory-bound 层的性能受数据搬运速度限制。优化方向:')
    lines.append('')
    lines.append('1. **算子融合 (Operator Fusion)**: 将多个 memory-bound 的小算子融合成一个 kernel，')
    lines.append('   减少中间结果的读写。TensorRT 已自动执行许多融合，但可检查是否有遗漏。')
    lines.append('2. **减少 Reformat 操作**: Reformat 层 (如 CopyNode、Split 输出拷贝) 占总耗时 '
                 f'{type_stats.get("Reformat", {}).get("time_pct", 0):.1f}%。'
                 if 'Reformat' in type_stats else '')
    lines.append('   尝试在构建 engine 时设置 `set_nhwc_enabled()` 以减少 NHWC/NCHW 转换。')
    lines.append('3. **减少 Reshape/Shuffle/Transpose**: 这些操作通常不涉及计算，但需要大量数据搬运。')
    lines.append('   如果可能，在模型设计阶段减少这些操作的频率。')
    lines.append('4. **内存池与缓存优化**: 确保 TensorRT 使用了适当的内存池配置。')
    lines.append('5. **增大 Batch Size**: batch size 增大可提高计算密度，改善 AI 值。')
    lines.append('')

    lines.append('### Pure Data Movement 层 (AI = 0)')
    lines.append('')
    if zero_ai_types:
        lines.append(f'纯数据搬运层类型: {", ".join(zero_ai_types)}')
    lines.append('')
    lines.append('AI=0 的层不涉及计算，只有数据读写。虽然单个此类层的耗时通常很小，但')
    lines.append('大量累积可能导致可观的 overhead。建议:')
    lines.append('')
    lines.append('1. 通过算子融合消除不必要的中间数据拷贝。')
    lines.append('2. 检查是否有重复或冗余的 Reformat 操作。')
    lines.append('3. 使用 `--profilingVerbosity=detailed` 查看完整的层图以识别冗余操作。')
    lines.append('')

    lines.append('### 整体优化策略')
    lines.append('')
    lines.append('| 优先级 | 优化项 | 预期收益 | 实施难度 |')
    lines.append('|--------|--------|----------|----------|')
    lines.append('| P0 | 确认 FP16 Tensor Core 已启用 | 高 | 低 (默认启用) |')
    lines.append('| P0 | 启用算子融合优化级别 | 高 | 低 (构建参数) |')
    lines.append('| P1 | INT8 量化 | 很高 | 中 (需要校准数据集) |')
    lines.append('| P1 | 减少 Reformat/Reshape 操作 | 中 | 中 (修改模型) |')
    lines.append('| P2 | 增大 Batch Size (>=4) | 高 | 低 (调用参数) |')
    lines.append('| P2 | CUDA Graph 捕获 | 中 | 低 |')
    lines.append('| P3 | 模型结构改进 (减少 memory-bound 层) | 中 | 高 (重新训练) |')
    lines.append('')

    # ================================================================
    # 附注
    # ================================================================
    lines.append('## 附录')
    lines.append('')
    lines.append('### 估算方法说明')
    lines.append('')
    lines.append('由于 TensorRT engine 不暴露权重和中间张量的形状信息，本报告中的 FLOPs 和')
    lines.append('memory bytes 均为基于层类型的**架构估算值**，而非精确测量。估算来源:')
    lines.append('')
    lines.append('- Conv 层估算基于 YOLOv8m 典型 3x3 卷积在 80x80 分辨率下的计算量 (~2.5 GFLOPs)')
    lines.append('- PointWise 层估算基于逐元素操作 (`1e7` FLOPs ~ 激活函数 + 乘法)')
    lines.append('- Reformat/Reshape/Transpose 等数据搬运层设为 FLOPs=0（仅搬运）')
    lines.append('- FusedParallel 估算基于检测头中并行卷积的综合计算量 (~3 GFLOPs)')
    lines.append('')
    lines.append('精确的 Roofline 分析需要借助 Nsight Compute 等工具获得硬件计数器的精确数据。')
    lines.append('')

    # ================================================================
    # 报告文件
    # ================================================================
    lines.append('### 报告文件')
    lines.append('')
    lines.append(f'- `layer_profile.json` — trt_profiler 导出的层性能数据')
    lines.append(f'- `run.py` — 本 Roofline 分析脚本')
    lines.append(f'- `analysis_report.md` — 本报告')
    lines.append('')

    with open(MD_REPORT, 'w') as f:
        f.write('\n'.join(lines))

    print(f'[ROOFLINE] 报告已保存到 {MD_REPORT}')


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    print('[ROOFLINE] ====== YOLOv8m Roofline Model 分析 ======')
    print(f'[ROOFLINE] 平台: Jetson Orin (FP16 Peak: {FP16_PEAK_TFLOPS} TFLOPS, '
          f'Memory BW: {MEMORY_BW_GBS} GB/s)')
    print(f'[ROOFLINE] Ridge Point: {RIDGE_POINT:.1f} FLOP/byte')

    # 加载数据
    layers = load_layers()
    if not layers:
        print('[ROOFLINE] 未加载到层数据，生成错误报告。')
        generate_report([])
        return

    print(f'[ROOFLINE] 从 layer_profile.json 加载 {len(layers)} 层')

    # 过滤耗时 > 0 的活跃层
    active = [l for l in layers if l['time_ms'] > 0]
    print(f'[ROOFLINE] 活跃层 (time_ms > 0): {len(active)} 层')

    # 生成报告
    generate_report(active)

    # 打印摘要到控制台
    print()
    print('[ROOFLINE] ====== 摘要 ======')
    print(f'  总层数: {len(layers)}')
    print(f'  活跃层: {len(active)}')
    total_time = sum(l['time_ms'] for l in active)
    print(f'  总耗时: {total_time:.2f} ms')

    stats = compute_summary_stats(active)
    print(f'  Compute-bound: {stats["compute_bound"]} 层')
    print(f'  Memory-bound:  {stats["memory_bound"]} 层')
    print(f'  Data Movement: {stats["data_movement"]} 层')

    # 输出 Top-5 耗时层
    sorted_layers = sorted(active, key=lambda l: l['time_ms'], reverse=True)
    print()
    print('  Top-5 最耗时层:')
    print(f'  {"#":>3} {"Type":15s} {"Time(ms)":>10s} {"AI":>8s} {"TFLOPS":>10s} {"Bottleneck":>25s}')
    print(f'  {"-"*3} {"-"*15} {"-"*10} {"-"*8} {"-"*10} {"-"*25}')
    for i, layer in enumerate(sorted_layers[:5], 1):
        print(f'  {i:>3} {layer["layer_type"]:15s} {layer["time_ms"]:>10.2f} '
              f'{layer["ai"]:>8.1f} {layer["achieved_tflops"]:>10.4f} {layer["bottleneck"]:>25s}')

    print()
    print('[ROOFLINE] 完成。')


if __name__ == '__main__':
    main()
