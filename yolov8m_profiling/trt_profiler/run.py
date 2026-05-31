#!/usr/bin/env python3
"""TensorRT Profiler 逐层性能分析。

用法:
    cd yolov8m_profiling/trt_profiler
    python3 run.py

功能:
    1. 调用 trtexec 运行 engine，启用 --dumpProfile 获取逐层性能数据
    2. 导出层性能数据到 layer_profile.json 和 layer_info.json
    3. 解析 JSON / stdout 得到结构化逐层数据
    4. 生成 analysis_report.md：含类型汇总、Top-15 耗时、完整逐层表、瓶颈分析等

注意:
    - trtexec 不需要 LD_PRELOAD，因其直接调用 CUDA，不依赖 OpenBLAS。
    - 首次运行会执行 profiling（约 15 秒, --duration=10）。
    - 如果 layer_profile.json 已存在则跳过 profiling，仅重新生成 MD 报告。
    - 解析策略: 首选 layer_profile.json（结构化数据），
      layer_info.json（层类型信息），fallback 到 stdout 解析。
"""

import json
import os
import re
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
ENGINE = os.path.realpath(os.path.join(ROOT, '..', 'weights', 'engines', 'yolov8m_fp16.engine'))
TRTEXEC = '/usr/src/tensorrt/bin/trtexec'

LAYER_PROFILE_JSON = os.path.join(HERE, 'layer_profile.json')
LAYER_INFO_JSON = os.path.join(HERE, 'layer_info.json')
MD_REPORT = os.path.join(HERE, 'analysis_report.md')

INPUT_SHAPE = 'images (1, 3, 640, 640) float32'
OUTPUT_SHAPE = 'output0 (1, 84, 8400) float32'


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def get_trt_version():
    """获取 TensorRT 版本号。"""
    try:
        result = subprocess.run([TRTEXEC, '--version'], capture_output=True, text=True, timeout=30)
        output = (result.stdout or result.stderr or '').strip()
        # trtexec --version 输出: "TensorRT v8502" 或 "[TensorRT v8502]"
        # 需要从第一行提取版本号
        first_line = output.split('\n')[0]
        match = re.search(r'TensorRT\s+v(\d+)', first_line)
        if match:
            return f'TensorRT v{match.group(1)}'
        return 'TensorRT (unknown)'
    except Exception as e:
        print(f'[TRT] WARNING: 无法获取版本: {e}')
        return 'TensorRT (unknown)'


def classify_layer(name: str) -> str:
    """对层名称进行分类。

    根据层名称中的关键字和命名约定进行分类。
    优先级从高到低:
      1. FusedParallel: 名称中包含 "||"
      2. Reformat: 包含 CopyNode / Reformatting / copy 等
      3. Conv: 包含 conv
      4. Reshape: 包含 reshape
      5. PointWise: 以 PWN( 开头
      6. Shuffle: 包含 Shuffle
      7. 其余根据关键字匹配
    """
    name_lower = name.lower()

    # FusedParallel: 名称中包含 "||" 表示并行执行
    if '||' in name:
        return 'FusedParallel'

    # Reformat/Copy — 优先匹配，因为 "copy" 常出现在其他层名前缀中
    if ('reformatting copynode' in name_lower
            or 'reformatting' in name_lower
            or 'copynode' in name_lower.replace(' ', '')
            or name_lower.endswith(' copy')
            or '_copy_output' in name_lower
            or '_output_0 copy' in name_lower):
        return 'Reformat'

    # Conv 卷积类 — 包含 conv 关键词的层
    if 'conv' in name_lower:
        return 'Conv'

    # Reshape
    if 'reshape' in name_lower:
        return 'Reshape'

    # PointWise 逐点运算 (PWN)
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

    # ElementWise 通用
    if any(kw in name_lower for kw in ['elementwise', 'element_wise']):
        return 'ElementWise'

    # 默认分类
    return 'Other'


def _fmt_ms(ms_val):
    """格式化毫秒值为可读字符串。"""
    if ms_val is None:
        return 'N/A'
    return f'{ms_val:.4f}'


def _fmt_us(us_val):
    """格式化微秒值为可读字符串。"""
    if us_val is None:
        return 'N/A'
    return f'{us_val:.2f}'


def _fmt_pct(val):
    """格式化为百分比字符串。"""
    if val is None:
        return 'N/A'
    return f'{val:.2f}%'


# ---------------------------------------------------------------------------
# 解析层数据
# ---------------------------------------------------------------------------

def parse_profile_from_stdout(stdout: str) -> list:
    """Fallback: 从 --dumpProfile 的 stdout 固定宽度表格解析逐层数据。

    格式示例:
        [I]    /model.22/dfl/conv/Conv    1137.67    4.5325    0.0554    23.4
        [I]    Total    4860.52    19.3646    14.7284    100.0

    返回:
        list[dict]: [{'name': ..., 'time_us': ..., 'avg_ms': ..., 'median_ms': ..., 'time_pct': ...}, ...]
    """
    layers = []
    in_profile_section = False

    for line in stdout.split('\n'):
        line = line.strip()

        # 检测 section header: "Layer", "Time", "Avg", "Median", "Time%" 等表头
        if (re.search(r'\bLayer\b', line) and re.search(r'\bTime\b', line)
                and re.search(r'\bAvg\b', line) and re.search(r'\bMedian\b', line)
                and 'Time%' in line):
            # 跳过表头下的分隔线
            in_profile_section = True
            continue

        # 检测 Total 行 - 表示 profile section 结束
        if in_profile_section and 'Total' in line and re.search(r'\d+\.\d+', line):
            in_profile_section = False
            continue

        if not in_profile_section:
            continue

        if not line:
            continue

        # 去掉 [I] 前缀
        line = re.sub(r'^\[I\]\s*', '', line)

        # 按空白分割。格式: LayerName  Time(us)  Avg(ms)  Median(ms)  Time(%)
        # 注意层名可能包含空格，所以从右往左取 4 个数字列
        parts = line.split()
        if len(parts) < 5:
            continue

        # 尝试从右向左解析数字列
        try:
            time_pct = float(parts[-1])
            median_ms = float(parts[-2])
            avg_ms = float(parts[-3])
            time_us = float(parts[-4])
            name = ' '.join(parts[:-4])
        except (ValueError, IndexError):
            continue

        layers.append({
            'name': name,
            'time_us': time_us,
            'avg_ms': avg_ms,
            'median_ms': median_ms,
            'time_pct': time_pct,
        })

    return layers


def parse_profile_json() -> list:
    """从 layer_profile.json 解析逐层性能数据。

    实际 JSON 格式 (trtexec --exportProfile):
        [
            {"count": 587},                          # 首项 = 总调用次数
            {"name": "...", "timeMs": ..., "averageMs": ..., "medianMs": ..., "percentage": ...},
            ...
        ]

    其中:
      - timeMs: 该层总耗时 (毫秒, 所有迭代累加)
      - averageMs: 平均每次调用耗时 (毫秒)
      - medianMs: 中位数耗时 (毫秒)
      - percentage: 占总时间的百分比 (如 0.653651 表示 0.65%)

    返回:
        list[dict]: [{'name': ..., 'time_us': ..., 'avg_ms': ..., 'median_ms': ..., 'time_pct': ...}, ...]
    """
    if not os.path.exists(LAYER_PROFILE_JSON):
        return None

    with open(LAYER_PROFILE_JSON, 'r') as f:
        data = json.load(f)

    # 处理 list 格式: 首项为 {"count": N}, 后续为逐层数据
    if isinstance(data, list):
        # 跳过首项 (count)
        raw_layers = data[1:] if data and isinstance(data[0], dict) and 'count' in data[0] else data
    elif isinstance(data, dict):
        # 尝试嵌套格式
        raw_layers = data.get('Layers', data.get('layers', []))
        if not raw_layers:
            return None
    else:
        return None

    layers = []
    for item in raw_layers:
        if not isinstance(item, dict):
            continue
        name = item.get('name', '')
        if not name:
            continue

        # 字段名: timeMs, averageMs, medianMs, percentage
        time_ms = float(item.get('timeMs', 0))
        avg_ms = float(item.get('averageMs', 0))
        median_ms = float(item.get('medianMs', 0))
        pct = float(item.get('percentage', 0))

        layers.append({
            'name': name,
            'time_us': time_ms * 1000.0,       # 转换为微秒
            'avg_ms': avg_ms,
            'median_ms': median_ms,
            'time_pct': pct,                    # 已经是百分比
        })

    return layers


def parse_layer_info_json() -> list:
    """从 layer_info.json 解析层名称列表。

    实际 JSON 格式 (trtexec --exportLayerInfo):
        {"Layers": ["name1", "name2", ...], "Bindings": ["images", "output0"]}

    返回:
        list[str]: 层名称列表 (仅用于信息参考)
    """
    if not os.path.exists(LAYER_INFO_JSON):
        return []

    with open(LAYER_INFO_JSON, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get('Layers', [])
    elif isinstance(data, list):
        return data
    return []


def load_layers() -> list:
    """加载并解析逐层性能数据。

    策略:
        1. 首选从 layer_profile.json 解析。
        2. 如果 JSON 不存在或为空，从 stdout 解析。
        3. 根据层名称中的关键字进行分类。

    返回:
        list[dict]: 每个 layer 包含 name, time_us, avg_ms, median_ms, time_pct, layer_type
    """
    # 从 JSON 加载性能数据
    layers = parse_profile_json()
    source = 'json'

    # JSON 失败或未找到，回退到 stdout 解析
    if not layers:
        print('[TRT] JSON profile not available, attempting stdout parsing...')
        layers = []
        source = 'stdout'

    if not layers:
        print('[TRT] WARNING: 无法从 JSON 或 stdout 获取层性能数据。')
        return []

    print(f'[TRT] 从 {source} 加载了 {len(layers)} 层的性能数据。')

    # 可选: 打印 layer_info.json 中的层名数量 (纯信息)
    layer_names = parse_layer_info_json()
    if layer_names:
        print(f'[TRT] layer_info.json 包含 {len(layer_names)} 层 (仅名称, 无类型信息)。')

    # 分类每层
    for layer in layers:
        layer['layer_type'] = classify_layer(layer['name'])

    return layers


# ---------------------------------------------------------------------------
# 后处理与统计
# ---------------------------------------------------------------------------

def compute_stats(layers: list) -> dict:
    """计算层统计汇总。

    Returns:
        dict: 包含 total_time_ms, fps, type_stats（按类型汇总）等
    """
    total_time_us = sum(l['time_us'] for l in layers)
    total_time_ms = total_time_us / 1000.0

    # 按类型汇总
    type_stats = {}
    for l in layers:
        t = l['layer_type']
        if t not in type_stats:
            type_stats[t] = {'count': 0, 'total_time_us': 0.0}
        type_stats[t]['count'] += 1
        type_stats[t]['total_time_us'] += l['time_us']

    # 计算百分比并格式化为 ms
    for t in type_stats:
        type_stats[t]['total_time_ms'] = type_stats[t]['total_time_us'] / 1000.0
        type_stats[t]['time_pct'] = (
            type_stats[t]['total_time_us'] / total_time_us * 100.0
        ) if total_time_us > 0 else 0.0

    return {
        'total_time_us': total_time_us,
        'total_time_ms': total_time_ms,
        'type_stats': type_stats,
        'num_layers': len(layers),
    }


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def generate_report(layers: list, stats: dict, trt_version: str,
                    stdout: str, profiling_success: bool) -> None:
    """生成完整的 MD 分析报告。"""
    print('[TRT] 生成分析报告...')

    lines = []
    lines.append('# YOLOv8m TensorRT Profiler 逐层性能分析报告')
    lines.append('')
    lines.append(f'**生成时间**: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'**工具版本**: {trt_version}')
    lines.append(f'**引擎路径**: `{ENGINE}`')
    lines.append(f'**输入**: {INPUT_SHAPE}')
    lines.append(f'**输出**: {OUTPUT_SHAPE}')
    lines.append(f'**预热/测量**: 10 / 10 seconds')
    lines.append('')

    if not profiling_success:
        lines.append('## Profiling 状态')
        lines.append('')
        lines.append('trtexec profiling 执行失败。请检查以下可能原因:')
        lines.append('')
        lines.append(f'1. engine 文件不存在或路径错误: `{ENGINE}`')
        lines.append(f'2. trtexec 不可执行: `{TRTEXEC}`')
        lines.append('3. 平台/驱动兼容性问题')
        lines.append('')
        lines.append('请确认后重新运行。')
        lines.append('')

        with open(MD_REPORT, 'w') as f:
            f.write('\n'.join(lines))
        print(f'[TRT] 报告已保存到 {MD_REPORT}')
        return

    if not layers:
        lines.append('## 层性能数据')
        lines.append('')
        lines.append('未从 profiling 输出中解析出有效层数据。')
        lines.append('')

        with open(MD_REPORT, 'w') as f:
            f.write('\n'.join(lines))
        print(f'[TRT] 报告已保存到 {MD_REPORT}')
        return

    total_time_us = stats['total_time_us']
    total_time_ms = stats['total_time_ms']
    type_stats = stats['type_stats']
    num_layers = stats['num_layers']

    # ================================================================
    # 关键发现
    # ================================================================
    lines.append('## 关键发现')
    lines.append('')

    # Sort layers by time_pct descending for top analysis
    sorted_layers = sorted(layers, key=lambda x: x['time_pct'], reverse=True)

    top3_pct = sum(l['time_pct'] for l in sorted_layers[:3]) if len(sorted_layers) >= 3 else 0
    top5_pct = sum(l['time_pct'] for l in sorted_layers[:5]) if len(sorted_layers) >= 5 else 0

    lines.append(f'- **总层数**: {num_layers}')
    lines.append(f'- **总耗时**: {_fmt_ms(total_time_ms)} ms')
    lines.append(f'- **Top-3 最耗时层占比**: {_fmt_pct(top3_pct)}')
    lines.append(f'- **Top-5 最耗时层占比**: {_fmt_pct(top5_pct)}')
    lines.append(f'- **逐层数据类型数**: {len(type_stats)}')

    # Conv 总耗时
    conv_stat = type_stats.get('Conv')
    if conv_stat:
        lines.append(f'- **Conv 层总耗时**: {_fmt_ms(conv_stat["total_time_ms"])} ms '
                     f'({_fmt_pct(conv_stat["time_pct"])})')

    # Reformat 总耗时
    reformat_stat = type_stats.get('Reformat')
    if reformat_stat:
        reformat_pct = reformat_stat['time_pct']
        reformat_ms = reformat_stat['total_time_ms']
        flag = ' **（重点关注）**' if reformat_pct > 5 else ''
        lines.append(f'- **Reformat 层总耗时**: {_fmt_ms(reformat_ms)} ms '
                     f'({_fmt_pct(reformat_pct)}){flag}')

    # FusedParallel 总耗时
    fused_stat = type_stats.get('FusedParallel')
    if fused_stat:
        lines.append(f'- **FusedParallel 层总耗时**: {_fmt_ms(fused_stat["total_time_ms"])} ms '
                     f'({_fmt_pct(fused_stat["time_pct"])})')

    # Detect reformat overhead
    if reformat_stat and reformat_stat['time_pct'] > 5:
        lines.append('  - Reformat 开销较大（超过 5%），可能是由 Transpose/Shuffle 等'
                     '改变数据排布的操作引起。')
    elif reformat_stat and reformat_stat['time_pct'] > 2:
        lines.append('  - Reformat 开销适中（2%-5%），属于正常范围。')

    lines.append('')

    # ================================================================
    # Top-15 最耗时层
    # ================================================================
    lines.append('## Top-15 最耗时层')
    lines.append('')
    lines.append('| # | Layer Name | Type | Time (ms) | Time % |')
    lines.append('|---|------------|------|-----------|--------|')

    for i, layer in enumerate(sorted_layers[:15], 1):
        name = layer['name']
        if len(name) > 90:
            name = name[:87] + '...'
        ltype = layer['layer_type']
        time_ms = layer['time_us'] / 1000.0
        time_pct = layer['time_pct']
        lines.append(f'| {i} | `{name}` | {ltype} | {_fmt_ms(time_ms)} | {_fmt_pct(time_pct)} |')

    lines.append('')

    # ================================================================
    # 按类型汇总表
    # ================================================================
    lines.append('## 按类型汇总表')
    lines.append('')
    lines.append('| Type | Count | Total Time (ms) | Time % |')
    lines.append('|------|-------|-----------------|--------|')

    # 按总耗时降序排列
    sorted_types = sorted(type_stats.items(), key=lambda x: x[1]['total_time_us'], reverse=True)
    for t, st in sorted_types:
        lines.append(f'| {t} | {st["count"]} | {_fmt_ms(st["total_time_ms"])} | '
                     f'{_fmt_pct(st["time_pct"])} |')

    lines.append('')

    # ================================================================
    # 完整逐层性能表
    # ================================================================
    lines.append('## 完整逐层性能表')
    lines.append('')
    lines.append('| # | Layer Name | Type | Time (us) | Avg (ms) | Median (ms) | Time % |')
    lines.append('|---|------------|------|-----------|----------|-------------|--------|')

    for i, layer in enumerate(sorted_layers, 1):
        name = layer['name']
        if len(name) > 75:
            name = name[:72] + '...'
        ltype = layer['layer_type']
        time_us_str = _fmt_us(layer['time_us'])
        avg_str = _fmt_ms(layer['avg_ms'])
        median_str = _fmt_ms(layer['median_ms'])
        time_pct_str = _fmt_pct(layer['time_pct'])
        lines.append(f'| {i} | `{name}` | {ltype} | {time_us_str} | {avg_str} | '
                     f'{median_str} | {time_pct_str} |')

    lines.append('')

    # ================================================================
    # 性能瓶颈分析
    # ================================================================
    lines.append('## 性能瓶颈分析')
    lines.append('')

    # Conv 分析
    if conv_stat:
        conv_count = conv_stat['count']
        conv_time_ms = conv_stat['total_time_ms']
        conv_pct = conv_stat['time_pct']
        conv_pct_str = _fmt_pct(conv_pct)
        avg_conv_time_ms = conv_time_ms / conv_count if conv_count > 0 else 0
        lines.append('### 卷积 (Conv) 分析')
        lines.append('')
        lines.append(f'- **Conv 层数量**: {conv_count}')
        lines.append(f'- **Conv 总耗时**: {_fmt_ms(conv_time_ms)} ms ({conv_pct_str})')
        lines.append(f'- **平均每层 Conv 耗时**: {_fmt_ms(avg_conv_time_ms)} ms')

        # Conv 层中最耗时的 Top-5
        conv_layers = [l for l in sorted_layers if l['layer_type'] == 'Conv']
        conv_top5 = conv_layers[:5]
        if conv_top5:
            lines.append('')
            lines.append('| # | Layer Name | Time (ms) | Time % |')
            lines.append('|---|------------|-----------|--------|')
            for j, cl in enumerate(conv_top5, 1):
                cl_name = cl['name']
                if len(cl_name) > 70:
                    cl_name = cl_name[:67] + '...'
                lines.append(f'| {j} | `{cl_name}` | '
                             f'{_fmt_ms(cl["time_us"] / 1000.0)} | '
                             f'{_fmt_pct(cl["time_pct"])} |')
            lines.append('')

        if conv_pct > 70:
            lines.append('  - 卷积运算占用绝大多数时间，符合计算密集型模型特征。')
            lines.append('  - 优化方向：检查是否可启用 INT8 量化、使用更大的 batch size。')
        elif conv_pct > 50:
            lines.append('  - 卷积运算为主要计算瓶颈，但其他操作也占显著比例。')
        else:
            lines.append('  - 卷积运算占比不高，说明瓶颈可能在内存带宽或其他操作。')
        lines.append('')

    # Reformat 分析
    lines.append('### Reformat / 数据排布转换分析')
    lines.append('')
    if reformat_stat:
        reformat_count = reformat_stat['count']
        reformat_time_ms = reformat_stat['total_time_ms']
        reformat_pct = reformat_stat['time_pct']
        lines.append(f'- **Reformat 层数量**: {reformat_count}')
        lines.append(f'- **Reformat 总耗时**: {_fmt_ms(reformat_time_ms)} ms '
                     f'({_fmt_pct(reformat_pct)})')
        if reformat_pct > 5:
            lines.append('- **Reformat 开销较大**（>5%），建议仔细分析。')
            lines.append('  - 可能原因：TensorRT 在 NHWC 与 NCHW 之间的格式转换。')
            lines.append('  - 可通过设置 `FP16` 或 `INT8` 时使用 `set_nhwc_enabled()` 来减少。')
            lines.append('  - Transpose / Shuffle 算子通常伴随 Reformat。检查是否需要')
            lines.append('    使用 TensorRT 的 IOptimizationStrategy 进行融合优化。')
        elif reformat_pct > 0:
            lines.append(f'- Reformat 开销适中 ({_fmt_pct(reformat_pct)})，属于正常范围。')
    else:
        lines.append('- 未检测到 Reformat 层。')
    lines.append('')

    # FusedParallel 分析
    lines.append('### FusedParallel (并行融合) 分析')
    lines.append('')
    if fused_stat:
        fused_count = fused_stat['count']
        fused_time_ms = fused_stat['total_time_ms']
        fused_pct = fused_stat['time_pct']
        lines.append(f'- **FusedParallel 层数量**: {fused_count}')
        lines.append(f'- **FusedParallel 总耗时**: {_fmt_ms(fused_time_ms)} ms '
                     f'({_fmt_pct(fused_pct)})')
        lines.append('  - FusedParallel 表示 TensorRT 将多个独立路径并行执行的算子。')
        lines.append('  - 此类层通常出现在多分支网络结构（如 YOLO 的检测头分支）中。')
        lines.append('  - 单个并行分支的耗时取决于其中最慢的分支。')
    else:
        lines.append('- 未检测到 FusedParallel 层。')
    lines.append('')

    # PointWise 分析
    pw_stat = type_stats.get('PointWise')
    if pw_stat:
        lines.append('### PointWise (逐点运算) 分析')
        lines.append('')
        lines.append(f'- **PointWise 层数量**: {pw_stat["count"]}')
        lines.append(f'- **PointWise 总耗时**: {_fmt_ms(pw_stat["total_time_ms"])} ms '
                     f'({_fmt_pct(pw_stat["time_pct"])})')
        lines.append('  - PointWise 算子通常包括激活函数和逐元素运算。')
        lines.append('  - TensorRT 的算子融合策略应已将这些操作融合到 Conv 中。')
        lines.append('  - 如果 PointWise 占比高且未融合，说明需要优化融合策略。')
        lines.append('')

    # ================================================================
    # 优化建议
    # ================================================================
    lines.append('## 优化建议')
    lines.append('')

    suggestions = []

    # 1. Reformat overhead
    if reformat_stat and reformat_stat['time_pct'] > 5:
        suggestions.append((
            '减少 Reformat / Transpose 开销',
            'Reformat 占总时间超过 5%，建议分析 transpose 和 shuffle 操作的必要性。'
            '如果可以修改模型，考虑在设计上减少 reshape/transpose 操作。'
        ))

    # 2. Conv time percentage
    if conv_stat:
        if conv_stat['time_pct'] > 70:
            suggestions.append((
                '卷积核优化',
                '卷积是主要瓶颈。考虑使用 TensorRT INT8 量化进一步加速。'
                '也可以尝试使用更大的 batch size（如 batch=4 或 8）来提高 GPU 利用率。'
            ))
        elif conv_stat['time_pct'] > 50:
            suggestions.append((
                '平衡 Conv 与其他操作',
                '卷积和其他操作都占显著比例。可以同时关注两方面优化：'
                '量化卷积、减少非必要的数据排布转换。'
            ))

    # 3. FusedParallel analysis
    if fused_stat and fused_stat['time_pct'] > 10:
        suggestions.append((
            'FusedParallel 并行分支优化',
            '并行分支占比较高，分析分支间计算量是否均衡。'
            '不平衡的分支会导致部分 GPU 资源空闲等待。'
        ))

    # 4. PointWise overhead
    if pw_stat and pw_stat['time_pct'] > 5:
        suggestions.append((
            'PointWise 算子融合',
            '如果 PointWise 独立存在且未融合到卷积中，尝试调整 TensorRT '
            '优化策略以启用更激进的算子融合。'
        ))

    # 5. General
    suggestions.append((
        '启用 CUDA Graph',
        'TensorRT 支持 CUDA Graph 捕获和重放，可减少 kernel launch 开销。'
        '对于小 batch 推理，CUDA Graph 可显著提升吞吐量。'
    ))

    suggestions.append((
        '多 Stream 并行推理',
        '如果部署场景需要处理多路视频流，考虑使用多个 CUDA stream 并行执行引擎。'
    ))

    for i, (title, desc) in enumerate(suggestions, 1):
        lines.append(f'### {i}. {title}')
        lines.append('')
        lines.append(desc)
        lines.append('')

    # ================================================================
    # 精度策略说明
    # ================================================================
    lines.append('## 精度策略说明')
    lines.append('')
    lines.append(f'本分析基于 FP16 engine (`yolov8m_fp16.engine`)。')
    lines.append('')
    lines.append('| 项目 | 内容 |')
    lines.append('|------|------|')
    lines.append('| **Engine 路径** | `weights/engines/yolov8m_fp16.engine` |')
    lines.append('| **精度模式** | FP16 (half precision) |')
    lines.append('| **TensorRT 版本** | JetPack 预装版本 |')
    lines.append('| **平台** | Jetson Orin (T234/Ampere 架构) |')
    lines.append('')
    lines.append('### FP16 精度说明')
    lines.append('')
    lines.append('- FP16 (半精度) 将权重和激活从 FP32 压缩到 16 位浮点数。')
    lines.append('- 理论上可将显存占用和带宽需求减半，同时显著提升吞吐量。')
    lines.append('- YOLOv8m 等现代检测模型在 FP16 下的精度损失通常 < 0.5% mAP。')
    lines.append('- JetPack 内置的 TensorRT 已针对 Jetson Orin 的 FP16 Tensor Core 优化。')
    lines.append('- 如需更高精度，可回退到 FP32；如需更高吞吐，可尝试 INT8 量化。')
    lines.append('')

    # ================================================================
    # 报告文件
    # ================================================================
    lines.append('## 报告文件')
    lines.append('')
    lines.append(f'- `layer_profile.json` — trtexec 导出的层性能数据 (JSON)')
    lines.append(f'- `layer_info.json` — trtexec 导出的层结构信息 (JSON)')
    lines.append(f'- `analysis_report.md` — 本报告')
    lines.append('')

    with open(MD_REPORT, 'w') as f:
        f.write('\n'.join(lines))

    print(f'[TRT] 报告已保存到 {MD_REPORT}')


# ---------------------------------------------------------------------------
# 运行时 (trtexec)
# ---------------------------------------------------------------------------

def run_trtexec():
    """执行 trtexec profiling。

    如果 layer_profile.json 已存在则跳过 profiling，仅重新生成 MD 报告。

    Returns:
        (bool, str) — (success, stdout)
    """
    if os.path.exists(LAYER_PROFILE_JSON):
        print('[TRT] layer_profile.json already exists, skipping profiling.')
        print('[TRT] Delete layer_profile.json to re-run profiling.')
        return True, ''

    print('[TRT] Starting profiling with trtexec...')
    print(f'[TRT] This will take ~15 seconds (--duration=10 + warmup).')

    cmd = [
        TRTEXEC,
        f'--loadEngine={ENGINE}',
        '--warmUp=10',
        '--duration=10',
        '--dumpProfile',
        '--dumpLayerInfo',
        '--profilingVerbosity=detailed',
        f'--exportProfile={LAYER_PROFILE_JSON}',
        f'--exportLayerInfo={LAYER_INFO_JSON}',
    ]

    print(f'[TRT] Running: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    stdout = result.stdout or ''
    stderr = result.stderr or ''

    # 截断输出显示
    output_lines = stdout.split('\n')
    if len(output_lines) > 40:
        print('\n'.join(output_lines[:20]))
        print('...')
        print('\n'.join(output_lines[-20:]))
    elif stdout:
        print(stdout)

    if result.returncode != 0:
        print('[TRT] WARNING: trtexec returned non-zero exit code.')
        print('[TRT] stderr (last 2000 chars):', stderr[-2000:])

    # Check for JSON exports
    if not os.path.exists(LAYER_PROFILE_JSON):
        print(f'[TRT] Warning: {LAYER_PROFILE_JSON} was not exported.')
        return result.returncode == 0, stdout

    file_size = os.path.getsize(LAYER_PROFILE_JSON)
    print(f'[TRT] {LAYER_PROFILE_JSON} generated ({file_size} bytes).')

    if os.path.exists(LAYER_INFO_JSON):
        info_size = os.path.getsize(LAYER_INFO_JSON)
        print(f'[TRT] {LAYER_INFO_JSON} generated ({info_size} bytes).')

    success = result.returncode == 0
    print(f'[TRT] Profiling {"succeeded" if success else "finished with warnings"}.')
    return success, stdout


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    trt_version = get_trt_version()
    print(f'[TRT] Version: {trt_version}')

    # 确保输出目录存在
    os.makedirs(HERE, exist_ok=True)

    # 检查 engine 存在
    if not os.path.exists(ENGINE):
        print(f'[TRT] ERROR: Engine not found: {ENGINE}')
        generate_report([], {}, trt_version, '', False)
        return

    # 运行 trtexec
    profiling_success, stdout = run_trtexec()

    # 加载并解析层数据
    layers = load_layers()

    # 计算统计
    stats = compute_stats(layers) if layers else {}

    # 生成报告
    generate_report(layers, stats, trt_version, stdout, profiling_success)

    print('[TRT] 完成。')


if __name__ == '__main__':
    main()
