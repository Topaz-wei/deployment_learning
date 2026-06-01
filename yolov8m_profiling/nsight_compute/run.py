#!/usr/bin/env python3
"""Nsight Compute 分析: 生成 report.ncu-rep + report.csv, 解析并写 MD 报告。

用法:
    conda activate py38
    cd yolov8m_profiling/nsight_compute
    python3 run.py

注意:
    - Jetson Orin 集成 GPU 上 ncu 需要 root 权限才能访问性能计数器。
      如果 profiling 因权限不足而失败，脚本将继续生成说明文档，
      并建议在宿主机 (x86_64 with discrete GPU) 上使用 ncu-ui 分析。
    - LD_PRELOAD 与 ncu 冲突，不要在调用 ncu 时预加载 OpenBLAS。
    - TensorRT 推理走 CUDA，不需要 CPU BLAS 加速。
    - 如果 report.ncu-rep 已存在则跳过 profiling，仅重新生成 MD 报告。
"""
import csv
import os
import re
import subprocess
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
WORKLOAD = os.path.join(ROOT, 'common', 'inference_workload.py')
ENGINE = os.path.join(ROOT, '..', 'weights', 'engines', 'yolov8m_fp16.engine')
PYTHON = '/home/ssd/anaconda3/envs/py38/bin/python3'
NCU = '/opt/nvidia/nsight-compute/2022.2.1/ncu'

REPORT_BASE = os.path.join(HERE, 'report')        # 生成 report.ncu-rep
CSV_REPORT = os.path.join(HERE, 'report.csv')      # CSV 导出
MD_REPORT = os.path.join(HERE, 'analysis_report.md')


def get_ncu_version():
    """获取已安装 Nsight Compute 版本号。"""
    try:
        result = subprocess.run([NCU, '--version'], capture_output=True, text=True, timeout=30)
        raw = (result.stdout or result.stderr or '').strip()
        # 第一行应为 "NVIDIA (R) Nsight Compute Command Line Profiler"
        lines = raw.split('\n')
        for line in lines:
            if 'Version' in line:
                return line.strip()
        return raw if raw else 'Nsight Compute (unknown)'
    except Exception as e:
        print(f'[NCU] WARNING: 无法获取版本: {e}')
        return 'Nsight Compute (unknown)'


def check_ncu_root():
    """检查 ncu 是否需要 root 权限（Jetson Orin 上的已知限制）。"""
    try:
        result = subprocess.run(
            [NCU, '--query-metrics'],
            capture_output=True, text=True, timeout=30
        )
        if 'Insufficient privileges' in (result.stdout + result.stderr):
            return False, 'Insufficient privileges — ncu 需要 root 权限'
        if result.returncode != 0:
            err = (result.stderr or result.stdout or '').strip()
            return False, f'ncu 返回非零退出码: {err}'
        return True, 'OK'
    except FileNotFoundError:
        return False, f'ncu 未找到: {NCU}'
    except Exception as e:
        return False, f'检查 ncu 权限时出错: {e}'


def run_ncu():
    """执行 ncu profiling (如果 report.ncu-rep 已存在则跳过)。

    Returns:
        (bool, str) — (是否成功, ncu 版本字符串或错误信息)
    """
    ncu_version = get_ncu_version()
    print(f'[NCU] Version: {ncu_version}')

    # 如果 report.ncu-rep 已存在，跳过 profiling
    if os.path.exists(REPORT_BASE + '.ncu-rep'):
        print('[NCU] report.ncu-rep 已存在，跳过 profiling。')
        print('[NCU] 删除 report.ncu-rep 以重新运行 profiling。')
        return True, ncu_version

    # 检查 root 权限
    can_profile, msg = check_ncu_root()
    if not can_profile:
        print(f'[NCU] WARNING: {msg}')
        print('[NCU] Jetson Orin 集成 GPU 上 ncu 需要 root 权限。')
        print('[NCU] 跳过 profiling，生成限制说明报告。')
        print('[NCU] 如需在 Jetson 上运行，请使用 sudo:')
        print(f'    sudo {NCU} ...')
        return False, ncu_version

    print('[NCU] Starting profiling...')

    # ncu 参数:
    #   --set basic        — 收集基本指标，比 full 更兼容
    #   --replay-mode kernel  — 默认值，每个 kernel launch replay 多次
    #   --launch-count 1   — 只 profile 一次
    #   --cache-control none  — 不修改缓存策略
    # 低迭代次数 (5 warmup, 10 iters) 保证 profiling 速度
    cmd = [
        NCU,
        '--set', 'basic',
        '--export', REPORT_BASE + '.ncu-rep',
        '--force-overwrite',
        '--launch-count', '1',
        '--cache-control', 'none',
        '--target-processes', 'application-only',
        PYTHON, WORKLOAD,
        '--engine', ENGINE,
        '--data', 'dummy',
        '--warmup', '5',
        '--iters', '10',
    ]

    print(f'[NCU] Running: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)

    # ncu 输出
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print('[NCU] stderr:', result.stderr[-2000:])

    if result.returncode != 0:
        print('[NCU] Profiling 失败。')
        print('[NCU] 建议在宿主机 (x86_64 with discrete GPU) 上使用 ncu-ui 进行分析。')
        return False, ncu_version

    if not os.path.exists(REPORT_BASE + '.ncu-rep'):
        print('[NCU] report.ncu-rep 未生成。')
        return False, ncu_version

    print(f'[NCU] Profiling 完成。报告: {REPORT_BASE}.ncu-rep')
    return True, ncu_version


def dump_csv():
    """从 report.ncu-rep 导出 CSV 数据。

    使用 ncu --import 打开 report.ncu-rep 并导出为 CSV。
    支持两种导出方式:
      1. ncu -i report.ncu-rep --csv --page details       (详细指标)
      2. ncu -i report.ncu-rep --csv --page raw            (原始指标)

    Returns:
        bool — CSV 是否成功导出
    """
    ncu_rep = REPORT_BASE + '.ncu-rep'
    if not os.path.exists(ncu_rep):
        print('[NCU] CSV 导出跳过: report.ncu-rep 不存在')
        return False

    print('[NCU] 导出 CSV (page raw)...')
    try:
        with open(CSV_REPORT, 'w') as csv_f:
            result = subprocess.run(
                [NCU, '--import', ncu_rep, '--csv', '--page', 'raw'],
                capture_output=True, text=True, timeout=120
            )
            csv_f.write(result.stdout)
            if result.stderr:
                csv_f.write('\n# STDERR:\n')
                csv_f.write(result.stderr)

        # 检查是否成功写入
        file_size = os.path.getsize(CSV_REPORT)
        print(f'[NCU] CSV 导出成功: {CSV_REPORT} ({file_size} bytes)')
        return True
    except Exception as e:
        print(f'[NCU] CSV 导出失败: {e}')
        return False


def parse_kernel_csv():
    """解析 report.csv, 提取 kernel 性能指标。

    支持两种 CSV 格式:
      1. --page raw: 每行一个 kernel launch，指标作为列 (230+ 列)
      2. --page details: 每行一个指标 (kernel, metric, value)

    Returns:
        list[dict]: 每个唯一 kernel 的汇总指标字典列表
    """
    kernels = []
    if not os.path.exists(CSV_REPORT):
        return kernels

    print('[NCU] 解析 kernel CSV 数据...')

    try:
        with open(CSV_REPORT, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f'[NCU] CSV 读取失败: {e}')
        return kernels

    if not rows:
        return kernels

    # 判断格式: raw 格式的列名中含 "launch__" 或 "sm__"
    fieldnames = list(rows[0].keys())
    is_raw_format = any('launch__' in fn or 'sm__' in fn for fn in fieldnames)

    if is_raw_format:
        print('[NCU] 检测到 raw CSV 格式，按 kernel 类型汇总...')
        return _parse_raw_csv(rows)
    else:
        print('[NCU] 检测到 details CSV 格式...')
        return _parse_standard_csv(rows)


def _safe_float(value, default=0.0):
    """安全地将值转为 float，失败时返回 default。"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _parse_raw_csv(rows):
    """解析 --page raw 格式的 CSV。

    每行是一个 kernel launch 事件，有 230+ 列，包括:
      - Kernel Name, Kernel Time (datetime)
      - launch__* (block/grid/register/shared_mem 等)
      - sm__* (warp occupancy 等)
      - device__attribute_* (GPU 属性)
    """
    from collections import defaultdict

    kernel_stats = defaultdict(lambda: {
        'name': '', 'calls': 0,
        'registers_per_thread': 0,
        'block_size': 0,
        'grid_size': 0,
        'thread_count': 0,
        'shared_mem_static_kb': 0,
        'shared_mem_dynamic_kb': 0,
        'warps_avg': 0,
        'warps_pct': 0,
        'occupancy_block': 0,
        'occupancy_register': 0,
        'occupancy_shared': 0,
    })

    for row in rows:
        name = row.get('Kernel Name', '')
        if not name:
            continue
        short = name.split('(')[0].strip().replace('void ', '')

        stats = kernel_stats[short]
        stats['name'] = short
        stats['calls'] += 1

        # 取第一个样本的指标（同一 kernel 的不同 launch 参数基本相同）
        if stats['calls'] == 1:
            stats['registers_per_thread'] = _safe_float(row.get('launch__registers_per_thread', 0))
            stats['block_size'] = int(_safe_float(row.get('launch__block_size', 0)))
            stats['grid_size'] = int(_safe_float(row.get('launch__grid_size', 0)))
            stats['thread_count'] = int(_safe_float(row.get('launch__thread_count', 0)))
            stats['shared_mem_static_kb'] = _safe_float(row.get('launch__shared_mem_per_block_static', 0))
            stats['shared_mem_dynamic_kb'] = _safe_float(row.get('launch__shared_mem_per_block_dynamic', 0))
            stats['warps_avg'] = _safe_float(row.get('sm__maximum_warps_avg_per_active_cycle', 0))
            stats['warps_pct'] = _safe_float(row.get('sm__maximum_warps_per_active_cycle_pct', 0))
            stats['occupancy_block'] = _safe_float(row.get('launch__occupancy_per_block_size', 0))
            stats['occupancy_register'] = _safe_float(row.get('launch__occupancy_per_register_count', 0))
            stats['occupancy_shared'] = _safe_float(row.get('launch__occupancy_per_shared_mem_size', 0))

    return sorted(kernel_stats.values(), key=lambda x: x['calls'], reverse=True)


def _parse_standard_csv(lines):
    """解析标准单行指标 CSV 格式。"""
    kernels = {}
    ordered_kernels = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('"ID"'):
            continue

        # 解析 CSV 行: "ID","Kernel Name","Metric","Value","Unit"
        parts = _parse_csv_line(line)
        if len(parts) < 5:
            continue

        kernel_id = parts[0]
        kernel_name = parts[1]
        metric_name = parts[2]
        metric_value = parts[3]
        metric_unit = parts[4]

        if kernel_name not in kernels:
            kernels[kernel_name] = {
                'name': kernel_name,
                'id': kernel_id,
                'metrics': {},
            }
            ordered_kernels.append(kernel_name)

        if metric_value and metric_value != '-':
            try:
                num_val = float(metric_value)
                kernels[kernel_name]['metrics'][metric_name] = {
                    'value': num_val,
                    'unit': metric_unit,
                }
            except ValueError:
                pass

    return [kernels[name] for name in ordered_kernels]


def _parse_section_csv(lines):
    """解析 section-based CSV 格式。"""
    kernels = {}
    ordered_kernels = []

    kernel_name = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 检查 section header: "[Section Name]","Kernel Name",...
        if line.startswith('['):
            parts = _parse_csv_line(line)
            if len(parts) >= 2:
                kernel_name = parts[1]
                if kernel_name and kernel_name != 'Kernel Name':
                    if kernel_name not in kernels:
                        kernels[kernel_name] = {
                            'name': kernel_name,
                            'metrics': {},
                        }
                        ordered_kernels.append(kernel_name)
            continue

        # 数据行: "Metric","Value","Unit"
        parts = _parse_csv_line(line)
        if len(parts) >= 3 and kernel_name and kernel_name in kernels:
            metric_name = parts[0]
            metric_value = parts[1]
            metric_unit = parts[2] if len(parts) > 2 else ''
            if metric_value and metric_value != '-':
                try:
                    num_val = float(metric_value)
                    kernels[kernel_name]['metrics'][metric_name] = {
                        'value': num_val,
                        'unit': metric_unit,
                    }
                except ValueError:
                    pass

    return [kernels[name] for name in ordered_kernels]


def _parse_csv_line(line):
    """解析一行 CSV, 处理引号包围的字段。"""
    parts = []
    current = ''
    in_quotes = False
    for ch in line:
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == ',' and not in_quotes:
            parts.append(current)
            current = ''
        else:
            current += ch
    parts.append(current)
    return parts


def detect_nv_fuser():
    """检测 GPU 平台信息。"""
    return 'Jetson Orin 集成 GPU (DRAM 统一内存)'


def _fmt_ns(ns):
    """将纳秒格式化为可读字符串 (us / ms / s)。"""
    if ns is None or ns == 0:
        return 'N/A'
    if ns < 1000:
        return f'{ns:.0f} ns'
    elif ns < 1_000_000:
        return f'{ns / 1000:.1f} us'
    elif ns < 1_000_000_000:
        return f'{ns / 1_000_000:.2f} ms'
    else:
        return f'{ns / 1_000_000_000:.3f} s'


def _fmt_pct(val):
    """将 0-1 或 0-100 值格式化为百分比字符串。"""
    if val is None:
        return 'N/A'
    if val <= 1.0:
        return f'{val * 100:.1f}%'
    return f'{val:.1f}%'


def generate_report(ncu_success, ncu_version, kernels, csv_generated):
    """生成 MD 分析报告。

    Args:
        ncu_success: bool — ncu profiling 是否成功
        ncu_version: str — ncu 版本描述
        kernels: list[dict] — 解析后的 kernel 数据
        csv_generated: bool — CSV 是否成功导出
    """
    print('[NCU] 生成分析报告...')

    lines = []
    lines.append('# YOLOv8m Nsight Compute 性能分析报告')
    lines.append('')
    lines.append(f'**生成时间**: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'**工具版本**: {ncu_version}')
    lines.append(f'**引擎**: `weights/engines/yolov8m_fp16.engine`')
    lines.append(f'**输入**: images (1, 3, 640, 640) float32, dummy data')
    lines.append(f'**输出**: output0 (1, 84, 8400) float32')
    lines.append(f'**平台**: {detect_nv_fuser()}')
    lines.append(f'**预热/测量**: 5 / 10 iterations')
    lines.append('')

    # ================================================================
    # Profiling 状态
    # ================================================================
    if not ncu_success:
        lines.append('## Profiling 状态: 无法在当前平台运行')
        lines.append('')
        lines.append('Nsight Compute 命令行工具 (`ncu`) 在 Jetson Orin 集成 GPU 上无法直接进行 kernel')
        lines.append('级 profiling，原因如下：')
        lines.append('')
        lines.append('### 已知限制')
        lines.append('')
        lines.append('- **权限要求**: ncu 需要 root 权限才能访问 GPU 性能计数器，但嵌入式 Jetson 平台')
        lines.append('  通常限制直接 root 访问。')
        lines.append('- **架构差异**: Jetson Orin 使用集成 GPU (T234/Ampere 架构)，与 x86_64 桌面/服务器')
        lines.append('  GPU 在驱动程序模型和性能计数器访问接口上存在差异。')
        lines.append('- **ncu-ui 不支持**: 当前的 Nsight Compute 版本 (2022.2.1) 不支持在 ARM64 上运行')
        lines.append('  ncu-ui 图形界面。')
        lines.append('')
        lines.append('### 建议的替代方案')
        lines.append('')
        lines.append('1. **在宿主机上使用 ncu-ui 分析**:')
        lines.append('   - 在 x86_64 桌面 (带独立 NVIDIA GPU) 上安装 Nsight Compute。')
        lines.append('   - 将 TensorRT engine 拷贝到宿主机。')
        lines.append('   - 使用 ncu-ui 打开并分析 engine 的 profiling 结果。')
        lines.append('')
        lines.append('2. **使用 TensorRT 内置 Profiler**:')
        lines.append('   - 见 `trt_profiler/` 目录下的 IProfiler 实现，可获取每个 layer 的性能数据。')
        lines.append('')
        lines.append('3. **使用 Nsight Systems 做粗粒度分析**:')
        lines.append('   - 见 `nsight_systems/` 目录，可获取 kernel 时间线、CPU/GPU 时间分布。')
        lines.append('')
        lines.append('4. **通过 ncu --set basic 手动重试 (需 sudo)**:')
        lines.append('   ```bash')
        lines.append(f'   sudo {NCU} --set basic \\')
        lines.append(f'     --export {REPORT_BASE}.ncu-rep \\')
        lines.append(f'     --force-overwrite \\')
        lines.append(f'     {PYTHON} {WORKLOAD} \\')
        lines.append(f'     --engine {ENGINE} --data dummy --warmup 5 --iters 10')
        lines.append('   ```')
        lines.append('   ```bash')
        lines.append(f'   {NCU} -i {REPORT_BASE}.ncu-rep --csv --page details > {CSV_REPORT}')
        lines.append('   ```')
        lines.append('')
        lines.append('   然后再运行本脚本 (因检测到 report.ncu-rep 已存在, 将跳过 profiling,')
        lines.append('   直接解析 CSV 并生成完整报告)。')
        lines.append('')

        # 即使 profiling 没成功，也记录报告文件
        lines.append('## 报告文件')
        lines.append('')
        if os.path.exists(REPORT_BASE + '.ncu-rep'):
            lines.append(f'- `report.ncu-rep` — 手动生成的 profiling 文件')
        if csv_generated:
            lines.append(f'- `report.csv` — CSV 导出文件')
        lines.append(f'- `analysis_report.md` — 本报告')
        lines.append('')

        with open(MD_REPORT, 'w') as f:
            f.write('\n'.join(lines))
        print(f'[NCU] 报告已保存到 {MD_REPORT}')
        return

    # ================================================================
    # 如果 ncu profiling 成功, 生成详细分析报告
    # ================================================================
    lines.append('## 关键发现')
    lines.append('')

    if kernels:
        # 计算总调用次数
        total_calls = sum(k['calls'] for k in kernels)

        lines.append(f'- **总 kernel 调用次数**: {total_calls}')
        lines.append(f'- **唯一 kernel 类型数**: {len(kernels)}')
        lines.append(f'- **最高调用频率**: `{kernels[0]["name"][:60]}` ({kernels[0]["calls"]} 次, 占 {kernels[0]["calls"]/total_calls*100:.1f}%)')
        lines.append('')

        # Kernel 调用次数排名表
        lines.append('## Kernel 调用次数排名')
        lines.append('')
        lines.append('| # | Kernel | 调用次数 | 占比 | Warp Occupancy | Registers | Block Size | Shared Mem |')
        lines.append('|---|--------|---------|------|----------------|-----------|------------|------------|')
        for i, k in enumerate(kernels[:20], 1):
            name = k['name'][:50] + ('...' if len(k['name']) > 50 else '')
            pct = k['calls'] / total_calls * 100
            warp_str = f'{k["warps_pct"]:.0f}%' if k['warps_pct'] > 0 else 'N/A'
            reg_str = f'{k["registers_per_thread"]:.0f}' if k['registers_per_thread'] > 0 else 'N/A'
            shm = k['shared_mem_static_kb'] + k['shared_mem_dynamic_kb']
            shm_str = f'{shm:.1f} KB' if shm > 0 else '0'
            lines.append(f'| {i} | `{name}` | {k["calls"]} | {pct:.1f}% | {warp_str} | {reg_str} | {k["block_size"]} | {shm_str} |')
        lines.append('')

        # 按 warp occupancy 分类
        high_occ = [k for k in kernels if k['warps_pct'] >= 50]
        mid_occ = [k for k in kernels if 20 <= k['warps_pct'] < 50]
        low_occ = [k for k in kernels if 0 < k['warps_pct'] < 20]
        zero_occ = [k for k in kernels if k['warps_pct'] == 0]

        lines.append('## SM 利用率与 Occupancy 分析')
        lines.append('')
        lines.append(f'| 分类 | Kernel 类型数 | 说明 |')
        lines.append('|------|-------------|------|')
        lines.append(f'| Warp Occupancy >= 50% | {len(high_occ)} | 计算/延迟隐藏良好 |')
        lines.append(f'| Warp Occupancy 20-50% | {len(mid_occ)} | 中等利用率 |')
        lines.append(f'| Warp Occupancy < 20% | {len(low_occ)} | 低利用率，被寄存器或共享内存限制 |')
        lines.append(f'| 无数据 | {len(zero_occ)} | 指标不可用 |')
        lines.append('')

        # 关键指标统计
        lines.append('## 内存与寄存器使用')
        lines.append('')
        lines.append('| Kernel | 调用数 | Registers | Block Sz | Threads | Static SM | Dynamic SM | Warps |')
        lines.append('|--------|--------|-----------|----------|---------|-----------|------------|-------|')
        for k in kernels[:15]:
            name = k['name'][:45] + ('...' if len(k['name']) > 45 else '')
            lines.append(f'| `{name}` | {k["calls"]} | {k["registers_per_thread"]:.0f} | {k["block_size"]} | {k["thread_count"]} | {k["shared_mem_static_kb"]:.1f} KB | {k["shared_mem_dynamic_kb"]:.1f} KB | {k["warps_avg"]:.0f} |')
        lines.append('')

        # 分析
        lines.append('## 关键分析')
        lines.append('')

        # 按 kernel 类别分组
        gemm_kernels = [k for k in kernels if 'xmma' in k['name'].lower() or 'gemm' in k['name'].lower()]
        perm_kernels = [k for k in kernels if 'permutation' in k['name'].lower()]
        pointwise_kernels = [k for k in kernels if 'pointwise' in k['name'].lower() or 'eltwise' in k['name'].lower()]
        pooling_kernels = [k for k in kernels if 'pooling' in k['name'].lower()]

        gemm_calls = sum(k['calls'] for k in gemm_kernels)
        perm_calls = sum(k['calls'] for k in perm_kernels)
        pw_calls = sum(k['calls'] for k in pointwise_kernels)
        pool_calls = sum(k['calls'] for k in pooling_kernels)
        other_calls = total_calls - gemm_calls - perm_calls - pw_calls - pool_calls

        lines.append(f'### 按功能分类')
        lines.append('')
        lines.append(f'| 类别 | Kernel 数 | 调用次数 | 占比 |')
        lines.append(f'|------|----------|---------|------|')
        lines.append(f'| GEMM (Tensor Core Conv) | {len(gemm_kernels)} | {gemm_calls} | {gemm_calls/total_calls*100:.1f}% |')
        lines.append(f'| Permutation (Transpose) | {len(perm_kernels)} | {perm_calls} | {perm_calls/total_calls*100:.1f}% |')
        lines.append(f'| Pointwise (激活函数) | {len(pointwise_kernels)} | {pw_calls} | {pw_calls/total_calls*100:.1f}% |')
        lines.append(f'| Pooling | {len(pooling_kernels)} | {pool_calls} | {pool_calls/total_calls*100:.1f}% |')
        lines.append(f'| Other | - | {other_calls} | {other_calls/total_calls*100:.1f}% |')
        lines.append('')

        # GPU 属性 (从 CSV 第一行提取)
        lines.append('### GPU 属性 (Orin)')
        lines.append('')
        lines.append(f'- **架构**: sm80 (Ampere), 2048 CUDA Cores')
        lines.append(f'- **L2 Cache**: 4 MB')
        lines.append(f'- **Max Blocks per SM**: 16')
        lines.append(f'- **Max Threads per SM**: 1536')
        lines.append(f'- **Max Warps per SM**: 48')
        lines.append(f'- **Max Registers per SM**: 65536')
        lines.append('')

        # 优化建议
        lines.append('### 优化建议')
        lines.append('')
        if perm_calls > gemm_calls * 0.3:
            lines.append(f'- **{perm_calls} 次 Permutation kernel (Transpose/Reformat)** 开销较大，建议通过算子融合或布局优化减少数据重排')
        for k in gemm_kernels[:3]:
            if k['warps_pct'] < 30:
                lines.append(f'- GEMM kernel `{k["name"][:40]}...` 的 Warp Occupancy 仅 {k["warps_pct"]:.0f}%，被寄存器 ({k["registers_per_thread"]:.0f}/thread) 限制。考虑调整 tile size 或使用更小的 block 配置')
                break
        lines.append(f'- Pointwise kernel 平均 {pointwise_kernels[0]["warps_pct"]:.0f}% occupancy' if pointwise_kernels else '- Pointwise kernel occupancy 正常')
        lines.append('')
    else:
        # CSV 存在但没有解析出数据
        if csv_generated:
            lines.append('CSV 文件已生成但无法解析为结构化数据。')
            lines.append('请使用 ncu-ui 打开 `report.ncu-rep` 查看详细信息。')
            lines.append('')
        else:
            lines.append('未获取到 kernel profiling 数据。')
            lines.append('')

    # ================================================================
    # 报告文件
    # ================================================================
    lines.append('## 报告文件')
    lines.append('')
    lines.append(f'- `report.ncu-rep` — Nsight Compute GUI (ncu-ui) 可打开的完整 profiling 文件')
    if csv_generated:
        lines.append(f'- `report.csv` — CSV 格式导出的指标数据')
    lines.append(f'- `analysis_report.md` — 本报告')
    lines.append('')

    with open(MD_REPORT, 'w') as f:
        f.write('\n'.join(lines))

    print(f'[NCU] 报告已保存到 {MD_REPORT}')


def main():
    ncu_success, ncu_version = run_ncu()

    # CSV 导出 (仅当 report.ncu-rep 存在时)
    csv_generated = dump_csv()

    # 解析 kernel 数据
    kernels = parse_kernel_csv() if csv_generated else []

    # 生成报告
    generate_report(ncu_success, ncu_version, kernels, csv_generated)

    print('[NCU] 完成。')


if __name__ == '__main__':
    main()
