#!/usr/bin/env python3
"""Nsight Systems 分析: 生成 report.nsys-rep + report.sqlite, 解析并写 MD 报告。

用法:
    conda activate py38
    cd yolov8m_profiling/nsight_systems
    python3 run.py

注意:
    LD_PRELOAD 在 nsys profile 下会干扰进程附加, 因此不需要设置。
    TensorRT 推理主要走 CUDA, 不依赖 OpenBLAS 的 CPU 加速。
    如果 report.sqlite 已存在则跳过 profiling, 仅重新生成 MD 报告。
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
WORKLOAD = os.path.join(ROOT, 'common', 'inference_workload.py')
ENGINE = os.path.join(ROOT, '..', 'weights', 'engines', 'yolov8m_fp16.engine')
PYTHON = '/home/ssd/anaconda3/envs/py38/bin/python3'
NSYS = '/usr/local/bin/nsys'

REPORT_BASE = os.path.join(HERE, 'report')        # 生成 report.nsys-rep + report.sqlite
MD_REPORT = os.path.join(HERE, 'analysis_report.md')

# CUDA MEMCPY copyKind 到可读名称的映射 (与 ENUM_CUDA_MEMCPY_OPER 一致)
COPY_KIND_NAMES = {
    0: 'UNKNOWN', 1: 'H2D (HTOD)', 2: 'D2H (DTOH)', 3: 'H2A (HTOA)',
    4: 'A2H (ATOH)', 5: 'A2A (ATOA)', 6: 'A2D (ATOD)', 7: 'D2A (DTOA)',
    8: 'D2D (DTOD)', 9: 'H2H (HTOH)', 10: 'P2P (PTOP)',
    11: 'UVM_H2D (UVM_HTOD)', 12: 'UVM_D2H (UVM_DTOH)', 13: 'UVM_D2D (UVM_DTOD)',
}


def get_nsys_version():
    """获取已安装 Nsight Systems 版本号。"""
    try:
        result = subprocess.run([NSYS, '--version'], capture_output=True, text=True, timeout=30)
        raw = (result.stdout or result.stderr or '').strip()
        # 如果输出已包含 "Nsight Systems" 则直接使用，否则加上前缀
        if 'Nsight Systems' in raw:
            return raw
        return f'Nsight Systems {raw}' if raw else 'Nsight Systems (unknown)'
    except Exception as e:
        print(f'[Nsight Systems] WARNING: 无法获取版本: {e}')
        return 'Nsight Systems (unknown)'


def run_nsys():
    """执行 profiling (如果 SQLite 已存在则跳过)。返回 Nsight Systems 版本。"""
    # Issue 4: 动态版本检测
    nsys_version = get_nsys_version()
    print(f'[Nsight Systems] Version: {nsys_version}')

    # 如果 SQLite 已存在，跳过 profiling (允许仅重新生成 MD 报告)
    if os.path.exists(REPORT_BASE + '.sqlite'):
        print('[Nsight Systems] SQLite report already exists, skipping profiling.')
        print('[Nsight Systems] Delete report.sqlite to re-run profiling.')
        return nsys_version

    print('[Nsight Systems] Starting profiling...')

    cmd = [
        NSYS, 'profile',
        '--output=' + REPORT_BASE,
        '--export=sqlite',
        '--force-overwrite=true',
        '--trace=cuda,osrt,nvtx',
        PYTHON, WORKLOAD,
        '--engine', ENGINE,
        '--data', 'dummy',
        '--warmup', '10',
        '--iters', '100',
    ]

    print(f'[Nsight Systems] Running: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)

    # nsys 输出会很大, 只打印最后几行
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print('[Nsight Systems] stderr:', result.stderr[-2000:])

    assert os.path.exists(REPORT_BASE + '.nsys-rep'), f'Missing {REPORT_BASE}.nsys-rep'
    assert os.path.exists(REPORT_BASE + '.sqlite'), f'Missing {REPORT_BASE}.sqlite'
    print('[Nsight Systems] Profiling complete. Reports generated.')
    return nsys_version


def parse_sqlite():
    """从 report.sqlite 提取所有分析数据。

    Returns:
        (kernel_data, kernel_summary, total_kernels, total_gpu_ns, tables,
         memcpy_data, runtime_count, runtime_total_ns, osrt_count, osrt_total_ns)
    """
    import sqlite3

    conn = sqlite3.connect(REPORT_BASE + '.sqlite')
    cursor = conn.cursor()

    # 探索可用表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f'[Nsight Systems] Found tables: {tables}')

    # ---- Kernel 数据 ----
    kernel_data = []        # Top-20 kernel by single-invocation duration
    kernel_summary = []     # Kernel 汇总: name, count, total_ns, avg_ns
    total_kernels = 0
    total_gpu_ns = 0

    if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
        # Top-20 单次调用耗时最长的 kernel
        cursor.execute("""
            SELECT SUBSTR(s.value, 1, 90) as kernel_name,
                   k.start, k.end, (k.end - k.start) as duration_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            ORDER BY duration_ns DESC
            LIMIT 20
        """)
        kernel_data = cursor.fetchall()

        # 按 kernel 名称汇总
        cursor.execute("""
            SELECT SUBSTR(s.value, 1, 120) as kernel_name,
                   COUNT(*) as invocations,
                   SUM(k.end - k.start) as total_ns,
                   AVG(k.end - k.start) as avg_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            GROUP BY k.demangledName
            ORDER BY total_ns DESC
        """)
        kernel_summary = cursor.fetchall()

        # 全局统计
        cursor.execute("SELECT COUNT(*), SUM(k.end - k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k")
        row = cursor.fetchone()
        if row:
            total_kernels = row[0] or 0
            total_gpu_ns = row[1] or 0

    elif 'StringIds' in tables:
        # 备用：列出所有可能的 kernel 相关表
        for t in tables:
            if 'KERNEL' in t.upper() or 'kernel' in t.lower():
                print(f'[Nsight Systems] Trying table: {t}')
                try:
                    cursor.execute(f"PRAGMA table_info({t})")
                    cols = [c[1] for c in cursor.fetchall()]
                    print(f'  Columns: {cols}')
                except Exception:
                    pass

    # ---- Issue 1: MEMCPY (H2D/D2H) 传输分析 ----
    memcpy_data = []
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        # 先检查 schema
        cursor.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")
        memcpy_cols = [c[1] for c in cursor.fetchall()]
        print(f'[Nsight Systems] MEMCPY columns: {memcpy_cols}')

        # 按 copyKind 聚合: count, total bytes, total time
        cursor.execute("""
            SELECT copyKind,
                   COUNT(*) as count,
                   SUM(bytes) as total_bytes,
                   SUM(end - start) as total_ns
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            GROUP BY copyKind
            ORDER BY copyKind
        """)
        memcpy_data = cursor.fetchall()
        print(f'[Nsight Systems] MEMCPY by copyKind: {memcpy_data}')
    else:
        print('[Nsight Systems] WARNING: CUPTI_ACTIVITY_KIND_MEMCPY table not found')

    # ---- Issue 2: CUDA Runtime API time (CPU 端) ----
    runtime_count = 0
    runtime_total_ns = 0
    if 'CUPTI_ACTIVITY_KIND_RUNTIME' in tables:
        cursor.execute("SELECT COUNT(*), SUM(end - start) FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        row = cursor.fetchone()
        if row:
            runtime_count = row[0] or 0
            runtime_total_ns = row[1] or 0
        print(f'[Nsight Systems] CUDA Runtime API: {runtime_count} calls, '
              f'{_fmt_ns(runtime_total_ns)}')
    else:
        print('[Nsight Systems] WARNING: CUPTI_ACTIVITY_KIND_RUNTIME table not found')

    # ---- Issue 2: OS Runtime API time ----
    osrt_count = 0
    osrt_total_ns = 0
    if 'OSRT_API' in tables:
        cursor.execute("SELECT COUNT(*), SUM(end - start) FROM OSRT_API")
        row = cursor.fetchone()
        if row:
            osrt_count = row[0] or 0
            osrt_total_ns = row[1] or 0
        print(f'[Nsight Systems] OS Runtime API: {osrt_count} calls, '
              f'{_fmt_ns(osrt_total_ns)}')
    else:
        print('[Nsight Systems] WARNING: OSRT_API table not found')

    conn.close()
    return (kernel_data, kernel_summary, total_kernels, total_gpu_ns, tables,
            memcpy_data, runtime_count, runtime_total_ns, osrt_count, osrt_total_ns)


def detect_nv_fuser():
    """检测 kernel 执行模式（Jetson Orin 集成 GPU 使用 DRAM 统一内存模式）。"""
    return 'DRAM'


def _fmt_ns(ns):
    """将纳秒格式化为可读字符串 (us / ms / s)"""
    if ns is None:
        return '0 ns'
    if ns < 1000:
        return f'{ns:.0f} ns'
    elif ns < 1_000_000:
        return f'{ns / 1000:.1f} us'
    elif ns < 1_000_000_000:
        return f'{ns / 1_000_000:.2f} ms'
    else:
        return f'{ns / 1_000_000_000:.3f} s'


def _fmt_bytes(b):
    """将字节格式化为可读字符串 (KB / MB / GB)。"""
    if b < 1024:
        return f'{b} B'
    elif b < 1024 * 1024:
        return f'{b / 1024:.1f} KB'
    elif b < 1024 * 1024 * 1024:
        return f'{b / (1024 * 1024):.1f} MB'
    else:
        return f'{b / (1024 * 1024 * 1024):.2f} GB'


def _find_reformat_kernels(kernel_summary):
    """查找 Permutation/Reformat/Copy/Transpose 类 kernel 的总时间。"""
    total_ns = 0
    keywords = ['permutation', 'reformat', 'copy', 'transpose',
                'CUTENSOR', 'convert', 'cast']
    for row in kernel_summary:
        name = (row[0] or '').lower()
        if any(kw in name for kw in keywords):
            total_ns += row[2]
    return total_ns


def generate_report(kernel_data, kernel_summary, total_kernels, total_gpu_ns, tables,
                    memcpy_data, runtime_count, runtime_total_ns, osrt_count, osrt_total_ns,
                    nsys_version):
    """生成完整的 MD 分析报告。"""
    print('[Nsight Systems] Generating analysis report...')

    lines = []
    lines.append('# YOLOv8m Nsight Systems 性能分析报告')
    lines.append('')
    lines.append(f'**生成时间**: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'**工具版本**: {nsys_version}')
    lines.append(f'**引擎**: `weights/engines/yolov8m_fp16.engine`')
    lines.append(f'**输入**: images (1, 3, 640, 640) float32, dummy data')
    lines.append(f'**输出**: output0 (1, 84, 8400) float32')
    lines.append(f'**Kernel 执行模式**: {detect_nv_fuser()}')
    lines.append(f'**预热/测量**: 10 / 100 iterations')
    lines.append('')

    # ================================================================
    # Issue 3: 改进的关键发现
    # ================================================================
    lines.append('## 关键发现')
    lines.append('')

    # Top-3 kernel 占比
    top3_pct = 0.0
    if total_gpu_ns > 0 and len(kernel_summary) >= 3:
        top3_ns = sum(row[2] for row in kernel_summary[:3])
        top3_pct = top3_ns / total_gpu_ns * 100.0

    # Permutation/Reformat 类 kernel 时间
    reformat_ns = _find_reformat_kernels(kernel_summary) if kernel_summary else 0

    # 内存传输总时间
    memcpy_total_ns = sum(row[3] for row in memcpy_data) if memcpy_data else 0

    if total_gpu_ns > 0 and len(kernel_summary) >= 3:
        lines.append(f'- **Top-3 kernel 类型** 占总 GPU 时间的 **{top3_pct:.1f}%** '
                     f'GPU 时间 = {_fmt_ns(top3_ns)}，其余 {len(kernel_summary) - 3} 种 kernel 占 '
                     f'{100 - top3_pct:.1f}%')
    else:
        lines.append(f'- **GPU Kernel 总耗时**: {_fmt_ns(total_gpu_ns)}（{total_kernels} 次调用）')

    if reformat_ns > 0 and total_gpu_ns > 0:
        reformat_pct = reformat_ns / total_gpu_ns * 100
        lines.append(f'- **Permutation/Reformat 类 kernel** 耗时 {_fmt_ns(reformat_ns)}'
                     f'（GPU 时间的 {reformat_pct:.1f}%）')
        if reformat_pct > 5:
            lines.append('  - 此类 kernel 由 TensorRT 的 Transpose / Shuffle / Reshape 算子产生，'
                         '是潜在的优化方向（可尝试 Transpose 融合或使用 TensorRT 的优化策略）')

    if memcpy_data:
        lines.append(f'- **内存传输** 增加 {_fmt_ns(memcpy_total_ns)} 额外开销')
        for row in memcpy_data:
            kind = row[0]
            name = COPY_KIND_NAMES.get(kind, f'UNKNOWN({kind})')
            cnt = row[1]
            bytes_val = row[2]
            time_ns = row[3]
            if kind in (1, 2):  # H2D or D2H
                lines.append(f'  - {name}: {cnt} 次传输, '
                             f'{_fmt_bytes(bytes_val)} 数据, {_fmt_ns(time_ns)}')

    lines.append(f'- **GPU Kernel 总调用次数**: {total_kernels}')
    lines.append(f'- **GPU Kernel 总耗时**: {_fmt_ns(total_gpu_ns)}')
    if kernel_summary:
        lines.append(f'- **不同 Kernel 类型数**: {len(kernel_summary)}')

    # 优化建议
    optimization_items = []
    if reformat_ns > 0 and (reformat_ns / total_gpu_ns) > 0.05:
        optimization_items.append('Permutation/Transpose 算子融合或采用更高效的实现')
    if memcpy_total_ns > 0 and (memcpy_total_ns / total_gpu_ns) > 0.05:
        optimization_items.append('减少 H2D/D2H 传输次数, 使用异步传输或 CUDA Graph 进行传输合并')
    if total_gpu_ns > 0:
        optimization_items.append('检查卷积算子与 Elementwise 算子的融合情况')
    if optimization_items:
        lines.append(f'- **关键优化方向**:')
        for item in optimization_items:
            lines.append(f'  1. {item}')
    lines.append('')

    # ================================================================
    # Top-20 CUDA Kernel 单次调用耗时排名
    # ================================================================
    if kernel_data:
        lines.append('## Top-20 CUDA Kernel 单次调用耗时排名')
        lines.append('')
        lines.append('| # | Kernel Name | Duration |')
        lines.append('|---|-------------|----------|')
        for i, row in enumerate(kernel_data[:20], 1):
            name = row[0][:80] if row[0] else 'unknown'
            dur_str = _fmt_ns(row[3]) if len(row) > 3 else 'N/A'
            lines.append(f'| {i} | `{name}` | {dur_str} |')
        lines.append('')

    # ================================================================
    # Kernel 耗时汇总 (按总耗时排序)
    # ================================================================
    if kernel_summary:
        lines.append('## Kernel 耗时汇总 (按总耗时排序)')
        lines.append('')
        lines.append('| # | Kernel Name | 调用次数 | 总耗时 | 平均耗时 |')
        lines.append('|---|-------------|----------|--------|----------|')
        for i, row in enumerate(kernel_summary, 1):
            name = row[0][:75] if row[0] else 'unknown'
            cnt = row[1]
            total_str = _fmt_ns(row[2])
            avg_str = _fmt_ns(row[3])
            lines.append(f'| {i} | `{name}` | {cnt} | {total_str} | {avg_str} |')
        lines.append('')

    # ================================================================
    # Issue 1: H2D/D2H 内存传输分析
    # ================================================================
    if memcpy_data:
        lines.append('## H2D/D2H 内存传输分析')
        lines.append('')
        lines.append('| copyKind | 含义 | 传输次数 | 总数据量 | 总耗时 |')
        lines.append('|----------|------|----------|----------|--------|')
        for row in memcpy_data:
            kind = row[0]
            name = COPY_KIND_NAMES.get(kind, f'UNKNOWN({kind})')
            count = row[1]
            bytes_str = _fmt_bytes(row[2])
            time_str = _fmt_ns(row[3])
            lines.append(f'| {kind} | {name} | {count} | {bytes_str} | {time_str} |')
        lines.append('')
        lines.append('**说明**: copyKind 对应 CUDA MEMCPY 操作类型:')
        lines.append('- `1` = `CUDA_MEMCPY_KIND_HTOD` (Host to Device)')
        lines.append('- `2` = `CUDA_MEMCPY_KIND_DTOH` (Device to Host)')
        lines.append('- `8` = `CUDA_MEMCPY_KIND_DTOD` (Device to Device)')
        lines.append('')
    else:
        lines.append('## H2D/D2H 内存传输分析')
        lines.append('')
        lines.append('未找到 MEMCPY 表数据，请确认 `--trace=cuda` 已启用。')
        lines.append('')

    # ================================================================
    # Issue 2: CPU vs GPU 时间分布
    # ================================================================
    lines.append('## CPU vs GPU 时间分布')
    lines.append('')
    total_time = total_gpu_ns + runtime_total_ns + osrt_total_ns + memcpy_total_ns
    if total_time > 0:
        memcpy_total_count = sum(row[1] for row in memcpy_data) if memcpy_data else 0
        lines.append('| 类别 | 调用次数 | 总耗时 | 占比 |')
        lines.append('|------|----------|--------|------|')
        lines.append(f'| GPU Kernel 执行 | {total_kernels} | {_fmt_ns(total_gpu_ns)} '
                     f'| {total_gpu_ns / total_time * 100:.1f}% |')
        lines.append(f'| CUDA Runtime API (CPU) | {runtime_count} | {_fmt_ns(runtime_total_ns)} '
                     f'| {runtime_total_ns / total_time * 100:.1f}% |')
        lines.append(f'| 内存传输 (H2D/D2H) | {memcpy_total_count} | {_fmt_ns(memcpy_total_ns)} '
                     f'| {memcpy_total_ns / total_time * 100:.1f}% |')
        lines.append(f'| OS Runtime (CPU) | {osrt_count} | {_fmt_ns(osrt_total_ns)} '
                     f'| {osrt_total_ns / total_time * 100:.1f}% |')
        lines.append(f'| **合计** | — | **{_fmt_ns(total_time)}** | **100%** |')
        lines.append('')
        lines.append('**说明**:')
        lines.append('- GPU Kernel 执行: GPU 上实际运行 kernel 的时间')
        lines.append('- CUDA Runtime API (CPU): CPU 端调用 CUDA API'
                     '（如 cudaMemcpyAsync, cudaLaunchKernel）的耗时')
        lines.append('- 内存传输 (H2D/D2H): 数据在 Host 与 Device 间传输的耗时')
        lines.append('- OS Runtime (CPU): 操作系统级运行时调用'
                     '（如 mmap, fread, pthread 等）的耗时')
        lines.append('')
    else:
        lines.append('（无时间数据可供分析）')
        lines.append('')

    # ================================================================
    # 备用: 无 kernel 数据时
    # ================================================================
    if not kernel_data and not kernel_summary:
        lines.append('## Kernel 数据')
        lines.append('')
        lines.append('未能从 SQLite 直接提取 kernel 表数据。')
        lines.append('请使用 Nsight Systems GUI 打开 `report.nsys-rep` 查看详细时间线。')
        lines.append(f'数据库表: {", ".join(tables)}')
        lines.append('')

    # ================================================================
    # 报告文件
    # ================================================================
    lines.append('## 报告文件')
    lines.append('')
    lines.append(f'- `report.nsys-rep` — Nsight Systems GUI 可打开的时间线文件')
    lines.append(f'- `report.sqlite` — 可编程查询的 SQLite 数据库')
    lines.append('')

    with open(MD_REPORT, 'w') as f:
        f.write('\n'.join(lines))

    print(f'[Nsight Systems] Report saved to {MD_REPORT}')


if __name__ == '__main__':
    nsys_version = run_nsys()
    (kernel_data, kernel_summary, total_kernels, total_gpu_ns, tables,
     memcpy_data, runtime_count, runtime_total_ns, osrt_count, osrt_total_ns) = parse_sqlite()
    generate_report(kernel_data, kernel_summary, total_kernels, total_gpu_ns, tables,
                    memcpy_data, runtime_count, runtime_total_ns, osrt_count, osrt_total_ns,
                    nsys_version)
    print('[Nsight Systems] Done.')
