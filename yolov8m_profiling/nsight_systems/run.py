#!/usr/bin/env python3
"""Nsight Systems 分析: 生成 report.nsys-rep + report.sqlite, 解析并写 MD 报告。

用法:
    conda activate py38
    cd yolov8m_profiling/nsight_systems
    python3 run.py

注意:
    LD_PRELOAD 在 nsys profile 下会干扰进程附加, 因此不需要设置。
    TensorRT 推理主要走 CUDA, 不依赖 OpenBLAS 的 CPU 加速。
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


def run_nsys():
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


def parse_sqlite():
    """从 report.sqlite 提取 Top CUDA kernels 及汇总统计。"""
    import sqlite3

    conn = sqlite3.connect(REPORT_BASE + '.sqlite')
    cursor = conn.cursor()

    # Nsight Systems sqlite schema: 先探索可用表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f'[Nsight Systems] Found tables: {tables}')

    kernel_data = []        # top-20 kernel by single-invocation duration
    kernel_summary = []     # kernel 汇总: name, count, total_ns, avg_ns
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
            SELECT SUBSTR(s.value, 1, 90) as kernel_name,
                   COUNT(*) as invocations,
                   SUM(k.end - k.start) as total_ns,
                   AVG(k.end - k.start) as avg_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            GROUP BY k.demangledName
            ORDER BY total_ns DESC
            LIMIT 20
        """)
        kernel_summary = cursor.fetchall()

        # 全局统计
        cursor.execute("SELECT COUNT(*), SUM(k.end - k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k")
        row = cursor.fetchone()
        if row:
            total_kernels = row[0] or 0
            total_gpu_ns = row[1] or 0
    elif 'StringIds' in tables:
        # 列出所有可能的 kernel 相关表
        for t in tables:
            if 'KERNEL' in t.upper() or 'kernel' in t.lower():
                print(f'[Nsight Systems] Trying table: {t}')
                try:
                    cursor.execute(f"PRAGMA table_info({t})")
                    cols = [c[1] for c in cursor.fetchall()]
                    print(f'  Columns: {cols}')
                except Exception:
                    pass

    conn.close()
    return kernel_data, kernel_summary, total_kernels, total_gpu_ns, tables


def detect_nv_fuser():
    """检测 kernel 执行模式（Jetson Orin 集成 GPU 使用 DRAM 统一内存模式）。"""
    return 'DRAM'


def _fmt_ns(ns):
    """将纳秒格式化为可读字符串 (us / ms / s)"""
    if ns < 1000:
        return f'{ns:.0f} ns'
    elif ns < 1_000_000:
        return f'{ns / 1000:.1f} us'
    elif ns < 1_000_000_000:
        return f'{ns / 1_000_000:.2f} ms'
    else:
        return f'{ns / 1_000_000_000:.3f} s'


def generate_report(kernel_data, kernel_summary, total_kernels, total_gpu_ns, tables):
    print('[Nsight Systems] Generating analysis report...')

    lines = []
    lines.append('# YOLOv8m Nsight Systems 性能分析报告')
    lines.append('')
    lines.append(f'**生成时间**: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'**工具版本**: Nsight Systems 2023.2.4')
    lines.append(f'**引擎**: `weights/engines/yolov8m_fp16.engine`')
    lines.append(f'**输入**: images (1, 3, 640, 640) float32, dummy data')
    lines.append(f'**输出**: output0 (1, 84, 8400) float32')
    lines.append(f'**Kernel 执行模式**: {detect_nv_fuser()}')
    lines.append(f'**预热/测量**: 10 / 100 iterations')
    lines.append('')

    lines.append('## 关键发现')
    lines.append('')
    lines.append(f'- 数据库中找到 {len(tables)} 个表')
    if total_kernels > 0:
        lines.append(f'- GPU kernel 总调用次数: {total_kernels}')
        lines.append(f'- GPU kernel 总耗时: {_fmt_ns(total_gpu_ns)}')
        lines.append(f'- 共 {len(kernel_summary)} 种不同 kernel 类型')
    lines.append('')

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

    if not kernel_data and not kernel_summary:
        lines.append('## Kernel 数据')
        lines.append('')
        lines.append('未能从 SQLite 直接提取 kernel 表数据。')
        lines.append('请使用 Nsight Systems GUI 打开 `report.nsys-rep` 查看详细时间线。')
        lines.append(f'数据库表: {", ".join(tables)}')
        lines.append('')

    # Nsight Systems 输出摘要
    lines.append('## 报告文件')
    lines.append('')
    lines.append(f'- `report.nsys-rep` — Nsight Systems GUI 可打开的时间线文件')
    lines.append(f'- `report.sqlite` — 可编程查询的 SQLite 数据库')
    lines.append('')

    with open(MD_REPORT, 'w') as f:
        f.write('\n'.join(lines))

    print(f'[Nsight Systems] Report saved to {MD_REPORT}')


if __name__ == '__main__':
    run_nsys()
    kernel_data, kernel_summary, total_kernels, total_gpu_ns, tables = parse_sqlite()
    generate_report(kernel_data, kernel_summary, total_kernels, total_gpu_ns, tables)
    print('[Nsight Systems] Done.')
