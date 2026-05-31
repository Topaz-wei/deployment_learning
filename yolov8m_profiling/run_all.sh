#!/bin/bash
# YOLOv8m 一键性能分析脚本
# 用法: conda activate py38 && cd yolov8m_profiling && bash run_all.sh

set -e

CONDA_LIB="/home/ssd/anaconda3/envs/py38/lib"
export LD_PRELOAD="${CONDA_LIB}/libopenblas.so.0:${CONDA_LIB}/libgomp.so.1"
PYTHON="/home/ssd/anaconda3/envs/py38/bin/python3"
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  YOLOv8m Performance Profiling Suite"
echo "  Platform: Jetson Orin | Precision: FP16"
echo "=========================================="
echo ""

# 1. TRT Profiler (先跑，生成逐层数据供 Roofline 使用)
echo "[1/4] TensorRT Profiler — 逐层性能分析"
echo "------------------------------------------"
cd "$ROOT/trt_profiler"
$PYTHON run.py
echo ""

# 2. Roofline Model (依赖 TRT Profiler 输出)
echo "[2/4] Roofline Model — 计算/内存瓶颈分析"
echo "------------------------------------------"
cd "$ROOT/roofline"
$PYTHON run.py
echo ""

# 3. Nsight Systems
echo "[3/4] Nsight Systems — 系统级时间线分析"
echo "------------------------------------------"
cd "$ROOT/nsight_systems"
$PYTHON run.py
echo ""

# 4. Nsight Compute
echo "[4/4] Nsight Compute — Kernel 级详细分析"
echo "------------------------------------------"
cd "$ROOT/nsight_compute"
$PYTHON run.py
echo ""

echo "=========================================="
echo "  全部分析完成！"
echo "=========================================="
echo ""
echo "报告文件:"
echo "  trt_profiler/analysis_report.md"
echo "  trt_profiler/layer_profile.json"
echo "  trt_profiler/layer_info.json"
echo "  roofline/analysis_report.md"
echo "  nsight_systems/analysis_report.md"
echo "  nsight_systems/report.nsys-rep"
echo "  nsight_systems/report.sqlite"
echo "  nsight_compute/analysis_report.md"
echo "  nsight_compute/report.ncu-rep"
echo ""
