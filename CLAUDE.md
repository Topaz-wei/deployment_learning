# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Jetson Orin 上的 TensorRT 模型部署学习仓库。代码从主项目 (`/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/`) 中提取，用于学习将 ONNX/PyTorch 模型迁移到 TensorRT FP16 engine。

## 运行环境

- **平台**: NVIDIA Jetson Orin (ARM64, L4T kernel 5.10)
- **Python**: conda 环境 `py38`，路径 `/home/ssd/anaconda3/envs/py38`
- **关键库**: TensorRT（Jetson 预装，`import tensorrt`）、OpenCV with CUDA (`cv2`)、PyTorch 1.x（Jetson 优化版）、onnxruntime（仅 CPU）

## 运行方式

所有脚本需在 `py38` conda 环境下运行，建议预加载 OpenBLAS/GNU OpenMP：

```bash
conda activate py38
LD_PRELOAD=/home/ssd/anaconda3/envs/py38/lib/libopenblas.so.0:/home/ssd/anaconda3/envs/py38/lib/libgomp.so.1 \
  python3 <script>.py
```

## 所有回答均用中文