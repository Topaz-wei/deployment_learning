# Transpose 插入对 ONNX 推理速度影响实验

## 目标

通过 `onnx.helper` 在 YOLOv8m ONNX 图中修改/插入 Transpose 节点，系统性测量 Transpose 在不同位置、不同数量、不同分辨率下对 ONNX Runtime（CPU）和 TensorRT 推理速度的影响。

## 环境

- **平台**: Jetson Orin ARM64, conda py38
- **模型**: YOLOv8m ONNX (325 节点, 输入 images [1,3,640,640] NCHW, 输出 output0 [1,84,8400])
- **后端**: ONNX Runtime CPU + TensorRT FP16
- **输入**: 模拟随机图像 (np.random.randn)，无需摄像头

## 目录结构

```
analyze_transpose/
├── yolov8m.onnx
├── common/
│   ├── __init__.py
│   ├── benchmark.py          # 统一基准测试 (ORT + TRT)
│   └── model_utils.py        # onnx.helper 图操作工具
├── exp01_baseline/run.py
├── exp02_input_transpose/
│   ├── modify_model.py
│   └── run.py
├── exp03_output_transpose/
│   ├── modify_model.py
│   └── run.py
├── exp04_dual_transpose/
│   ├── modify_model.py
│   └── run.py
├── exp05_mid_transpose/
│   ├── modify_model.py
│   └── run.py
├── exp06_multi_transpose/
│   ├── modify_model.py
│   └── run.py
├── exp07_ort_opt_transpose/run.py
├── exp08_transpose_conv_fusion/run.py
├── exp09_resolution/run.py
├── modified_models/            # 生成的所有修改版 ONNX
├── results/                    # 汇总 JSON
└── run_all.sh
```

## 9 个实验详情

### 实验 1: Baseline 基准测试
- 原始 YOLOv8m ONNX，无修改
- 测量 ORT CPU (多种优化级别) 和 TRT FP16 下的 latency
- 输出: mean/min/max/std/p50 latency, FPS

### 实验 2: 输入层 Transpose 开销
- 在 images 后、首层 Conv 前插入 Transpose(perm=[0,2,3,1])，NCHW→NHWC
- 测量与 Baseline 的时间差

### 实验 3: 输出层 Transpose 开销
- 在模型最后输出 (Concat_5) 前插入 Transpose(perm=[0,3,1,2])，NHWC→NCHW
- 测量输出延迟变化

### 实验 4: 输入+输出双 Transpose
- 同时插入输入和输出 Transpose，保持外部接口仍为 NCHW
- 测量端到端 latency

### 实验 5: 中间层插入 Transpose
- 在第 20 个 Conv 节点后插入 Transpose (NCHW→NHWC)
- 观察对该层及后续层的影响

### 实验 6: 多层 Transpose 累积效应
- 在 Backbone 的 4 个 Stage 之间各插入 1 个 Transpose
- 测量边际延迟

### 实验 7: ORT 图优化对 Transpose 的消除
- 原始模型，切换 ORT SessionOptions 优化级别: DISABLE_ALL / BASIC / EXTENDED / ALL
- 观察节点数量变化、latency 变化

### 实验 8: Transpose 与 Conv 的融合边界
- 构建紧邻 Conv 前后的 Transpose，用 Netron 可视化检查融合
- 并跑 ORT 验证性能

### 实验 9: 不同输入分辨率
- 原始模型，三种分辨率: 320x320 / 640x640 / 1280x1280
- 测量每像素 Transpose 开销

## 统一测试协议

- 预热: 10 iterations
- 测量: 100 iterations
- 指标: mean/min/max/std/p50 latency (ms), FPS
- 每个实验输出 `results/expXX_<name>.json`

## 实验后扩展

- 所有实验用模拟图像，后续接入真实摄像头时复用相同修改模型
- results/ 下的汇总脚本可生成 latex/markdown 对比表
