# YOLOv8m TensorRT Engine 性能分析设计

## 概述

对 `weights/engines/yolov8m_fp16.engine` 进行 4 维度性能分析，在 `yolov8m_profiling/` 下完成。

- **平台**: Jetson Orin (ARM64, L4T 5.10)
- **环境**: conda py38
- **LLD 预加载**: `LD_PRELOAD=libopenblas.so.0:libgomp.so.1`
- **TensorRT**: 8.5.2.2
- **Nsight Systems**: 2023.2.4
- **Nsight Compute**: 2022.2.1

## 目录结构

```
yolov8m_profiling/
├── common/
│   ├── __init__.py
│   ├── data_source.py          # DummySource + CameraSource 统一接口
│   └── inference_workload.py   # 共享推理负载脚本（供 nsys/ncu 调用）
├── nsight_systems/
│   ├── run.py                  # nsys CLI 封装 + 报告生成
│   └── analysis_report.md      # 生成的 MD 分析报告
├── nsight_compute/
│   ├── run.py                  # ncu CLI 封装 + 报告生成
│   └── analysis_report.md
├── trt_profiler/
│   ├── run.py                  # trtexec 调用 + layer JSON 解析 + 报告生成
│   └── analysis_report.md
├── roofline/
│   ├── run.py                  # Roofline 模型计算 + 报告生成
│   └── analysis_report.md
├── run_all.sh                  # 一键运行全部 4 项分析
└── summary_report.md           # 4 项分析的汇总对比
```

## 数据源（`common/data_source.py`）

统一接口 `DataSource`，两个实现：

```python
class DataSource(ABC):
    def get_input(self) -> dict[str, np.ndarray]: ...
    def get_metadata(self) -> dict: ...

class DummySource(DataSource):
    # 随机生成 640x640 RGB float32，立即可用
    def __init__(self, input_name='images', shape=(1,3,640,640), dtype=np.float32): ...

class CameraSource(DataSource):
    # 占位实现，等待真实摄像头。当前抛 NotImplementedError
    def __init__(self, device_id=0, input_name='images', shape=(1,3,640,640), dtype=np.float32): ...
```

调用方不感知使用的是哪个 source，无 if/else 分支。

## 共享推理负载（`common/inference_workload.py`）

独立脚本，供 nsys/ncu 作为目标进程。命令行参数控制行为：

```bash
python3 inference_workload.py \
  --engine weights/engines/yolov8m_fp16.engine \
  --data dummy \
  --warmup 10 \
  --iters 100
```

流程：argparse → 选择 DataSource → TrtEngine 加载 → warmup → 循环推理 → 打印统计。

不涉及任何 profiler API，不生成报告文件。

## 任务 1: Nsight Systems（`nsight_systems/`）

- 调用 `nsys profile`，生成 `report.qdrep` + `report.sqlite`
- 读取 sqlite 提取 CUDA kernel 数据
- 报告内容：时间线摘要、Top-15 kernel 排名、H2D/D2H 开销

## 任务 2: Nsight Compute（`nsight_compute/`）

- 调用 `ncu --set full`，生成 `report.ncu-rep`
- 导出 CSV 并解析
- 报告内容：occupancy、SM 利用率、内存访问模式、warp 发散、compute vs memory bound 定性分析

## 任务 3: TensorRT Profiler（`trt_profiler/`）

- 调用 `trtexec --dumpProfile --dumpLayerInfo --profilingVerbosity=detailed`
- 解析逐层输出 JSON
- 报告内容：
  - 逐层表：名称 | 类型 | 输入 shape | 输出 shape | 精度 | 耗时(ms) | 占比(%)
  - 按类型汇总：Conv/BN/ReLU/Transpose 等各自总耗时
  - 按精度汇总：FP16 vs FP32
  - Top-10 最耗时层分析

## 任务 4: Roofline Model（`roofline/`）

- 纯分析计算，不需要 GPU 推理
- 从 TRT Profiler 逐层数据获取每层 FLOPs 和内存传输量
- 使用 Orin GPU 理论峰值参数
- 计算每层算术强度，判定 compute-bound vs memory-bound
- 报告内容：理论参数、脊点、逐层分类、优化建议

## 运行方式

所有脚本在 py38 conda 环境下运行：

```bash
conda activate py38
cd yolov8m_profiling
bash run_all.sh
```

或单独运行某个分析：

```bash
cd yolov8m_profiling/trt_profiler
LD_PRELOAD=/home/ssd/anaconda3/envs/py38/lib/libopenblas.so.0:... \
  python3 run.py
```
