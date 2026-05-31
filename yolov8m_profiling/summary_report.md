# YOLOv8m 性能分析汇总报告

**生成时间**: 2026-05-31
**平台**: NVIDIA Jetson Orin (Ampere GA10B, 2048 CUDA Cores)
**引擎**: yolov8m_fp16.engine | 输入 (1, 3, 640, 640) FP16 | 输出 (1, 84, 8400) FP16
**运行环境**: ARM64 L4T kernel 5.10 | TensorRT v8502 | Python 3.8

---

## 1. 分析工具执行状态

| 工具 | 状态 | 说明 |
|------|------|------|
| TensorRT Profiler | 已完成 | 逐层性能分析，191 层数据已采集 |
| Roofline Model | 已完成 | 基于 TRT Profiler 数据的计算/内存瓶颈分析 |
| Nsight Systems | 已完成 | 系统级 CUDA Kernel 时间线和 CPU/GPU 分布 |
| Nsight Compute | 不可用 | Jetson Orin 集成 GPU 不支持 kernel 级 profiling |

---

## 2. 关键性能指标

| 指标 | 值 |
|------|-----|
| **Mean Latency (逐层累计)** | 16.01 ms (600 次迭代平均) |
| **FPS** | 62.5 |
| **GPU Kernel Mean Latency (Nsight Systems)** | 14.48 ms (100 次迭代平均) |
| **总层数** | 191 |
| **活跃层数 (含计算)** | 154 |
| **总估算 FLOPs** | 204.25 GFLOPs |
| **总估算 Memory Transfer** | 1.52 GB |
| **总体 Arithmetic Intensity** | 134.0 FLOP/byte |

### Top-3 最耗时层

| 排名 | 层名 | 类型 | 耗时 (ms) | 占比 |
|------|------|------|-----------|------|
| 1 | `/model.22/dfl/conv/Conv` | Conv | 1183.14 | 12.32% |
| 2 | `/model.22/cv3.0/cv3.0.1/conv/Conv + PWN(...)` | Conv | 475.78 | 4.95% |
| 3 | `/model.2/cv2/conv/Conv + PWN(...)` | Conv | 355.49 | 3.70% |

**Top-3 累计占比**: 20.97%

---

## 3. 层类型分解 (TensorRT Profiler)

| 类型 | 层数 | 总耗时 (ms) | 耗时占比 |
|------|------|-------------|----------|
| Conv | 78 | 7019.91 | 73.07% |
| Reformat | 69 | 1078.99 | 11.23% |
| PointWise | 21 | 553.01 | 5.76% |
| FusedParallel | 3 | 475.72 | 4.95% |
| Softmax | 1 | 213.97 | 2.23% |
| Reshape | 5 | 98.44 | 1.02% |
| Shuffle | 3 | 56.86 | 0.59% |
| Resize | 2 | 44.16 | 0.46% |
| MaxPool | 3 | 41.54 | 0.43% |
| Other | 6 | 23.94 | 0.25% |

### 关键发现

- **Conv 层是绝对主力**: 78 层卷积占总耗时 73%，其中 `/model.22/dfl/conv/Conv` 单层占 12.3%，是最大的单一瓶颈。
- **Reformat 开销显著**: 69 层 Reformat 耗时占比 11.23%（超过 5% 警戒线），主要由 TensorRT 的 NHWC/NCHW 格式转换、Split 输出拷贝以及 Transpose/Shuffle 引起。
- **PointWise 算子**: 21 层占比 5.76%，主要为 SiLU 激活函数（Sigmoid + Mul）和 Add 操作，大部分已与 Conv 融合。

---

## 4. 瓶颈分类 (Roofline Model)

| 类别 | 层数 | 占比 |
|------|------|------|
| Compute-bound (计算受限) | 81 | 52.6% |
| Memory-bound (内存受限) | 73 | 47.4% |
| 其中 Pure Data Movement (AI=0) | 44 | — |

### 瓶颈分布分析

模型呈现 **计算和内存受限层基本均衡** 的特征：

- **Compute-bound 层** (81 层): 主要为 Conv 和 FusedParallel 类型，FP16 Tensor Core 已启用但仍受限于 GPU 计算吞吐。单层最大 AI 达 166.7 FLOP/byte。
- **Memory-bound 层** (73 层): 主要为 Reformat、PointWise、Softmax 等轻量算子，受限于 LPDDR5 204.8 GB/s 的内存带宽。
- **Pure Data Movement 层** (44 层): Reshape、Reformat 等不涉及计算的算子，AI=0，需要优化数据通路。

### Roofline 平台参数

| 参数 | 值 |
|------|-----|
| GPU | Ampere GA10B (2048 CUDA Cores) |
| GPU 时钟 | ~1.3 GHz |
| FP16 峰值算力 | 5.33 TFLOPS |
| 内存带宽 (LPDDR5) | 204.8 GB/s |
| Ridge Point | 26.0 FLOP/byte |

---

## 5. 系统级分析 (Nsight Systems)

| 指标 | 值 |
|------|-----|
| GPU Kernel 总耗时 | 1.448 s (100 次迭代) |
| GPU Kernel 总调用次数 | 17162 |
| 不同 Kernel 类型数 | 51 |
| Permutation/Reformat 类 Kernel 耗时 | 208.17 ms (GPU 时间的 14.4%) |
| 迭代次数 | 100 |

### CPU vs GPU 时间分布

| 类别 | 总耗时 | 占比 |
|------|--------|------|
| GPU Kernel 执行 | 1.448 s | 5.5% |
| CUDA Runtime API (CPU) | 5.945 s | 22.6% |
| 内存传输 (H2D/D2H) | 155.36 ms | 0.6% |
| OS Runtime (CPU) | 18.795 s | 71.3% |
| **合计** | **26.342 s** | **100%** |

### 内存传输开销

| 类型 | 传输次数 | 总数据量 | 总耗时 |
|------|----------|----------|--------|
| H2D (Host to Device) | 111 | 565.1 MB | 76.63 ms |
| D2H (Device to Host) | 110 | 296.1 MB | 78.71 ms |

### 系统级发现

- **GPU 利用率仅 5.5%**: 绝大部分时间消耗在 OS Runtime (71.3%) 和 CUDA Runtime API (22.6%)，GPU 远未饱和。
- **Permutation Kernel 是最大单一 kernel 开销**: 3300 次调用，累计 163.37 ms，占 GPU 时间的 11.3%，由 Transpose/Shuffle 产生。
- **CPU-side 开销巨大**: CUDA API 调用 (launchKernel, memcpy 等) 合计 5.945 s，是 GPU 执行时间的 4 倍。

---

## 6. 优化优先级建议

| 优先级 | 优化项 | 预期收益 | 实施难度 | 依据 |
|--------|--------|----------|----------|------|
| **P0** | **启用 CUDA Graph** | 极高 | 低 | GPU 利用率仅 5.5%，CPU launch 开销占 22.6%，CUDA Graph 可大幅减少 kernel launch 开销 |
| **P0** | **减少 Reformat / Permutation 操作** | 高 | 中 | Reformat 占 11.23%，Permutation kernel 占 GPU 时间 14.4%，两者合计开销巨大 |
| **P1** | **INT8 量化** | 很高 | 中 | Conv 占 73.07% 计算量，INT8 可再提升 2 倍吞吐，需要校准数据集 |
| **P1** | **增大 Batch Size** | 高 | 低 | 当前 batch=1 利用率低，batch>=4 可显著提升 GPU 利用率 |
| **P2** | **PointWise 算子融合检查** | 中 | 低 | PointWise 占 5.76%，确认 TensorRT 是否已全部融合 |
| **P2** | **减少 H2D/D2H 传输次数** | 中 | 低 | 111 次 H2D + 110 次 D2H 传输，可合并或使用异步传输 |
| **P3** | **模型结构简化** | 中 | 高 | 减少 Reshape/Transpose 等纯数据搬运操作需要修改模型结构 |

### 首要优化行动

1. **CUDA Graph**: Jetson Orin 支持 CUDA Graph，可将整个推理图捕获并重放，消除 CPU-side launch 开销。预估可提升 20-30% 吞吐。
2. **Reformat 削减**: Reformat 占 11.23% 且 Permutation kernel 占 GPU 时间 14.4%，尝试在 builder 中设置 `set_nhwc_enabled()` 或在模型层面减少 Transpose/Shuffle。
3. **INT8 量化**: 由 FP16 转为 INT8 后可望获得 2 倍 Conv 加速，需准备校准集并验证 mAP 损失。

---

## 7. 详细报告链接

| 报告 | 路径 | 内容 |
|------|------|------|
| TRT Profiler 逐层分析 | `trt_profiler/analysis_report.md` | 191 层逐层性能、类型分解、Top-15 |
| TRT Profiler 原始数据 | `trt_profiler/layer_profile.json` | 逐层 timeMs/averageMs/medianMs |
| TRT Profiler 结构数据 | `trt_profiler/layer_info.json` | 逐层 I/O 张量维度、数据类型 |
| Roofline 瓶颈分析 | `roofline/analysis_report.md` | AI 分布、compute/memory-bound 分类 |
| Nsight Systems 时间线 | `nsight_systems/analysis_report.md` | GPU timeline, Kernel 排名, CPU/GPU 分布 |
| Nsight Systems 图形界面 | `nsight_systems/report.nsys-rep` | Nsight Systems GUI 打开 |
| Nsight Compute 报告 | `nsight_compute/analysis_report.md` | 无法在 Jetson Orin 运行的限制说明 |

---

*本报告由 `trt_profiler/run.py`, `roofline/run.py`, `nsight_systems/run.py`, `nsight_compute/run.py` 的实测数据自动汇总生成。*
