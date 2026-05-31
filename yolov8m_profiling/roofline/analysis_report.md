# YOLOv8m Roofline Model 分析报告

**生成时间**: 2026-05-31 19:38:00
**数据来源**: `trt_profiler/layer_profile.json`
**引擎**: yolov8m_fp16.engine
**输入**: (1, 3, 640, 640) FP16

## 平台参数 (Jetson Orin)

| 参数 | 值 |
|------|-----|
| GPU | Ampere GA10B (2048 CUDA Cores) |
| GPU 时钟 | ~1.3 GHz |
| FP16 峰值算力 | 5.33 TFLOPS |
| 内存带宽 (LPDDR5) | 204.8 GB/s |
| Ridge Point (AI 阈值) | 26.0 FLOP/byte |

> **Ridge Point** = FP16 Peak / Memory BW = 5.33 TFLOPS / 204.8 GB/s = 26.0 FLOP/byte

> 当 Arithmetic Intensity (AI) < Ridge Point 时，层性能受内存带宽限制（memory-bound）；
> 当 AI > Ridge Point 时，层性能受计算能力限制（compute-bound）。

## Roofline 概览

- **总层数 (活跃)**: 154
- **总耗时**: 9606.5502 ms
- **总估算 FLOPs**: 204.25 GFLOPs
- **总估算 Memory Transfer**: 1.52 GB
- **总体 Arithmetic Intensity**: 134.0 FLOP/byte
- **Compute-bound 层**: 81 层
- **Memory-bound 层**: 73 层
- **Pure Data Movement 层 (AI=0)**: 44 层

### 瓶颈分布结论

模型计算和内存受限层较为均衡 (compute-bound 52.6%, memory-bound 47.4%)，
需要同时从计算和内存两方面进行优化。

## ASCII Roofline 图

```
                  YOLOv8m Roofline Chart (ASCII)

   T
   F 10^2 +---------------------------------------------------+
   L      |  Ridge Point (26.0 FLOP/byte)         Compute     |
   O  10^1+- - - - - - - - - - - - - - - - - - - Bound - - - -+
   P      |                                                  |
   S  10^0+                                                  |
         |            * (Conv ~166.7 TFLOPS peak)            |
       10^{-1}+                                     *        |
         |                                    *              |
       10^{-2}+           **********                         |
         |          ***    Memory-Bound  Region              |
       10^{-3}+   ****                                       |
         | ****                                              |
       10^{-4}+*                                             |
  TFLOPS |                                                  |
       10^{-5}+----------------------------------------------+
         10^{-1}  10^0   10^1  10^2   10^3  10^4   10^5  10^6
            Arithmetic Intensity (FLOP/byte)


   Peak FP16: 5.33 TFLOPS     Memory BW: 204.8 GB/s
   Ridge Point: 26.0 FLOP/byte


   Distribution:
     * Compute-bound layers: 81
     * Memory-bound layers:  73
     * Pure data movement:   44

   Top-5 layers on roofline:
     /model.22/dfl/conv/Conv                             AI=   166.7  TFLOPS=0.0021  [compute-bound]
     /model.22/cv3.0/cv3.0.1/conv/Conv + PWN(PWN(/mo...  AI=   166.7  TFLOPS=0.0053  [compute-bound]
     /model.2/cv2/conv/Conv + PWN(PWN(/model.2/cv2/a...  AI=   166.7  TFLOPS=0.0070  [compute-bound]
     /model.6/m.0/cv2/conv/Conv                          AI=   166.7  TFLOPS=0.0097  [compute-bound]
     /model.12/m.0/cv2/conv/Conv + PWN(PWN(/model.12...  AI=   166.7  TFLOPS=0.0115  [compute-bound]
```

## 逐层 Roofline 分析表 (Top-30 耗时)

| # | Layer Type | Time (ms) | FLOPs | Bytes | AI (FLOP/byte) | Achieved TFLOPS | BW (GB/s) | Bottleneck |
|---|------------|-----------|-------|-------|----------------|-----------------|-----------|------------|
| 1 | Conv | 1183.1400 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0021 | 0.0 | compute-bound |
| 2 | Conv | 475.7830 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0053 | 0.0 | compute-bound |
| 3 | Conv | 355.4890 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0070 | 0.0 | compute-bound |
| 4 | Conv | 257.3940 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0097 | 0.1 | compute-bound |
| 5 | Conv | 216.4980 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0115 | 0.1 | compute-bound |
| 6 | Softmax | 213.9720 | 5.00 MFLOPs | 2.00 MB | 2.5 | 0.0000 | 0.0 | memory-bound |
| 7 | FusedParallel | 205.2490 | 3.00 GFLOPs | 20.00 MB | 150.0 | 0.0146 | 0.1 | compute-bound |
| 8 | Conv | 177.7450 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0141 | 0.1 | compute-bound |
| 9 | Conv | 172.1860 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0145 | 0.1 | compute-bound |
| 10 | FusedParallel | 170.5840 | 3.00 GFLOPs | 20.00 MB | 150.0 | 0.0176 | 0.1 | compute-bound |
| 11 | PointWise | 144.9230 | 10.00 MFLOPs | 5.00 MB | 2.0 | 0.0001 | 0.0 | memory-bound |
| 12 | Reformat | 132.3530 | 0.00e+00 FLOPs | 4.00 MB | 0.0 | 0.0000 | 0.0 | memory-bound (pure data movement) |
| 13 | Conv | 120.8210 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0207 | 0.1 | compute-bound |
| 14 | Conv | 107.1020 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0233 | 0.1 | compute-bound |
| 15 | Conv | 106.5220 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0235 | 0.1 | compute-bound |
| 16 | Conv | 101.0180 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0247 | 0.1 | compute-bound |
| 17 | FusedParallel | 99.8906 | 3.00 GFLOPs | 20.00 MB | 150.0 | 0.0300 | 0.2 | compute-bound |
| 18 | Conv | 99.4785 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0251 | 0.2 | compute-bound |
| 19 | Reshape | 98.4398 | 0.00e+00 FLOPs | 1.00 MB | 0.0 | 0.0000 | 0.0 | memory-bound (pure data movement) |
| 20 | Conv | 94.1234 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0266 | 0.2 | compute-bound |
| 21 | Conv | 91.7105 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0273 | 0.2 | compute-bound |
| 22 | Conv | 89.4279 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0280 | 0.2 | compute-bound |
| 23 | Conv | 88.8913 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0281 | 0.2 | compute-bound |
| 24 | Conv | 88.5647 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0282 | 0.2 | compute-bound |
| 25 | Conv | 88.0965 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0284 | 0.2 | compute-bound |
| 26 | Conv | 86.7425 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0288 | 0.2 | compute-bound |
| 27 | Conv | 85.1910 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0293 | 0.2 | compute-bound |
| 28 | Reformat | 85.0811 | 0.00e+00 FLOPs | 4.00 MB | 0.0 | 0.0000 | 0.0 | memory-bound (pure data movement) |
| 29 | Conv | 84.2473 | 2.50 GFLOPs | 15.00 MB | 166.7 | 0.0297 | 0.2 | compute-bound |
| 30 | Reformat | 76.6654 | 0.00e+00 FLOPs | 4.00 MB | 0.0 | 0.0000 | 0.1 | memory-bound (pure data movement) |

> 总计 154 层活跃层，此处仅列出 Top-30。

## 按层类型汇总的 Roofline 指标

| Layer Type | Count | Time (ms) | Time % | Total FLOPs | Total Bytes | AI (FLOP/byte) |
|------------|-------|-----------|--------|-------------|-------------|----------------|
| Conv | 78 | 7019.9097 | 73.07% | 195.00 GFLOPs | 1.17 GB | 166.7 |
| Reformat | 40 | 1078.9950 | 11.23% | 0.00e+00 FLOPs | 160.00 MB | 0.0 |
| PointWise | 21 | 553.0082 | 5.76% | 210.00 MFLOPs | 105.00 MB | 2.0 |
| FusedParallel | 3 | 475.7236 | 4.95% | 9.00 GFLOPs | 60.00 MB | 150.0 |
| Softmax | 1 | 213.9720 | 2.23% | 5.00 MFLOPs | 2.00 MB | 2.5 |
| Reshape | 1 | 98.4398 | 1.02% | 0.00e+00 FLOPs | 1.00 MB | 0.0 |
| Shuffle | 1 | 56.8615 | 0.59% | 0.00e+00 FLOPs | 4.00 MB | 0.0 |
| Resize | 2 | 44.1619 | 0.46% | 0.00e+00 FLOPs | 8.00 MB | 0.0 |
| MaxPool | 3 | 41.5384 | 0.43% | 15.00 MFLOPs | 6.00 MB | 2.5 |
| Other | 4 | 23.9401 | 0.25% | 20.00 MFLOPs | 8.00 MB | 2.5 |

## 优化建议

### Compute-bound 层优化

Compute-bound 层主要类型: Conv, FusedParallel

Compute-bound 层的性能受 GPU 计算单元限制。优化方向:

1. **使用 Tensor Cores**: Jetson Orin 的 Ampere 架构支持 Tensor Cores，
   可大幅提升矩阵运算吞吐。确保 TensorRT 已启用 FP16 Tensor Core (默认启用)。
2. **INT8 量化**: 如果精度允许，INT8 量化可将计算吞吐再提升 2 倍。
3. **Winograd 卷积**: 对小卷积核 (3x3) 使用 Winograd 算法减少乘法次数。
4. **增大 Batch Size**: batch size > 1 可以提升 GPU 利用率，摊薄 kernel launch 开销。
5. **CUDA Graph**: 捕获并重放推理图，减少 kernel launch 开销。

### Memory-bound 层优化

Memory-bound 层主要类型: PointWise, Softmax, MaxPool, Other

Memory-bound 层的性能受数据搬运速度限制。优化方向:

1. **算子融合 (Operator Fusion)**: 将多个 memory-bound 的小算子融合成一个 kernel，
   减少中间结果的读写。TensorRT 已自动执行许多融合，但可检查是否有遗漏。
2. **减少 Reformat 操作**: Reformat 层 (如 CopyNode、Split 输出拷贝) 占总耗时 11.2%。
   尝试在构建 engine 时设置 `set_nhwc_enabled()` 以减少 NHWC/NCHW 转换。
3. **减少 Reshape/Shuffle/Transpose**: 这些操作通常不涉及计算，但需要大量数据搬运。
   如果可能，在模型设计阶段减少这些操作的频率。
4. **内存池与缓存优化**: 确保 TensorRT 使用了适当的内存池配置。
5. **增大 Batch Size**: batch size 增大可提高计算密度，改善 AI 值。

### Pure Data Movement 层 (AI = 0)

纯数据搬运层类型: Reshape, Reformat, Shuffle, Resize

AI=0 的层不涉及计算，只有数据读写。虽然单个此类层的耗时通常很小，但
大量累积可能导致可观的 overhead。建议:

1. 通过算子融合消除不必要的中间数据拷贝。
2. 检查是否有重复或冗余的 Reformat 操作。
3. 使用 `--profilingVerbosity=detailed` 查看完整的层图以识别冗余操作。

### 整体优化策略

| 优先级 | 优化项 | 预期收益 | 实施难度 |
|--------|--------|----------|----------|
| P0 | 确认 FP16 Tensor Core 已启用 | 高 | 低 (默认启用) |
| P0 | 启用算子融合优化级别 | 高 | 低 (构建参数) |
| P1 | INT8 量化 | 很高 | 中 (需要校准数据集) |
| P1 | 减少 Reformat/Reshape 操作 | 中 | 中 (修改模型) |
| P2 | 增大 Batch Size (>=4) | 高 | 低 (调用参数) |
| P2 | CUDA Graph 捕获 | 中 | 低 |
| P3 | 模型结构改进 (减少 memory-bound 层) | 中 | 高 (重新训练) |

## 附录

### 估算方法说明

由于 TensorRT engine 不暴露权重和中间张量的形状信息，本报告中的 FLOPs 和
memory bytes 均为基于层类型的**架构估算值**，而非精确测量。估算来源:

- Conv 层估算基于 YOLOv8m 典型 3x3 卷积在 80x80 分辨率下的计算量 (~2.5 GFLOPs)
- PointWise 层估算基于逐元素操作 (`1e7` FLOPs ~ 激活函数 + 乘法)
- Reformat/Reshape/Transpose 等数据搬运层设为 FLOPs=0（仅搬运）
- FusedParallel 估算基于检测头中并行卷积的综合计算量 (~3 GFLOPs)

精确的 Roofline 分析需要借助 Nsight Compute 等工具获得硬件计数器的精确数据。

### 报告文件

- `layer_profile.json` — trt_profiler 导出的层性能数据
- `run.py` — 本 Roofline 分析脚本
- `analysis_report.md` — 本报告
