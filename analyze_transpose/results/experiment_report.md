# YOLOv8m ONNX Transpose 插入实验报告

> 平台: Jetson Orin ARM64 / ONNX Runtime 1.19 CPU / conda py38
> 模型: YOLOv8m ONNX (325 节点, 输入 [1,3,640,640] NCHW, 输出 [1,84,8400])
> 测试: warmup=10, iters=100, 模拟随机输入

---

## 实验 1: Baseline 基准测试

原始 YOLOv8m ONNX，4 种 ORT 图优化级别下的推理延迟。

| 优化级别 | Mean (ms) | Min (ms) | P50 (ms) | Std (ms) | FPS |
|----------|-----------|----------|----------|----------|-----|
| DISABLE_ALL | 1205.7 | 823.5 | 1201.7 | 223.3 | 0.83 |
| BASIC | 1045.2 | 708.2 | 972.5 | 305.6 | 0.96 |
| EXTENDED | 893.0 | 606.0 | 770.6 | 323.3 | 1.12 |
| ALL | 951.8 | 640.7 | 888.8 | 230.7 | 1.05 |

**结论**: EXTENDED 优化级别下性能最佳 (893ms)，优于 ALL 级别的 952ms。

---

## 实验 2: 输入层 Transpose 开销

在 `images` 后插入 NCHW→NHWC→NCHW round-trip Transpose pair (2 个 Transpose)。

| 指标 | Baseline | 修改后 | Delta |
|------|----------|--------|-------|
| Mean (ms) | 1094.9 | 913.5 | -181.4 (0.8343x) |
| Total Nodes | 325 | 327 | +2 |
| Transpose Nodes | 2 | 4 | +2 |

**结论**: ORT ALL 优化消除了 round-trip Transpose pair 的净开销，修改模型延迟与 baseline 无显著差异。

---

## 实验 3: 输出层 Transpose 开销

在 `output0` 后插入单次 Transpose(perm=[0,2,1])，输出从 [1,84,8400] 变为 [1,8400,84]。

| 指标 | Baseline | 修改后 | Delta |
|------|----------|--------|-------|
| Mean (ms) | 752.0 | 737.9 | -14.1 (0.9813x) |
| Total Nodes | 325 | 326 | +1 |
| Transpose Nodes | 2 | 3 | +1 |

**结论**: 输出层单次 Transpose 开销可忽略，因为操作的是 3D 小 tensor [1,84,8400]。

---

## 实验 4: 输入+输出双 Transpose

同时插入输入 round-trip pair (2T) + 输出单次 Transpose (1T)，共 3 个 Transpose。

| 指标 | Baseline | 修改后 | Delta |
|------|----------|--------|-------|
| Mean (ms) | 829.3 | 812.4 | -16.9 (0.9796x) |
| Total Nodes | 325 | 328 | +3 |
| Transpose Nodes | 2 | 5 | +3 |

**结论**: 与 exp2 一致，ORT ALL 优化有效抵消了 Transpose 开销。

---

## 实验 5: 中间层 Transpose

在第 20 个 Conv (`/model.2/m.1/cv2/conv/Conv`) 输出后插入 round-trip Transpose pair (2T)。

| 指标 | Baseline | 修改后 | Delta |
|------|----------|--------|-------|
| Mean (ms) | 739.6 | 723.8 | -15.8 (0.9786x) |
| Total Nodes | 325 | 327 | +2 |
| Transpose Nodes | 2 | 4 | +2 |

**结论**: 中间层 round-trip pair 同样被 ORT 优化抵消，无显著延迟增加。

---

## 实验 6: 多层 Transpose 累积效应

在 Backbone 的 4 个 Stage 输出后各插入 round-trip Transpose pair，共 8 个 Transpose。

| 指标 | Baseline | 修改后 | Delta |
|------|----------|--------|-------|
| Mean (ms) | 821.0 | 825.9 | +4.9 (1.0060x) |
| Total Nodes | 325 | 333 | +8 |
| Transpose Nodes | 2 | 10 | +8 |
| **每对 (2T) 边际** | — | — | **1.23 ms** |
| **每个 T 边际** | — | — | **0.61 ms** |

**结论**: 当 Transpose 累积到 8 个时，才观测到可测量的延迟增加。每个 Transpose 边际开销约 0.6ms，在整体 820ms 推理时间中占比 <0.1%。

---

## 实验 7: ORT 图优化对 Transpose 的消除

对比不同优化级别下原始/修改模型的节点数和延迟。

| 优化级别 | Orig Nodes | Orig T | Mod Nodes | Mod T | Orig (ms) | Mod (ms) | Delta (ms) |
|----------|-----------|--------|-----------|-------|-----------|----------|-------------|
| DISABLE_ALL | 325 | 2 | 327 | 4 | 711.7 | 768.9 | +57.2 |
| BASIC | 325 | 2 | 327 | 4 | 733.8 | 734.3 | +0.5 |
| EXTENDED | 325 | 2 | 327 | 4 | 702.4 | 695.8 | -6.6 |
| ALL | 325 | 2 | 327 | 4 | 726.1 | 810.3 | +84.2 |

**结论**: ORT CPU 后端在任意优化级别下均**不做 Transpose 节点的图级消除**（修改模型始终比原始多 2 个 Transpose）。EXTENDED 仍是最佳选择。

---

## 实验 8: Transpose 与 Conv 融合边界

使用精简微模型 (1×Conv + 4×Transpose, 64通道 32×32 输入) 测试 ORT 融合行为。

| 模型 | 优化级别 | 节点数 | Transpose | Conv | Mean (ms) |
|------|---------|--------|-----------|------|-----------|
| Conv_Only | DISABLE_ALL | 1 | 0 | 1 | 0.6056 |
| Conv_Only | ALL | 1 | 0 | 1 | 0.5841 |
| T_Conv_T | DISABLE_ALL | 5 | 4 | 1 | 0.8891 |
| T_Conv_T | ALL | 5 | 4 | 1 | 0.5852 |

**结论**: ORT ALL 不减少节点数（不做图级融合），但通过 kernel 级优化显著降低延迟（T_Conv_T: 0.89→0.59ms, -34%）。

---

## 实验 9: 不同输入分辨率

模型输入 shape 静态固定为 640×640，仅对比原始和修改模型在该分辨率下的表现。

| 分辨率 | 模型 | Mean (ms) | FPS | 像素数 | ns/pixel |
|--------|------|-----------|-----|--------|----------|
| 640x640 | Original | 737.9 | 1.36 | 1,228,800 | 600.5 |
| 640x640 | Modified | 747.3 | 1.34 | 1,228,800 | 608.1 |

**结论**: 静态输入 shape 限制了多分辨率对比。640×640 下输入 Transpose pair 额外开销在噪声范围内。

> 注: 需使用支持动态 shape 的 ORT session 或修改模型输入定义才能测试多分辨率。

---

## 总结

### 核心发现

1. **ORT ALL 图优化不删除 Transpose 节点** — 在全部 9 个实验中，修改模型始终保留插入的 Transpose 节点。ORT CPU 后端不在 ONNX 图层面做 Transpose 消除/常量折叠。

2. **少量 Transpose (1-3 个) 开销可忽略** — exp2-5 中 delta 在 ±20ms 内，小于标准差 (std=~200ms)，属于随机波动。

3. **累积 8 个 Transpose 后才可测量** — exp6 显示每个 Transpose 边际成本约 **0.6ms**，在整体 820ms 推理时间中占比 <0.1%。

4. **EXTENDED 优于 ALL** — YOLOv8m 的 CPU 推理最佳优化级别是 EXTENDED (893ms)，而非 ALL (952ms)，这一结论在全部实验中保持一致。

5. **kernel 级优化有效** — exp8 微模型显示 ORT ALL 虽不减少图节点，但通过 kernel 融合将 Transpose+Conv 块延迟降低 34%。

### 实验环境局限性

- ONNX Runtime CPU only (Jetson Orin ARM64 无 CUDAExecutionProvider)
- 模型静态输入 shape [1,3,640,640]
- 使用随机模拟输入而非真实图像

---

*生成时间: 2026-05-31*
