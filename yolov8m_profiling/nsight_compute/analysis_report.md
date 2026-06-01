# YOLOv8m Nsight Compute 性能分析报告

**生成时间**: 2026-06-01 11:05:21
**工具版本**: Version 2022.2.1.0 (build 32234930) (public-release)
**引擎**: `weights/engines/yolov8m_fp16.engine`
**输入**: images (1, 3, 640, 640) float32, dummy data
**输出**: output0 (1, 84, 8400) float32
**平台**: Jetson Orin 集成 GPU (DRAM 统一内存)
**预热/测量**: 5 / 10 iterations

## 关键发现

- **总 kernel 调用次数**: 2342
- **唯一 kernel 类型数**: 45
- **最高调用频率**: `CUTENSOR_NAMESPACE::permutationKernelPLC3<CUTENSOR_NAMESPACE` (495 次, 占 21.1%)

## Kernel 调用次数排名

| # | Kernel | 调用次数 | 占比 | Warp Occupancy | Registers | Block Size | Shared Mem |
|---|--------|---------|------|----------------|-----------|------------|------------|
| 1 | `CUTENSOR_NAMESPACE::permutationKernelPLC3<CUTENSOR...` | 495 | 21.1% | 100% | 33 | 128 | 5.1 KB |
| 2 | `generatedNativePointwise` | 330 | 14.1% | 67% | 63 | 128 | 0.0 KB |
| 3 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nh...` | 165 | 7.0% | 17% | 216 | 128 | 65.5 KB |
| 4 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nh...` | 120 | 5.1% | 17% | 216 | 128 | 65.5 KB |
| 5 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nh...` | 120 | 5.1% | 17% | 216 | 256 | 73.7 KB |
| 6 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nh...` | 90 | 3.8% | 17% | 216 | 256 | 73.7 KB |
| 7 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nh...` | 75 | 3.2% | 17% | 166 | 128 | 65.5 KB |
| 8 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nh...` | 75 | 3.2% | 17% | 152 | 128 | 61.4 KB |
| 9 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nh...` | 60 | 2.6% | 17% | 144 | 128 | 73.7 KB |
| 10 | `cuEltwise::eltwise<cuEltwise::SimpleAlgo<float, fl...` | 60 | 2.6% | 100% | 35 | 128 | 0 |
| 11 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nh...` | 45 | 1.9% | 17% | 216 | 128 | 61.4 KB |
| 12 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nh...` | 45 | 1.9% | 17% | 216 | 128 | 61.4 KB |
| 13 | `sm50_xmma_pooling_coalescedC_NHWC_kMAX_3_False_exe...` | 45 | 1.9% | 100% | 39 | 128 | 0 |
| 14 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nh...` | 30 | 1.3% | 33% | 104 | 128 | 41.0 KB |
| 15 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nh...` | 30 | 1.3% | 17% | 158 | 128 | 73.7 KB |
| 16 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nh...` | 30 | 1.3% | 17% | 186 | 128 | 61.4 KB |
| 17 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nh...` | 30 | 1.3% | 17% | 216 | 128 | 65.5 KB |
| 18 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nh...` | 30 | 1.3% | 17% | 186 | 128 | 61.4 KB |
| 19 | `cuResizeLayer::ResizeKernelVectorizedH2x4<cuResize...` | 30 | 1.3% | 100% | 16 | 128 | 0 |
| 20 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nh...` | 30 | 1.3% | 17% | 216 | 128 | 65.5 KB |

## SM 利用率与 Occupancy 分析

| 分类 | Kernel 类型数 | 说明 |
|------|-------------|------|
| Warp Occupancy >= 50% | 14 | 计算/延迟隐藏良好 |
| Warp Occupancy 20-50% | 4 | 中等利用率 |
| Warp Occupancy < 20% | 25 | 低利用率，被寄存器或共享内存限制 |
| 无数据 | 2 | 指标不可用 |

## 内存与寄存器使用

| Kernel | 调用数 | Registers | Block Sz | Threads | Static SM | Dynamic SM | Warps |
|--------|--------|-----------|----------|---------|-----------|------------|-------|
| `CUTENSOR_NAMESPACE::permutationKernelPLC3<CUT...` | 495 | 33 | 128 | 204800 | 5.1 KB | 0.0 KB | 48 |
| `generatedNativePointwise` | 330 | 63 | 128 | 76800 | 0.0 KB | 0.0 KB | 32 |
| `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f...` | 165 | 216 | 128 | 25600 | 0.0 KB | 65.5 KB | 8 |
| `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f...` | 120 | 216 | 128 | 6400 | 0.0 KB | 65.5 KB | 8 |
| `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f...` | 120 | 216 | 256 | 3328 | 0.0 KB | 73.7 KB | 8 |
| `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f...` | 90 | 216 | 256 | 3328 | 0.0 KB | 73.7 KB | 8 |
| `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f...` | 75 | 166 | 128 | 6400 | 0.0 KB | 65.5 KB | 8 |
| `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f...` | 75 | 152 | 128 | 4480 | 0.0 KB | 61.4 KB | 8 |
| `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f...` | 60 | 144 | 128 | 4992 | 0.0 KB | 73.7 KB | 8 |
| `cuEltwise::eltwise<cuEltwise::SimpleAlgo<floa...` | 60 | 35 | 128 | 16896 | 0.0 KB | 0.0 KB | 48 |
| `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f...` | 45 | 216 | 128 | 12800 | 0.0 KB | 61.4 KB | 8 |
| `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f...` | 45 | 216 | 128 | 9600 | 0.0 KB | 61.4 KB | 8 |
| `sm50_xmma_pooling_coalescedC_NHWC_kMAX_3_Fals...` | 45 | 39 | 128 | 28800 | 0.0 KB | 0.0 KB | 48 |
| `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f...` | 30 | 104 | 128 | 76800 | 0.0 KB | 41.0 KB | 16 |
| `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f...` | 30 | 158 | 128 | 25600 | 0.0 KB | 73.7 KB | 8 |

## 关键分析

### 按功能分类

| 类别 | Kernel 数 | 调用次数 | 占比 |
|------|----------|---------|------|
| GEMM (Tensor Core Conv) | 29 | 1215 | 51.9% |
| Permutation (Transpose) | 1 | 495 | 21.1% |
| Pointwise (激活函数) | 2 | 390 | 16.7% |
| Pooling | 1 | 45 | 1.9% |
| Other | - | 197 | 8.4% |

### GPU 属性 (Orin)

- **架构**: sm80 (Ampere), 2048 CUDA Cores
- **L2 Cache**: 4 MB
- **Max Blocks per SM**: 16
- **Max Threads per SM**: 1536
- **Max Warps per SM**: 48
- **Max Registers per SM**: 65536

### 优化建议

- **495 次 Permutation kernel (Transpose/Reformat)** 开销较大，建议通过算子融合或布局优化减少数据重排
- GEMM kernel `sm80_xmma_fprop_implicit_gemm_f16f16_f16...` 的 Warp Occupancy 仅 17%，被寄存器 (216/thread) 限制。考虑调整 tile size 或使用更小的 block 配置
- Pointwise kernel 平均 67% occupancy

## 报告文件

- `report.ncu-rep` — Nsight Compute GUI (ncu-ui) 可打开的完整 profiling 文件
- `report.csv` — CSV 格式导出的指标数据
- `analysis_report.md` — 本报告
