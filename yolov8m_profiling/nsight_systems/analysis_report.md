# YOLOv8m Nsight Systems 性能分析报告

**生成时间**: 2026-05-31 19:15:20
**工具版本**: NVIDIA Nsight Systems version 2023.2.4.44-33011852v0
**引擎**: `weights/engines/yolov8m_fp16.engine`
**输入**: images (1, 3, 640, 640) float32, dummy data
**输出**: output0 (1, 84, 8400) float32
**Kernel 执行模式**: DRAM
**预热/测量**: 10 / 100 iterations

## 关键发现

- **Top-3 kernel 类型** 占总 GPU 时间的 **30.0%** GPU 时间 = 434.07 ms，其余 48 种 kernel 占 70.0%
- **Permutation/Reformat 类 kernel** 耗时 208.17 ms（GPU 时间的 14.4%）
  - 此类 kernel 由 TensorRT 的 Transpose / Shuffle / Reshape 算子产生，是潜在的优化方向（可尝试 Transpose 融合或使用 TensorRT 的优化策略）
- **内存传输** 增加 155.36 ms 额外开销
  - H2D (HTOD): 111 次传输, 565.1 MB 数据, 76.63 ms
  - D2H (DTOH): 110 次传输, 296.1 MB 数据, 78.71 ms
- **GPU Kernel 总调用次数**: 17162
- **GPU Kernel 总耗时**: 1.448 s
- **不同 Kernel 类型数**: 51
- **关键优化方向**:
  1. Permutation/Transpose 算子融合或采用更高效的实现
  1. 减少 H2D/D2H 传输次数, 使用异步传输或 CUDA Graph 进行传输合并
  1. 检查卷积算子与 Elementwise 算子的融合情况

## Top-20 CUDA Kernel 单次调用耗时排名

| # | Kernel Name | Duration |
|---|-------------|----------|
| 1 | `void CUTENSOR_NAMESPACE::permutationKernelPLC3<CUTENSOR_NAMESPACE::VectorWrite2D` | 5.19 ms |
| 2 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x256x32` | 5.01 ms |
| 3 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x32x32_` | 4.97 ms |
| 4 | `void CUTENSOR_NAMESPACE::permutationKernelPLC3<CUTENSOR_NAMESPACE::VectorWrite2D` | 4.96 ms |
| 5 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x128x32` | 4.30 ms |
| 6 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x64x32_` | 3.99 ms |
| 7 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x256x32` | 3.89 ms |
| 8 | `void CUTENSOR_NAMESPACE::permutationKernelPLC3<CUTENSOR_NAMESPACE::VectorWrite2D` | 3.87 ms |
| 9 | `trt_ampere_h16816cudnn_64x128_sliced1x2_ldg8_relu_exp_stages_64x4_medium_nhwc_tn` | 3.87 ms |
| 10 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x128x32` | 3.79 ms |
| 11 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x256x32` | 3.79 ms |
| 12 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x64x32_` | 3.79 ms |
| 13 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x256x32` | 3.63 ms |
| 14 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x256x32` | 3.54 ms |
| 15 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x32x32_` | 3.52 ms |
| 16 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x64x32_` | 3.51 ms |
| 17 | `void CUTENSOR_NAMESPACE::permutationKernelPLC3<CUTENSOR_NAMESPACE::VectorWrite2D` | 3.49 ms |
| 18 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x64x32_` | 3.47 ms |
| 19 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x256x32` | 3.43 ms |
| 20 | `trt_ampere_h16816cudnn_64x128_sliced1x2_ldg8_relu_exp_stages_64x4_medium_nhwc_tn` | 3.37 ms |

## Kernel 耗时汇总 (按总耗时排序)

| # | Kernel Name | 调用次数 | 总耗时 | 平均耗时 |
|---|-------------|----------|--------|----------|
| 1 | `void CUTENSOR_NAMESPACE::permutationKernelPLC3<CUTENSOR_NAMESPACE::VectorWr` | 3300 | 163.37 ms | 49.5 us |
| 2 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x1` | 1210 | 138.85 ms | 114.8 us |
| 3 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x2` | 880 | 131.85 ms | 149.8 us |
| 4 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x1` | 880 | 99.06 ms | 112.6 us |
| 5 | `generatedNativePointwise` | 2420 | 91.99 ms | 38.0 us |
| 6 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x1` | 550 | 82.49 ms | 150.0 us |
| 7 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x2` | 660 | 76.05 ms | 115.2 us |
| 8 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x6` | 110 | 71.19 ms | 647.2 us |
| 9 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x6` | 440 | 53.86 ms | 122.4 us |
| 10 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x3` | 220 | 53.55 ms | 243.4 us |
| 11 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x6` | 330 | 48.97 ms | 148.4 us |
| 12 | `void cuSoftMaxLayer::nchwSoftmaxAxisWSmall<float, (unsigned int)32>(int, co` | 110 | 43.88 ms | 398.9 us |
| 13 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize64x12` | 550 | 37.91 ms | 68.9 us |
| 14 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x6` | 330 | 37.32 ms | 113.1 us |
| 15 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x6` | 220 | 32.73 ms | 148.8 us |
| 16 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x6` | 220 | 31.69 ms | 144.0 us |
| 17 | `trt_ampere_h16816cudnn_64x128_sliced1x2_ldg8_relu_exp_stages_64x4_medium_nh` | 110 | 28.08 ms | 255.3 us |
| 18 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x1` | 220 | 21.11 ms | 96.0 us |
| 19 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x2` | 110 | 18.37 ms | 167.0 us |
| 20 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x6` | 220 | 18.23 ms | 82.9 us |
| 21 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x1` | 220 | 16.84 ms | 76.6 us |
| 22 | `sm80_xmma_fprop_image_first_layer_f16f16_f32_f16_nhwckrsc_nhwc_hmma_k48c4r3` | 110 | 16.81 ms | 152.8 us |
| 23 | `trt_ampere_h16816cudnn_128x64_sliced1x2_ldg8_relu_exp_stages_64x4_small_nhw` | 110 | 16.67 ms | 151.6 us |
| 24 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x6` | 110 | 15.56 ms | 141.4 us |
| 25 | `void CUTENSOR_NAMESPACE::permutationKernelPLC3<CUTENSOR_NAMESPACE::VectorWr` | 110 | 11.67 ms | 106.1 us |
| 26 | `void genericReformat::copyPackedKernel<float, float, (bool)1, (bool)1, gene` | 110 | 10.14 ms | 92.2 us |
| 27 | `void cuInt8::nhwcTonchw<__half, (int)32, (int)32, (int)2>(const __half *, T` | 220 | 8.98 ms | 40.8 us |
| 28 | `void cuResizeLayer::ResizeKernelVectorizedH2x4<cuResizeLayer::NearestNeighb` | 220 | 8.67 ms | 39.4 us |
| 29 | `void CUTENSOR_NAMESPACE::permutationKernelPLC3<CUTENSOR_NAMESPACE::VectorWr` | 110 | 7.43 ms | 67.5 us |
| 30 | `void cudnn::cnn::conv2d_grouped_direct_kernel<(bool)0, (bool)1, (bool)0, (b` | 110 | 6.72 ms | 61.1 us |
| 31 | `sm50_xmma_pooling_coalescedC_NHWC_kMAX_3_False_execute_kernel_trt` | 330 | 6.60 ms | 20.0 us |
| 32 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x1` | 110 | 5.42 ms | 49.2 us |
| 33 | `void CUTENSOR_NAMESPACE::permutationKernelPLC3<CUTENSOR_NAMESPACE::VectorWr` | 110 | 4.02 ms | 36.5 us |
| 34 | `void genericReformat::copyVectorizedKernel<double, __half, __half, (bool)1,` | 220 | 3.38 ms | 15.3 us |
| 35 | `void genericReformat::copyVectorizedKernel<double, __half, float, (bool)1, ` | 110 | 3.33 ms | 30.3 us |
| 36 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x6` | 110 | 3.20 ms | 29.1 us |
| 37 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x6` | 110 | 2.65 ms | 24.1 us |
| 38 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize64x64` | 110 | 2.43 ms | 22.1 us |
| 39 | `void genericReformat::copyPackedKernel<__half, __half, (bool)0, (bool)1, ge` | 110 | 2.25 ms | 20.5 us |
| 40 | `sm80_xmma_fprop_implicit_gemm_indexed_f16f16_f16f16_f16_nhwckrsc_nhwc_tiles` | 110 | 2.09 ms | 19.0 us |
| 41 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize64x32` | 110 | 1.91 ms | 17.4 us |
| 42 | `void genericReformat::copyVectorizedKernel<double, __half, float, (bool)1, ` | 110 | 1.91 ms | 17.4 us |
| 43 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x6` | 110 | 1.39 ms | 12.6 us |
| 44 | `void cuEltwise::eltwise<cuEltwise::SimpleAlgo<float, float>, cuEltwise::Com` | 220 | 1.37 ms | 6.2 us |
| 45 | `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x3` | 110 | 1.37 ms | 12.4 us |
| 46 | `sm80_xmma_fprop_implicit_gemm_indexed_f16f16_f16f16_f16_nhwckrsc_nhwc_tiles` | 110 | 1.17 ms | 10.6 us |
| 47 | `void cuEltwise::eltwise<cuEltwise::SimpleAlgo<float, float>, cuEltwise::Com` | 110 | 1.01 ms | 9.2 us |
| 48 | `void cuInt8::nhwcTonchw<float, (int)32, (int)32, (int)2>(const __half *, T1` | 110 | 818.7 us | 7.4 us |
| 49 | `void genericReformat::copyVectorizedKernel<double, float, __half, (bool)1, ` | 110 | 670.1 us | 6.1 us |
| 50 | `void cuEltwise::eltwise<cuEltwise::SimpleAlgo<float, float>, cuEltwise::Com` | 110 | 577.0 us | 5.2 us |
| 51 | `void cask_trt::computeOffsetsKernel<(bool)0, (bool)0>(cask_trt::ComputeOffs` | 2 | 12.4 us | 6.2 us |

## H2D/D2H 内存传输分析

| copyKind | 含义 | 传输次数 | 总数据量 | 总耗时 |
|----------|------|----------|----------|--------|
| 1 | H2D (HTOD) | 111 | 565.1 MB | 76.63 ms |
| 2 | D2H (DTOH) | 110 | 296.1 MB | 78.71 ms |
| 8 | D2D (DTOD) | 2 | 5.0 KB | 15.9 us |

**说明**: copyKind 对应 CUDA MEMCPY 操作类型:
- `1` = `CUDA_MEMCPY_KIND_HTOD` (Host to Device)
- `2` = `CUDA_MEMCPY_KIND_DTOH` (Device to Host)
- `8` = `CUDA_MEMCPY_KIND_DTOD` (Device to Device)

## CPU vs GPU 时间分布

| 类别 | 调用次数 | 总耗时 | 占比 |
|------|----------|--------|------|
| GPU Kernel 执行 | 17162 | 1.448 s | 5.5% |
| CUDA Runtime API (CPU) | 21092 | 5.945 s | 22.6% |
| 内存传输 (H2D/D2H) | 223 | 155.36 ms | 0.6% |
| OS Runtime (CPU) | 19329 | 18.795 s | 71.3% |
| **合计** | — | **26.342 s** | **100%** |

**说明**:
- GPU Kernel 执行: GPU 上实际运行 kernel 的时间
- CUDA Runtime API (CPU): CPU 端调用 CUDA API（如 cudaMemcpyAsync, cudaLaunchKernel）的耗时
- 内存传输 (H2D/D2H): 数据在 Host 与 Device 间传输的耗时
- OS Runtime (CPU): 操作系统级运行时调用（如 mmap, fread, pthread 等）的耗时

## 报告文件

- `report.nsys-rep` — Nsight Systems GUI 可打开的时间线文件
- `report.sqlite` — 可编程查询的 SQLite 数据库
