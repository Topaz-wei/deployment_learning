# YOLOv8m Nsight Systems 性能分析报告

**生成时间**: 2026-05-31 19:08:13
**工具版本**: Nsight Systems 2023.2.4
**引擎**: `weights/engines/yolov8m_fp16.engine`
**输入**: images (1, 3, 640, 640) float32, dummy data
**输出**: output0 (1, 84, 8400) float32
**Kernel 执行模式**: DRAM
**预热/测量**: 10 / 100 iterations

## 关键发现

- 数据库中找到 72 个表
- GPU kernel 总调用次数: 17162
- GPU kernel 总耗时: 1.448 s
- 共 20 种不同 kernel 类型

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

## 报告文件

- `report.nsys-rep` — Nsight Systems GUI 可打开的时间线文件
- `report.sqlite` — 可编程查询的 SQLite 数据库
