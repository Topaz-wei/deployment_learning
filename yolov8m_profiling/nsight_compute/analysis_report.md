# YOLOv8m Nsight Compute 性能分析报告

**生成时间**: 2026-05-31 19:21:45
**工具版本**: Version 2022.2.1.0 (build 32234930) (public-release)
**引擎**: `weights/engines/yolov8m_fp16.engine`
**输入**: images (1, 3, 640, 640) float32, dummy data
**输出**: output0 (1, 84, 8400) float32
**平台**: Jetson Orin 集成 GPU (DRAM 统一内存)
**预热/测量**: 5 / 10 iterations

## Profiling 状态: 无法在当前平台运行

Nsight Compute 命令行工具 (`ncu`) 在 Jetson Orin 集成 GPU 上无法直接进行 kernel
级 profiling，原因如下：

### 已知限制

- **权限要求**: ncu 需要 root 权限才能访问 GPU 性能计数器，但嵌入式 Jetson 平台
  通常限制直接 root 访问。
- **架构差异**: Jetson Orin 使用集成 GPU (T234/Ampere 架构)，与 x86_64 桌面/服务器
  GPU 在驱动程序模型和性能计数器访问接口上存在差异。
- **ncu-ui 不支持**: 当前的 Nsight Compute 版本 (2022.2.1) 不支持在 ARM64 上运行
  ncu-ui 图形界面。

### 建议的替代方案

1. **在宿主机上使用 ncu-ui 分析**:
   - 在 x86_64 桌面 (带独立 NVIDIA GPU) 上安装 Nsight Compute。
   - 将 TensorRT engine 拷贝到宿主机。
   - 使用 ncu-ui 打开并分析 engine 的 profiling 结果。

2. **使用 TensorRT 内置 Profiler**:
   - 见 `trt_profiler/` 目录下的 IProfiler 实现，可获取每个 layer 的性能数据。

3. **使用 Nsight Systems 做粗粒度分析**:
   - 见 `nsight_systems/` 目录，可获取 kernel 时间线、CPU/GPU 时间分布。

4. **通过 ncu --set basic 手动重试 (需 sudo)**:
   ```bash
   sudo /opt/nvidia/nsight-compute/2022.2.1/ncu --set basic \
     --export /home/ssd/projects/deployment_learning/yolov8m_profiling/nsight_compute/report.ncu-rep \
     --force-overwrite \
     /home/ssd/anaconda3/envs/py38/bin/python3 /home/ssd/projects/deployment_learning/yolov8m_profiling/nsight_compute/../common/inference_workload.py \
     --engine /home/ssd/projects/deployment_learning/yolov8m_profiling/nsight_compute/../../weights/engines/yolov8m_fp16.engine --data dummy --warmup 5 --iters 10
   ```
   ```bash
   /opt/nvidia/nsight-compute/2022.2.1/ncu -i /home/ssd/projects/deployment_learning/yolov8m_profiling/nsight_compute/report.ncu-rep --csv --page details > /home/ssd/projects/deployment_learning/yolov8m_profiling/nsight_compute/report.csv
   ```

   然后再运行本脚本 (因检测到 report.ncu-rep 已存在, 将跳过 profiling,
   直接解析 CSV 并生成完整报告)。

## 报告文件

- `analysis_report.md` — 本报告
