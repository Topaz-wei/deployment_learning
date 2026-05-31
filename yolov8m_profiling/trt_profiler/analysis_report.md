# YOLOv8m TensorRT Profiler 逐层性能分析报告

**生成时间**: 2026-05-31 19:33:41
**工具版本**: TensorRT v8502
**引擎路径**: `/home/ssd/projects/deployment_learning/weights/engines/yolov8m_fp16.engine`
**输入**: images (1, 3, 640, 640) float32
**输出**: output0 (1, 84, 8400) float32
**预热/测量**: 10 / 10 seconds

## 关键发现

- **总层数**: 191
- **总耗时**: 9606.5502 ms
- **Top-3 最耗时层占比**: 20.97%
- **Top-5 最耗时层占比**: 25.90%
- **逐层数据类型数**: 10
- **Conv 层总耗时**: 7019.9097 ms (73.07%)
- **Reformat 层总耗时**: 1078.9950 ms (11.23%) **（重点关注）**
- **FusedParallel 层总耗时**: 475.7236 ms (4.95%)
  - Reformat 开销较大（超过 5%），可能是由 Transpose/Shuffle 等改变数据排布的操作引起。

## Top-15 最耗时层

| # | Layer Name | Type | Time (ms) | Time % |
|---|------------|------|-----------|--------|
| 1 | `/model.22/dfl/conv/Conv` | Conv | 1183.1400 | 12.32% |
| 2 | `/model.22/cv3.0/cv3.0.1/conv/Conv + PWN(PWN(/model.22/cv3.0/cv3.0.1/act/Sigmoid), /mode...` | Conv | 475.7830 | 4.95% |
| 3 | `/model.2/cv2/conv/Conv + PWN(PWN(/model.2/cv2/act/Sigmoid), /model.2/cv2/act/Mul)` | Conv | 355.4890 | 3.70% |
| 4 | `/model.6/m.0/cv2/conv/Conv` | Conv | 257.3940 | 2.68% |
| 5 | `/model.12/m.0/cv2/conv/Conv + PWN(PWN(/model.12/m.0/cv2/act/Sigmoid), /model.12/m.0/cv2...` | Conv | 216.4980 | 2.25% |
| 6 | `/model.22/dfl/Softmax` | Softmax | 213.9720 | 2.23% |
| 7 | `/model.22/cv2.0/cv2.0.0/conv/Conv || /model.22/cv3.0/cv3.0.0/conv/Conv` | FusedParallel | 205.2490 | 2.14% |
| 8 | `/model.1/conv/Conv + PWN(PWN(/model.1/act/Sigmoid), /model.1/act/Mul)` | Conv | 177.7450 | 1.85% |
| 9 | `/model.12/m.0/cv1/conv/Conv + PWN(PWN(/model.12/m.0/cv1/act/Sigmoid), /model.12/m.0/cv1...` | Conv | 172.1860 | 1.79% |
| 10 | `/model.22/cv2.2/cv2.2.0/conv/Conv || /model.22/cv3.2/cv3.2.0/conv/Conv` | FusedParallel | 170.5840 | 1.78% |
| 11 | `PWN(PWN(/model.0/act/Sigmoid), /model.0/act/Mul)` | PointWise | 144.9230 | 1.51% |
| 12 | `/model.2/m.0/Add_output_0 copy` | Reformat | 132.3530 | 1.38% |
| 13 | `/model.6/m.1/cv1/conv/Conv + PWN(PWN(/model.6/m.1/cv1/act/Sigmoid), /model.6/m.1/cv1/ac...` | Conv | 120.8210 | 1.26% |
| 14 | `/model.2/m.0/cv1/conv/Conv + PWN(PWN(/model.2/m.0/cv1/act/Sigmoid), /model.2/m.0/cv1/ac...` | Conv | 107.1020 | 1.11% |
| 15 | `/model.3/conv/Conv + PWN(PWN(/model.3/act/Sigmoid), /model.3/act/Mul)` | Conv | 106.5220 | 1.11% |

## 按类型汇总表

| Type | Count | Total Time (ms) | Time % |
|------|-------|-----------------|--------|
| Conv | 78 | 7019.9097 | 73.07% |
| Reformat | 69 | 1078.9950 | 11.23% |
| PointWise | 21 | 553.0082 | 5.76% |
| FusedParallel | 3 | 475.7236 | 4.95% |
| Softmax | 1 | 213.9720 | 2.23% |
| Reshape | 5 | 98.4398 | 1.02% |
| Shuffle | 3 | 56.8615 | 0.59% |
| Resize | 2 | 44.1619 | 0.46% |
| MaxPool | 3 | 41.5384 | 0.43% |
| Other | 6 | 23.9401 | 0.25% |

## 完整逐层性能表

| # | Layer Name | Type | Time (us) | Avg (ms) | Median (ms) | Time % |
|---|------------|------|-----------|----------|-------------|--------|
| 1 | `/model.22/dfl/conv/Conv` | Conv | 1183140.00 | 1.9719 | 0.0530 | 12.32% |
| 2 | `/model.22/cv3.0/cv3.0.1/conv/Conv + PWN(PWN(/model.22/cv3.0/cv3.0.1/act/...` | Conv | 475783.00 | 0.7930 | 0.2805 | 4.95% |
| 3 | `/model.2/cv2/conv/Conv + PWN(PWN(/model.2/cv2/act/Sigmoid), /model.2/cv2...` | Conv | 355489.00 | 0.5925 | 0.1676 | 3.70% |
| 4 | `/model.6/m.0/cv2/conv/Conv` | Conv | 257394.00 | 0.4290 | 0.0824 | 2.68% |
| 5 | `/model.12/m.0/cv2/conv/Conv + PWN(PWN(/model.12/m.0/cv2/act/Sigmoid), /m...` | Conv | 216498.00 | 0.3608 | 0.0940 | 2.25% |
| 6 | `/model.22/dfl/Softmax` | Softmax | 213972.00 | 0.3566 | 0.3440 | 2.23% |
| 7 | `/model.22/cv2.0/cv2.0.0/conv/Conv || /model.22/cv3.0/cv3.0.0/conv/Conv` | FusedParallel | 205249.00 | 0.3421 | 0.3143 | 2.14% |
| 8 | `/model.1/conv/Conv + PWN(PWN(/model.1/act/Sigmoid), /model.1/act/Mul)` | Conv | 177745.00 | 0.2962 | 0.2945 | 1.85% |
| 9 | `/model.12/m.0/cv1/conv/Conv + PWN(PWN(/model.12/m.0/cv1/act/Sigmoid), /m...` | Conv | 172186.00 | 0.2870 | 0.0973 | 1.79% |
| 10 | `/model.22/cv2.2/cv2.2.0/conv/Conv || /model.22/cv3.2/cv3.2.0/conv/Conv` | FusedParallel | 170584.00 | 0.2843 | 0.0776 | 1.78% |
| 11 | `PWN(PWN(/model.0/act/Sigmoid), /model.0/act/Mul)` | PointWise | 144923.00 | 0.2415 | 0.2398 | 1.51% |
| 12 | `/model.2/m.0/Add_output_0 copy` | Reformat | 132353.00 | 0.2206 | 0.1136 | 1.38% |
| 13 | `/model.6/m.1/cv1/conv/Conv + PWN(PWN(/model.6/m.1/cv1/act/Sigmoid), /mod...` | Conv | 120821.00 | 0.2014 | 0.0980 | 1.26% |
| 14 | `/model.2/m.0/cv1/conv/Conv + PWN(PWN(/model.2/m.0/cv1/act/Sigmoid), /mod...` | Conv | 107102.00 | 0.1785 | 0.1776 | 1.11% |
| 15 | `/model.3/conv/Conv + PWN(PWN(/model.3/act/Sigmoid), /model.3/act/Mul)` | Conv | 106522.00 | 0.1775 | 0.1765 | 1.11% |
| 16 | `/model.2/m.1/cv1/conv/Conv + PWN(PWN(/model.2/m.1/cv1/act/Sigmoid), /mod...` | Conv | 101018.00 | 0.1684 | 0.1675 | 1.05% |
| 17 | `/model.22/cv2.1/cv2.1.0/conv/Conv || /model.22/cv3.1/cv3.1.0/conv/Conv` | FusedParallel | 99890.60 | 0.1665 | 0.1616 | 1.04% |
| 18 | `/model.7/conv/Conv + PWN(PWN(/model.7/act/Sigmoid), /model.7/act/Mul)` | Conv | 99478.50 | 0.1658 | 0.1644 | 1.04% |
| 19 | `/model.22/dfl/Reshape + /model.22/dfl/Transpose` | Reshape | 98439.80 | 0.1641 | 0.0647 | 1.02% |
| 20 | `/model.0/conv/Conv` | Conv | 94123.40 | 0.1569 | 0.1513 | 0.98% |
| 21 | `/model.4/cv2/conv/Conv + PWN(PWN(/model.4/cv2/act/Sigmoid), /model.4/cv2...` | Conv | 91710.50 | 0.1529 | 0.1408 | 0.95% |
| 22 | `/model.6/m.0/cv1/conv/Conv + PWN(PWN(/model.6/m.0/cv1/act/Sigmoid), /mod...` | Conv | 89427.90 | 0.1490 | 0.0975 | 0.93% |
| 23 | `/model.21/cv2/conv/Conv + PWN(PWN(/model.21/cv2/act/Sigmoid), /model.21/...` | Conv | 88891.30 | 0.1482 | 0.0793 | 0.93% |
| 24 | `/model.2/m.0/cv2/conv/Conv` | Conv | 88564.70 | 0.1476 | 0.1468 | 0.92% |
| 25 | `/model.12/cv1/conv/Conv + PWN(PWN(/model.12/cv1/act/Sigmoid), /model.12/...` | Conv | 88096.50 | 0.1468 | 0.0913 | 0.92% |
| 26 | `/model.5/conv/Conv + PWN(PWN(/model.5/act/Sigmoid), /model.5/act/Mul)` | Conv | 86742.50 | 0.1446 | 0.1422 | 0.90% |
| 27 | `/model.15/cv1/conv/Conv + PWN(PWN(/model.15/cv1/act/Sigmoid), /model.15/...` | Conv | 85191.00 | 0.1420 | 0.1412 | 0.89% |
| 28 | `/model.13/Resize_output_0 copy` | Reformat | 85081.10 | 0.1418 | 0.1410 | 0.89% |
| 29 | `/model.2/m.1/cv2/conv/Conv` | Conv | 84247.30 | 0.1404 | 0.1396 | 0.88% |
| 30 | `/model.2/Split_output_1 copy` | Reformat | 76665.40 | 0.1278 | 0.1136 | 0.80% |
| 31 | `/model.22/cv3.2/cv3.2.1/conv/Conv + PWN(PWN(/model.22/cv3.2/cv3.2.1/act/...` | Conv | 74936.40 | 0.1249 | 0.0457 | 0.78% |
| 32 | `/model.2/cv1/conv/Conv + PWN(PWN(/model.2/cv1/act/Sigmoid), /model.2/cv1...` | Conv | 71212.00 | 0.1187 | 0.1180 | 0.74% |
| 33 | `/model.22/cv2.0/cv2.0.1/conv/Conv + PWN(PWN(/model.22/cv2.0/cv2.0.1/act/...` | Conv | 70008.70 | 0.1167 | 0.0536 | 0.73% |
| 34 | `/model.2/Split_output_0 copy` | Reformat | 69411.30 | 0.1157 | 0.1150 | 0.72% |
| 35 | `/model.6/cv2/conv/Conv + PWN(PWN(/model.6/cv2/act/Sigmoid), /model.6/cv2...` | Conv | 67402.40 | 0.1123 | 0.1093 | 0.70% |
| 36 | `/model.15/m.0/cv1/conv/Conv + PWN(PWN(/model.15/m.0/cv1/act/Sigmoid), /m...` | Conv | 66444.00 | 0.1107 | 0.1102 | 0.69% |
| 37 | `/model.15/cv2/conv/Conv + PWN(PWN(/model.15/cv2/act/Sigmoid), /model.15/...` | Conv | 65411.60 | 0.1090 | 0.1085 | 0.68% |
| 38 | `/model.4/m.0/cv1/conv/Conv + PWN(PWN(/model.4/m.0/cv1/act/Sigmoid), /mod...` | Conv | 65032.90 | 0.1084 | 0.1078 | 0.68% |
| 39 | `Reformatting CopyNode for Input Tensor 0 to /model.0/conv/Conv` | Reformat | 64743.30 | 0.1079 | 0.1052 | 0.67% |
| 40 | `/model.15/m.0/cv2/conv/Conv + PWN(PWN(/model.15/m.0/cv2/act/Sigmoid), /m...` | Conv | 64168.20 | 0.1069 | 0.1063 | 0.67% |
| 41 | `/model.12/m.1/cv1/conv/Conv + PWN(PWN(/model.12/m.1/cv1/act/Sigmoid), /m...` | Conv | 63555.80 | 0.1059 | 0.0975 | 0.66% |
| 42 | `/model.4/m.1/cv1/conv/Conv + PWN(PWN(/model.4/m.1/cv1/act/Sigmoid), /mod...` | Conv | 63274.70 | 0.1055 | 0.1049 | 0.66% |
| 43 | `/model.22/cv3.1/cv3.1.1/conv/Conv + PWN(PWN(/model.22/cv3.1/cv3.1.1/act/...` | Conv | 63124.30 | 0.1052 | 0.0980 | 0.66% |
| 44 | `/model.15/m.1/cv2/conv/Conv + PWN(PWN(/model.15/m.1/cv2/act/Sigmoid), /m...` | Conv | 63059.40 | 0.1051 | 0.1032 | 0.66% |
| 45 | `/model.15/m.1/cv1/conv/Conv + PWN(PWN(/model.15/m.1/cv1/act/Sigmoid), /m...` | Conv | 62794.00 | 0.1047 | 0.1041 | 0.65% |
| 46 | `/model.4/m.2/cv1/conv/Conv + PWN(PWN(/model.4/m.2/cv1/act/Sigmoid), /mod...` | Conv | 62724.20 | 0.1045 | 0.1040 | 0.65% |
| 47 | `/model.4/m.3/cv1/conv/Conv + PWN(PWN(/model.4/m.3/cv1/act/Sigmoid), /mod...` | Conv | 62723.20 | 0.1045 | 0.1040 | 0.65% |
| 48 | `/model.16/conv/Conv + PWN(PWN(/model.16/act/Sigmoid), /model.16/act/Mul)` | Conv | 62510.00 | 0.1042 | 0.1006 | 0.65% |
| 49 | `/model.19/conv/Conv + PWN(PWN(/model.19/act/Sigmoid), /model.19/act/Mul)` | Conv | 61280.80 | 0.1021 | 0.1016 | 0.64% |
| 50 | `/model.18/m.0/cv1/conv/Conv + PWN(PWN(/model.18/m.0/cv1/act/Sigmoid), /m...` | Conv | 60014.50 | 0.1000 | 0.0995 | 0.62% |
| 51 | `/model.18/m.1/cv1/conv/Conv + PWN(PWN(/model.18/m.1/cv1/act/Sigmoid), /m...` | Conv | 59352.00 | 0.0989 | 0.0984 | 0.62% |
| 52 | `/model.18/m.0/cv2/conv/Conv + PWN(PWN(/model.18/m.0/cv2/act/Sigmoid), /m...` | Conv | 58493.30 | 0.0975 | 0.0970 | 0.61% |
| 53 | `/model.4/m.0/cv2/conv/Conv` | Conv | 57682.40 | 0.0961 | 0.0956 | 0.60% |
| 54 | `(Unnamed Layer* 296) [Shuffle] + /model.22/dfl/Transpose_1` | Shuffle | 56861.50 | 0.0948 | 0.0943 | 0.59% |
| 55 | `/model.6/m.2/cv1/conv/Conv + PWN(PWN(/model.6/m.2/cv1/act/Sigmoid), /mod...` | Conv | 56706.50 | 0.0945 | 0.0940 | 0.59% |
| 56 | `/model.18/m.1/cv2/conv/Conv + PWN(PWN(/model.18/m.1/cv2/act/Sigmoid), /m...` | Conv | 56664.80 | 0.0944 | 0.0940 | 0.59% |
| 57 | `/model.6/m.3/cv1/conv/Conv + PWN(PWN(/model.6/m.3/cv1/act/Sigmoid), /mod...` | Conv | 56642.70 | 0.0944 | 0.0940 | 0.59% |
| 58 | `/model.4/m.1/cv2/conv/Conv` | Conv | 56547.40 | 0.0942 | 0.0937 | 0.59% |
| 59 | `/model.4/m.2/cv2/conv/Conv` | Conv | 56487.50 | 0.0941 | 0.0936 | 0.59% |
| 60 | `/model.12/m.1/cv2/conv/Conv + PWN(PWN(/model.12/m.1/cv2/act/Sigmoid), /m...` | Conv | 56371.30 | 0.0940 | 0.0932 | 0.59% |
| 61 | `/model.8/m.1/cv2/conv/Conv` | Conv | 55928.00 | 0.0932 | 0.0862 | 0.58% |
| 62 | `/model.4/m.3/cv2/conv/Conv` | Conv | 55895.20 | 0.0932 | 0.0926 | 0.58% |
| 63 | `/model.8/m.0/cv2/conv/Conv` | Conv | 54570.90 | 0.0910 | 0.0906 | 0.57% |
| 64 | `/model.18/cv1/conv/Conv + PWN(PWN(/model.18/cv1/act/Sigmoid), /model.18/...` | Conv | 51181.70 | 0.0853 | 0.0868 | 0.53% |
| 65 | `PWN(PWN(PWN(/model.2/m.1/cv2/act/Sigmoid), /model.2/m.1/cv2/act/Mul), /m...` | PointWise | 49328.80 | 0.0822 | 0.0787 | 0.51% |
| 66 | `/model.9/cv2/conv/Conv + PWN(PWN(/model.9/cv2/act/Sigmoid), /model.9/cv2...` | Conv | 49087.10 | 0.0818 | 0.0775 | 0.51% |
| 67 | `/model.4/cv1/conv/Conv + PWN(PWN(/model.4/cv1/act/Sigmoid), /model.4/cv1...` | Conv | 49051.90 | 0.0818 | 0.0814 | 0.51% |
| 68 | `/model.21/m.0/cv1/conv/Conv + PWN(PWN(/model.21/m.0/cv1/act/Sigmoid), /m...` | Conv | 49038.10 | 0.0817 | 0.0773 | 0.51% |
| 69 | `/model.6/m.1/cv2/conv/Conv` | Conv | 48624.10 | 0.0810 | 0.0805 | 0.51% |
| 70 | `/model.6/m.2/cv2/conv/Conv` | Conv | 48426.20 | 0.0807 | 0.0803 | 0.50% |
| 71 | `/model.6/m.3/cv2/conv/Conv` | Conv | 48411.10 | 0.0807 | 0.0802 | 0.50% |
| 72 | `/model.21/m.1/cv1/conv/Conv + PWN(PWN(/model.21/m.1/cv1/act/Sigmoid), /m...` | Conv | 48143.60 | 0.0802 | 0.0742 | 0.50% |
| 73 | `/model.18/cv2/conv/Conv + PWN(PWN(/model.18/cv2/act/Sigmoid), /model.18/...` | Conv | 47784.50 | 0.0796 | 0.0793 | 0.50% |
| 74 | `/model.12/cv2/conv/Conv + PWN(PWN(/model.12/cv2/act/Sigmoid), /model.12/...` | Conv | 47641.40 | 0.0794 | 0.0790 | 0.50% |
| 75 | `PWN(PWN(PWN(/model.2/m.0/cv2/act/Sigmoid), /model.2/m.0/cv2/act/Mul), /m...` | PointWise | 47587.40 | 0.0793 | 0.0788 | 0.50% |
| 76 | `/model.21/m.1/cv2/conv/Conv + PWN(PWN(/model.21/m.1/cv2/act/Sigmoid), /m...` | Conv | 47445.10 | 0.0791 | 0.0754 | 0.49% |
| 77 | `/model.8/cv2/conv/Conv + PWN(PWN(/model.8/cv2/act/Sigmoid), /model.8/cv2...` | Conv | 47302.60 | 0.0788 | 0.0785 | 0.49% |
| 78 | `/model.21/cv1/conv/Conv + PWN(PWN(/model.21/cv1/act/Sigmoid), /model.21/...` | Conv | 46004.30 | 0.0767 | 0.0728 | 0.48% |
| 79 | `/model.8/m.1/cv1/conv/Conv + PWN(PWN(/model.8/m.1/cv1/act/Sigmoid), /mod...` | Conv | 45819.70 | 0.0764 | 0.0759 | 0.48% |
| 80 | `/model.8/m.0/cv1/conv/Conv + PWN(PWN(/model.8/m.0/cv1/act/Sigmoid), /mod...` | Conv | 45382.80 | 0.0756 | 0.0752 | 0.47% |
| 81 | `/model.21/m.0/cv2/conv/Conv + PWN(PWN(/model.21/m.0/cv2/act/Sigmoid), /m...` | Conv | 44130.60 | 0.0736 | 0.0732 | 0.46% |
| 82 | `/model.12/cv2/act/Mul_output_0 copy` | Reformat | 41126.20 | 0.0685 | 0.0453 | 0.43% |
| 83 | `PWN(PWN(/model.22/cv3.0/cv3.0.0/act/Sigmoid), /model.22/cv3.0/cv3.0.0/ac...` | PointWise | 39319.40 | 0.0655 | 0.0516 | 0.41% |
| 84 | `Reformatting CopyNode for Input Tensor 0 to /model.22/Reshape` | Reformat | 39302.10 | 0.0655 | 0.0652 | 0.41% |
| 85 | `PWN(PWN(PWN(/model.6/m.0/cv2/act/Sigmoid), /model.6/m.0/cv2/act/Mul), /m...` | PointWise | 37385.20 | 0.0623 | 0.0211 | 0.39% |
| 86 | `/model.6/cv1/conv/Conv + PWN(PWN(/model.6/cv1/act/Sigmoid), /model.6/cv1...` | Conv | 36191.40 | 0.0603 | 0.0564 | 0.38% |
| 87 | `/model.10/Resize_output_0 copy` | Reformat | 35691.80 | 0.0595 | 0.0580 | 0.37% |
| 88 | `/model.13/Resize` | Resize | 31761.00 | 0.0529 | 0.0526 | 0.33% |
| 89 | `/model.22/cv3.0/cv3.0.2/Conv` | Conv | 31291.20 | 0.0522 | 0.0519 | 0.33% |
| 90 | `PWN(/model.22/Sigmoid)` | PointWise | 28629.60 | 0.0477 | 0.0476 | 0.30% |
| 91 | `/model.8/cv1/conv/Conv + PWN(PWN(/model.8/cv1/act/Sigmoid), /model.8/cv1...` | Conv | 28428.10 | 0.0474 | 0.0471 | 0.30% |
| 92 | `/model.22/cv2.2/cv2.2.1/conv/Conv + PWN(PWN(/model.22/cv2.2/cv2.2.1/act/...` | Conv | 27743.20 | 0.0462 | 0.0207 | 0.29% |
| 93 | `/model.15/Split_output_0 copy` | Reformat | 27217.10 | 0.0454 | 0.0451 | 0.28% |
| 94 | `/model.4/Split_output_0 copy` | Reformat | 27197.00 | 0.0453 | 0.0452 | 0.28% |
| 95 | `PWN(PWN(PWN(/model.4/m.2/cv2/act/Sigmoid), /model.4/m.2/cv2/act/Mul), /m...` | PointWise | 26281.60 | 0.0438 | 0.0389 | 0.27% |
| 96 | `/model.4/Split_output_1 copy` | Reformat | 26057.00 | 0.0434 | 0.0432 | 0.27% |
| 97 | `/model.4/m.1/Add_output_0 copy` | Reformat | 26051.30 | 0.0434 | 0.0432 | 0.27% |
| 98 | `/model.15/Split_output_1 copy` | Reformat | 25992.60 | 0.0433 | 0.0431 | 0.27% |
| 99 | `/model.4/m.0/Add_output_0 copy` | Reformat | 25911.20 | 0.0432 | 0.0430 | 0.27% |
| 100 | `/model.4/m.2/Add_output_0 copy` | Reformat | 25883.20 | 0.0431 | 0.0429 | 0.27% |
| 101 | `PWN(PWN(PWN(/model.4/m.1/cv2/act/Sigmoid), /model.4/m.1/cv2/act/Mul), /m...` | PointWise | 23899.70 | 0.0398 | 0.0397 | 0.25% |
| 102 | `PWN(PWN(PWN(/model.4/m.0/cv2/act/Sigmoid), /model.4/m.0/cv2/act/Mul), /m...` | PointWise | 23288.70 | 0.0388 | 0.0386 | 0.24% |
| 103 | `/model.22/cv3.2/cv3.2.2/Conv` | Conv | 22621.90 | 0.0377 | 0.0165 | 0.24% |
| 104 | `PWN(PWN(PWN(/model.4/m.3/cv2/act/Sigmoid), /model.4/m.3/cv2/act/Mul), /m...` | PointWise | 22566.70 | 0.0376 | 0.0374 | 0.23% |
| 105 | `/model.9/cv1/conv/Conv + PWN(PWN(/model.9/cv1/act/Sigmoid), /model.9/cv1...` | Conv | 21001.60 | 0.0350 | 0.0326 | 0.22% |
| 106 | `Reformatting CopyNode for Output Tensor 0 to PWN(/model.22/Sigmoid)` | Reformat | 19248.70 | 0.0321 | 0.0290 | 0.20% |
| 107 | `/model.22/Reshape_2_copy_output` | Reformat | 17775.00 | 0.0296 | 0.0099 | 0.19% |
| 108 | `/model.18/Split_output_0 copy` | Reformat | 17724.50 | 0.0295 | 0.0295 | 0.18% |
| 109 | `/model.12/Split_output_0 copy` | Reformat | 17656.10 | 0.0294 | 0.0293 | 0.18% |
| 110 | `/model.6/Split_output_0 copy` | Reformat | 17422.10 | 0.0290 | 0.0290 | 0.18% |
| 111 | `/model.22/cv2.0/cv2.0.2/Conv` | Conv | 16810.10 | 0.0280 | 0.0280 | 0.17% |
| 112 | `/model.6/Split_output_1 copy` | Reformat | 16804.20 | 0.0280 | 0.0279 | 0.17% |
| 113 | `/model.6/m.2/Add_output_0 copy` | Reformat | 16425.40 | 0.0274 | 0.0273 | 0.17% |
| 114 | `/model.6/m.0/Add_output_0 copy` | Reformat | 16402.70 | 0.0273 | 0.0273 | 0.17% |
| 115 | `/model.12/Split_output_1 copy` | Reformat | 16384.70 | 0.0273 | 0.0272 | 0.17% |
| 116 | `/model.18/Split_output_1 copy` | Reformat | 16375.40 | 0.0273 | 0.0272 | 0.17% |
| 117 | `/model.6/m.1/Add_output_0 copy` | Reformat | 16098.80 | 0.0268 | 0.0268 | 0.17% |
| 118 | `Reformatting CopyNode for Input Tensor 0 to /model.22/dfl/Softmax` | Reformat | 15625.60 | 0.0260 | 0.0203 | 0.16% |
| 119 | `/model.22/cv2.1/cv2.1.1/conv/Conv + PWN(PWN(/model.22/cv2.1/cv2.1.1/act/...` | Conv | 15616.30 | 0.0260 | 0.0259 | 0.16% |
| 120 | `/model.22/Reshape_copy_output` | Reformat | 15574.20 | 0.0260 | 0.0258 | 0.16% |
| 121 | `/model.9/cv2/act/Mul_output_0 copy` | Reformat | 15312.50 | 0.0255 | 0.0254 | 0.16% |
| 122 | `/model.21/Split_output_1 copy` | Reformat | 15100.20 | 0.0252 | 0.0229 | 0.16% |
| 123 | `PWN(PWN(/model.22/cv2.0/cv2.0.0/act/Sigmoid), /model.22/cv2.0/cv2.0.0/ac...` | PointWise | 14858.70 | 0.0248 | 0.0221 | 0.15% |
| 124 | `/model.9/m/MaxPool` | MaxPool | 14126.60 | 0.0235 | 0.0235 | 0.15% |
| 125 | `/model.9/m_2/MaxPool` | MaxPool | 13733.30 | 0.0229 | 0.0228 | 0.14% |
| 126 | `/model.9/m_1/MaxPool` | MaxPool | 13678.50 | 0.0228 | 0.0227 | 0.14% |
| 127 | `Reformatting CopyNode for Input Tensor 0 to /model.22/Reshape_2` | Reformat | 13667.50 | 0.0228 | 0.0108 | 0.14% |
| 128 | `/model.22/cv3.1/cv3.1.2/Conv` | Conv | 13502.50 | 0.0225 | 0.0224 | 0.14% |
| 129 | `/model.10/Resize` | Resize | 12400.90 | 0.0207 | 0.0204 | 0.13% |
| 130 | `PWN(PWN(PWN(/model.6/m.1/cv2/act/Sigmoid), /model.6/m.1/cv2/act/Mul), /m...` | PointWise | 12369.00 | 0.0206 | 0.0205 | 0.13% |
| 131 | `PWN(PWN(PWN(/model.6/m.3/cv2/act/Sigmoid), /model.6/m.3/cv2/act/Mul), /m...` | PointWise | 12219.30 | 0.0204 | 0.0203 | 0.13% |
| 132 | `PWN(PWN(PWN(/model.6/m.2/cv2/act/Sigmoid), /model.6/m.2/cv2/act/Mul), /m...` | PointWise | 12140.70 | 0.0202 | 0.0201 | 0.13% |
| 133 | `/model.22/cv2.2/cv2.2.2/Conv` | Conv | 11705.10 | 0.0195 | 0.0143 | 0.12% |
| 134 | `Reformatting CopyNode for Input Tensor 0 to /model.22/Reshape_1` | Reformat | 10708.20 | 0.0178 | 0.0178 | 0.11% |
| 135 | `/model.9/m_1/MaxPool_output_0 copy` | Reformat | 10496.20 | 0.0175 | 0.0152 | 0.11% |
| 136 | `/model.8/Split_output_0 copy` | Reformat | 10438.80 | 0.0174 | 0.0174 | 0.11% |
| 137 | `PWN(PWN(/model.22/cv3.2/cv3.2.0/act/Sigmoid), /model.22/cv3.2/cv3.2.0/ac...` | PointWise | 10424.60 | 0.0174 | 0.0097 | 0.11% |
| 138 | `/model.22/cv2.1/cv2.1.2/Conv` | Conv | 10332.80 | 0.0172 | 0.0172 | 0.11% |
| 139 | `/model.8/Split_output_1 copy` | Reformat | 10320.40 | 0.0172 | 0.0171 | 0.11% |
| 140 | `/model.21/Split_output_0 copy` | Reformat | 10291.10 | 0.0172 | 0.0171 | 0.11% |
| 141 | `PWN(PWN(/model.22/cv2.2/cv2.2.0/act/Sigmoid), /model.22/cv2.2/cv2.2.0/ac...` | PointWise | 10075.40 | 0.0168 | 0.0091 | 0.10% |
| 142 | `/model.8/m.0/Add_output_0 copy` | Reformat | 9173.91 | 0.0153 | 0.0152 | 0.10% |
| 143 | `/model.9/m/MaxPool_output_0 copy` | Reformat | 9146.11 | 0.0152 | 0.0152 | 0.10% |
| 144 | `/model.9/cv1/act/Mul_output_0 copy` | Reformat | 9140.45 | 0.0152 | 0.0152 | 0.10% |
| 145 | `PWN(PWN(/model.22/cv3.1/cv3.1.0/act/Sigmoid), /model.22/cv3.1/cv3.1.0/ac...` | PointWise | 8982.71 | 0.0150 | 0.0149 | 0.09% |
| 146 | `PWN(PWN(PWN(/model.8/m.1/cv2/act/Sigmoid), /model.8/m.1/cv2/act/Mul), /m...` | PointWise | 8456.06 | 0.0141 | 0.0140 | 0.09% |
| 147 | `PWN(PWN(PWN(/model.8/m.0/cv2/act/Sigmoid), /model.8/m.0/cv2/act/Mul), /m...` | PointWise | 8009.03 | 0.0133 | 0.0133 | 0.08% |
| 148 | `/model.22/Mul_2` | Other | 7600.71 | 0.0127 | 0.0126 | 0.08% |
| 149 | `/model.22/Reshape_1_copy_output` | Reformat | 6998.60 | 0.0117 | 0.0116 | 0.07% |
| 150 | `PWN(PWN(/model.22/cv2.1/cv2.1.0/act/Sigmoid), /model.22/cv2.1/cv2.1.0/ac...` | PointWise | 6809.98 | 0.0114 | 0.0112 | 0.07% |
| 151 | `/model.22/Sub` | Other | 6327.14 | 0.0105 | 0.0104 | 0.07% |
| 152 | `PWN(/model.22/Constant_11_output_0 + (Unnamed Layer* 362) [Shuffle], PWN...` | PointWise | 5452.61 | 0.0091 | 0.0091 | 0.06% |
| 153 | `/model.22/Add_1` | Other | 5304.77 | 0.0088 | 0.0089 | 0.06% |
| 154 | `/model.22/Sub_1` | Other | 4707.49 | 0.0078 | 0.0078 | 0.05% |
| 155 | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.0/act/Sigmoid...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 156 | `Reformatting CopyNode for Input Tensor 0 to /model.1/conv/Conv + PWN(PWN...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 157 | `Reformatting CopyNode for Output Tensor 0 to PWN(PWN(PWN(/model.2/m.0/cv...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 158 | `Reformatting CopyNode for Input Tensor 0 to /model.2/m.1/cv1/conv/Conv +...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 159 | `Reformatting CopyNode for Input Tensor 1 to PWN(PWN(PWN(/model.2/m.1/cv2...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 160 | `Reformatting CopyNode for Output Tensor 0 to PWN(PWN(PWN(/model.4/m.0/cv...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 161 | `Reformatting CopyNode for Input Tensor 0 to /model.4/m.1/cv1/conv/Conv +...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 162 | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(PWN(/model.4/m.1/cv2...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 163 | `Reformatting CopyNode for Input Tensor 0 to /model.4/m.2/cv1/conv/Conv +...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 164 | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(PWN(/model.4/m.2/cv2...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 165 | `Reformatting CopyNode for Output Tensor 0 to PWN(PWN(PWN(/model.4/m.2/cv...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 166 | `Reformatting CopyNode for Input Tensor 0 to /model.4/m.3/cv1/conv/Conv +...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 167 | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(PWN(/model.4/m.3/cv2...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 168 | `Reformatting CopyNode for Input Tensor 1 to PWN(PWN(PWN(/model.4/m.3/cv2...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 169 | `Reformatting CopyNode for Input Tensor 0 to /model.4/cv2/conv/Conv + PWN...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 170 | `Reformatting CopyNode for Output Tensor 0 to PWN(PWN(PWN(/model.6/m.0/cv...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 171 | `Reformatting CopyNode for Input Tensor 0 to /model.6/m.1/cv1/conv/Conv +...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 172 | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(PWN(/model.6/m.1/cv2...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 173 | `Reformatting CopyNode for Output Tensor 0 to PWN(PWN(PWN(/model.6/m.1/cv...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 174 | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(PWN(/model.6/m.2/cv2...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 175 | `Reformatting CopyNode for Input Tensor 1 to PWN(PWN(PWN(/model.6/m.2/cv2...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 176 | `Reformatting CopyNode for Output Tensor 0 to PWN(PWN(PWN(/model.6/m.2/cv...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 177 | `Reformatting CopyNode for Input Tensor 0 to PWN(PWN(PWN(/model.8/m.1/cv2...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 178 | `Reformatting CopyNode for Input Tensor 1 to PWN(PWN(PWN(/model.8/m.1/cv2...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 179 | `Reformatting CopyNode for Input Tensor 0 to /model.8/cv2/conv/Conv + PWN...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 180 | `Reformatting CopyNode for Output Tensor 0 to /model.9/cv2/conv/Conv + PW...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 181 | `Reformatting CopyNode for Input Tensor 0 to /model.10/Resize` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 182 | `Reformatting CopyNode for Output Tensor 0 to /model.12/cv2/conv/Conv + P...` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 183 | `Reformatting CopyNode for Input Tensor 0 to /model.13/Resize` | Reformat | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 184 | `/model.22/Reshape` | Reshape | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 185 | `/model.22/Reshape_1` | Reshape | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 186 | `/model.22/Reshape_2` | Reshape | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 187 | `(Unnamed Layer* 294) [Shuffle]` | Shuffle | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 188 | `/model.22/dfl/Reshape_1` | Reshape | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 189 | `/model.22/Constant_9_output_0` | Other | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 190 | `/model.22/Constant_10_output_0` | Other | 0.00 | 0.0000 | 0.0000 | 0.00% |
| 191 | `/model.22/Constant_12_output_0 + (Unnamed Layer* 367) [Shuffle]` | Shuffle | 0.00 | 0.0000 | 0.0000 | 0.00% |

## 性能瓶颈分析

### 卷积 (Conv) 分析

- **Conv 层数量**: 78
- **Conv 总耗时**: 7019.9097 ms (73.07%)
- **平均每层 Conv 耗时**: 89.9988 ms

| # | Layer Name | Time (ms) | Time % |
|---|------------|-----------|--------|
| 1 | `/model.22/dfl/conv/Conv` | 1183.1400 | 12.32% |
| 2 | `/model.22/cv3.0/cv3.0.1/conv/Conv + PWN(PWN(/model.22/cv3.0/cv3.0.1...` | 475.7830 | 4.95% |
| 3 | `/model.2/cv2/conv/Conv + PWN(PWN(/model.2/cv2/act/Sigmoid), /model....` | 355.4890 | 3.70% |
| 4 | `/model.6/m.0/cv2/conv/Conv` | 257.3940 | 2.68% |
| 5 | `/model.12/m.0/cv2/conv/Conv + PWN(PWN(/model.12/m.0/cv2/act/Sigmoid...` | 216.4980 | 2.25% |

  - 卷积运算占用绝大多数时间，符合计算密集型模型特征。
  - 优化方向：检查是否可启用 INT8 量化、使用更大的 batch size。

### Reformat / 数据排布转换分析

- **Reformat 层数量**: 69
- **Reformat 总耗时**: 1078.9950 ms (11.23%)
- **Reformat 开销较大**（>5%），建议仔细分析。
  - 可能原因：TensorRT 在 NHWC 与 NCHW 之间的格式转换。
  - 可通过设置 `FP16` 或 `INT8` 时使用 `set_nhwc_enabled()` 来减少。
  - Transpose / Shuffle 算子通常伴随 Reformat。检查是否需要
    使用 TensorRT 的 IOptimizationStrategy 进行融合优化。

### FusedParallel (并行融合) 分析

- **FusedParallel 层数量**: 3
- **FusedParallel 总耗时**: 475.7236 ms (4.95%)
  - FusedParallel 表示 TensorRT 将多个独立路径并行执行的算子。
  - 此类层通常出现在多分支网络结构（如 YOLO 的检测头分支）中。
  - 单个并行分支的耗时取决于其中最慢的分支。

### PointWise (逐点运算) 分析

- **PointWise 层数量**: 21
- **PointWise 总耗时**: 553.0082 ms (5.76%)
  - PointWise 算子通常包括激活函数和逐元素运算。
  - TensorRT 的算子融合策略应已将这些操作融合到 Conv 中。
  - 如果 PointWise 占比高且未融合，说明需要优化融合策略。

## 优化建议

### 1. 减少 Reformat / Transpose 开销

Reformat 占总时间超过 5%，建议分析 transpose 和 shuffle 操作的必要性。如果可以修改模型，考虑在设计上减少 reshape/transpose 操作。

### 2. 卷积核优化

卷积是主要瓶颈。考虑使用 TensorRT INT8 量化进一步加速。也可以尝试使用更大的 batch size（如 batch=4 或 8）来提高 GPU 利用率。

### 3. PointWise 算子融合

如果 PointWise 独立存在且未融合到卷积中，尝试调整 TensorRT 优化策略以启用更激进的算子融合。

### 4. 启用 CUDA Graph

TensorRT 支持 CUDA Graph 捕获和重放，可减少 kernel launch 开销。对于小 batch 推理，CUDA Graph 可显著提升吞吐量。

### 5. 多 Stream 并行推理

如果部署场景需要处理多路视频流，考虑使用多个 CUDA stream 并行执行引擎。

## 精度策略说明

本分析基于 FP16 engine (`yolov8m_fp16.engine`)。

| 项目 | 内容 |
|------|------|
| **Engine 路径** | `weights/engines/yolov8m_fp16.engine` |
| **精度模式** | FP16 (half precision) |
| **TensorRT 版本** | JetPack 预装版本 |
| **平台** | Jetson Orin (T234/Ampere 架构) |

### FP16 精度说明

- FP16 (半精度) 将权重和激活从 FP32 压缩到 16 位浮点数。
- 理论上可将显存占用和带宽需求减半，同时显著提升吞吐量。
- YOLOv8m 等现代检测模型在 FP16 下的精度损失通常 < 0.5% mAP。
- JetPack 内置的 TensorRT 已针对 Jetson Orin 的 FP16 Tensor Core 优化。
- 如需更高精度，可回退到 FP32；如需更高吞吐，可尝试 INT8 量化。

## 报告文件

- `layer_profile.json` — trtexec 导出的层性能数据 (JSON)
- `layer_info.json` — trtexec 导出的层结构信息 (JSON)
- `analysis_report.md` — 本报告
