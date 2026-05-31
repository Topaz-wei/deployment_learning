#!/bin/bash
# 一键运行全部 9 个 Transpose 实验
set -e
cd "$(dirname "$0")"

echo "=============================================="
echo "Transpose 实验套件 — 全部 9 个实验"
echo "=============================================="

if [[ "$CONDA_DEFAULT_ENV" != "py38" ]]; then
    echo "请先激活 py38 conda 环境: conda activate py38"
    exit 1
fi

echo ""
echo ">>> 第 1 步: 生成修改模型 <<<"
for exp in exp02_input_transpose exp03_output_transpose exp04_dual_transpose exp05_mid_transpose exp06_multi_transpose; do
    if [ -f "${exp}/modify_model.py" ]; then
        echo "--- ${exp} ---"
        python3 "${exp}/modify_model.py"
    fi
done

echo ""
echo ">>> 第 2 步: 运行实验 <<<"
for exp in exp01_baseline exp02_input_transpose exp03_output_transpose exp04_dual_transpose exp05_mid_transpose exp06_multi_transpose exp07_ort_opt_transpose exp08_transpose_conv_fusion exp09_resolution; do
    echo ""
    echo "=============================================="
    echo "  运行: ${exp}"
    echo "=============================================="
    python3 "${exp}/run.py"
done

echo ""
echo "=============================================="
echo "全部实验完成! 结果在 results/ 目录"
echo "=============================================="
ls -la results/
