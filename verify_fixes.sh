#!/bin/bash
# PRCC优化方案 - 完整修复验证脚本
# 在WSL的reid conda环境中运行此脚本

set -e

echo "============================================================"
echo "PRCC mAP优化方案 - 修复验证"
echo "============================================================"
echo ""

# 激活conda环境
echo "激活conda环境 'reid'..."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh || true
conda activate reid

# 切换到项目目录
cd /mnt/d/Code/PedestrianDetection

# 运行测试
echo ""
echo "运行单元测试..."
python test_fixes.py

# 检查修复后的代码
echo ""
echo "============================================================"
echo "验证关键修复"
echo "============================================================"
echo ""

echo "1. 检查双分类器标签偏移修复..."
grep -A 5 "def _prcc_local_labels" pedestrian_reid/engine/trainer.py | head -n 10

echo ""
echo "2. 检查检查点保存修复..."
grep -B 2 -A 2 "metric_value=selected_metric" pedestrian_reid/engine/trainer.py | head -n 5

echo ""
echo "3. 检查ExpT5配置修复..."
grep -A 5 "run_stage 5 train_model" run.sh | head -n 20

echo ""
echo "============================================================"
echo "✅ 验证完成"
echo "============================================================"
echo ""
echo "下一步操作:"
echo "1. 运行训练: bash run.sh 5"
echo "2. 监控训练日志: tail -f output/ExpT5/train.log"
echo "3. 查看mAP变化: cat output/ExpT5/evaluation_metrics.csv"
