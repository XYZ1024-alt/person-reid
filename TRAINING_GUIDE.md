# PRCC训练完整流程指南

## 📋 训练流程概览

你的训练是一个**5阶段渐进式迁移学习**流程：

```
Market-1501 (干净) → Market-1501 (增强) → Market+PRCC联合训练 → PRCC微调
     ExpT1               ExpT2, ExpT3                ExpT4              ExpT5
```

---

## 🎯 5个训练阶段详解

### **Stage 1: Market-1501基础训练** (ExpT1)
**目标**: 在干净的Market-1501数据集上训练基础模型

```bash
bash run.sh 1
# 或指定范围
START_STAGE=1 STOP_STAGE=1 bash run.sh
```

**训练配置**:
- 数据集: Market-1501 (751个身份)
- Epochs: 60
- 数据增强: 标准（翻转、裁剪、颜色抖动）
- 评估: Market-1501测试集

**输出**:
- 模型: `outputs/transfer/expT1_market_clean/best.pth`
- 日志: `outputs/transfer/expT1_market_clean/train.log`

---

### **Stage 2: Market-1501 + 暗光增强** (ExpT2)
**目标**: 从ExpT1检查点继续训练，增加暗光数据增强

```bash
bash run.sh 2
```

**训练配置**:
- 初始化: 从ExpT1的best.pth加载
- 数据增强: 标准 + **暗光增强**
- Epochs: 60
- 评估: Market-1501测试集

**输出**: `outputs/transfer/expT2_market_dark/best.pth`

---

### **Stage 3: Market-1501 + 遮挡增强** (ExpT3)
**目标**: 从ExpT2继续训练，增加遮挡数据增强

```bash
bash run.sh 3
```

**训练配置**:
- 初始化: 从ExpT2的best.pth加载
- 数据增强: 标准 + 暗光 + **遮挡增强**
- Epochs: 60
- 评估: Market-1501测试集

**输出**: `outputs/transfer/expT3_market_occlusion/best.pth`

---

### **Stage 4: Market+PRCC联合训练** (ExpT4) ⭐
**目标**: 联合训练Market-1501和PRCC，学习跨数据集特征

```bash
bash run.sh 4
```

**训练配置**:
- 初始化: 从ExpT3的best.pth加载
- 数据集: **Market-1501 + PRCC训练集（排除30人验证集）**
- 模式: `--mode joint`
- Epochs: 40
- 特殊功能:
  - 双分类器（Market头 + PRCC头）
  - Domain adversarial learning
  - Sketch-to-photo consistency
  - Knowledge distillation (教师: ExpT3)
- 评估: 
  - **PRCC-Dev (30人验证集)** ← 监控训练进度
  - Market-1501测试集
  - PRCC官方测试集（最终评估）

**输出**: `outputs/transfer/expT4_joint_v1/best.pth`

**关键点**:
- 这是训练最复杂的阶段
- PRCC-Dev用于超参数调优，避免过拟合到测试集
- 如果这个阶段的PRCC-Dev mAP从0.5下降到0.3，说明有问题

---

### **Stage 5: PRCC微调** (ExpT5) 🎯
**目标**: 在纯PRCC数据上微调，优化跨衣物重识别

```bash
bash run.sh 5
```

**训练配置** (已优化):
- 初始化: 从ExpT4的best.pth加载
- 数据集: **PRCC完整训练集**（200个身份，所有衣物变化）
- 模式: `--mode prcc`
- Epochs: 3
- 学习率: 1e-5 (微调专用)
- 损失函数:
  - ✅ Triplet loss (0.5)
  - ✅ **Cross-clothes contrastive loss (0.3)** ← 核心！
  - ✅ Knowledge distillation (0.05)
  - ❌ Part-triplet (0.0) - 已禁用
  - ❌ Cloth-invariant (0.0) - 已禁用
- 主干网络: **Layer4解冻，layer1-3冻结**
- 评估: **PRCC官方测试集** ← 这是你的最终成绩！

**输出**: 
- 模型: `outputs/transfer/expT5_prcc_finetune/best.pth`
- 评估: **mAP报告的是官方测试集结果**

---

## 🚀 使用MLflow追踪训练

### **启用MLflow**

MLflow默认是**关闭**的，需要手动开启：

```bash
# 方式1: 环境变量
export USE_MLFLOW=1
export MLFLOW_EXPERIMENT="prcc_optimization"
export MLFLOW_TRACKING_URI="file:./outputs/mlruns"
bash run.sh 5

# 方式2: 一行命令
USE_MLFLOW=1 MLFLOW_EXPERIMENT="prcc_fix_test" bash run.sh 5

# 方式3: 自定义run名称（推荐）
USE_MLFLOW=1 \
MLFLOW_EXPERIMENT="prcc_optimization" \
MLFLOW_RUN_NAME="exp5_fix_label_unfreeze_layer4" \
bash run.sh 5
```

### **MLflow记录内容**

MLflow会自动记录：

#### 1. **参数 (Parameters)**
- 所有训练超参数（学习率、batch size、损失权重等）
- Git commit hash和是否有未提交修改
- 数据集信息（类别数、batch数）

#### 2. **指标 (Metrics)**
每个epoch记录：
- **训练指标**: `train/loss`, `train/ce`, `train/triplet`, `train/cross_clothes_contrastive`等
- **评估指标**: `eval/prcc/standard/mAP`, `eval/prcc/standard/rank1`等

#### 3. **文件 (Artifacts)**
- 训练配置: `run_config.json`
- 训练指标CSV: `training_metrics.csv`
- 评估指标CSV: `evaluation_metrics.csv`
- 最佳模型: `checkpoints/best.pth`
- 源代码快照: `code/pedestrian_reid/`, `code/scripts/`

### **查看MLflow UI**

在WSL中启动MLflow UI：

```bash
cd /mnt/d/Code/PedestrianDetection
conda activate reid

# 启动UI
mlflow ui --backend-store-uri file:./outputs/mlruns --port 5000
```

然后在浏览器打开：`http://localhost:5000`

你可以：
- ✅ 对比不同run的mAP曲线
- ✅ 查看超参数对比表
- ✅ 下载训练日志和模型
- ✅ 追踪Git commit和代码版本

---

## 🔄 完整训练流程（推荐）

### **场景1: 从头开始完整训练**

```bash
cd /mnt/d/Code/PedestrianDetection
conda activate reid

# 训练所有阶段（约需几天时间）
USE_MLFLOW=1 \
MLFLOW_EXPERIMENT="full_training" \
START_STAGE=1 \
STOP_STAGE=5 \
bash run.sh
```

### **场景2: 只运行ExpT5（使用现有ExpT4检查点）** ⭐ **推荐**

```bash
# 假设你已经有ExpT4的检查点
USE_MLFLOW=1 \
MLFLOW_EXPERIMENT="prcc_optimization" \
MLFLOW_RUN_NAME="exp5_with_fixes" \
START_STAGE=5 \
STOP_STAGE=5 \
bash run.sh
```

### **场景3: 重新训练ExpT4和ExpT5**

```bash
# 从ExpT3的检查点开始
USE_MLFLOW=1 \
MLFLOW_EXPERIMENT="prcc_retrain" \
START_STAGE=4 \
STOP_STAGE=5 \
bash run.sh
```

### **场景4: 并行对比实验**

```bash
# Run 1: 使用修复后的配置
USE_MLFLOW=1 \
MLFLOW_EXPERIMENT="prcc_ablation" \
MLFLOW_RUN_NAME="fixed_config" \
bash run.sh 5

# Run 2: 使用原始配置（对比）
# 先恢复原run.sh配置，然后：
USE_MLFLOW=1 \
MLFLOW_EXPERIMENT="prcc_ablation" \
MLFLOW_RUN_NAME="original_config" \
bash run.sh 5
```

---

## 📊 监控训练进度

### **实时监控**

```bash
# 终端1: 训练日志
tail -f outputs/transfer/expT5_prcc_finetune/train.log

# 终端2: MLflow UI
mlflow ui --backend-store-uri file:./outputs/mlruns

# 终端3: 监控GPU
watch -n 1 nvidia-smi
```

### **检查关键指标**

训练结束后：

```bash
# 查看mAP历史
cat outputs/transfer/expT5_prcc_finetune/evaluation_metrics.csv | grep "mAP"

# 查看训练损失
cat outputs/transfer/expT5_prcc_finetune/training_metrics.csv | tail -5

# 查看最终结果
python -m scripts.evaluate \
  --checkpoint outputs/transfer/expT5_prcc_finetune/best.pth \
  --dataset prcc \
  --feature-key combined_features
```

---

## ⚙️ 高级配置选项

### **多GPU训练**

```bash
# 使用2个GPU
GPUS=2 \
USE_MLFLOW=1 \
bash run.sh 5

# 使用4个GPU
GPUS=4 USE_MLFLOW=1 bash run.sh 5
```

### **调整Batch Size和Workers**

```bash
# 增大batch size（需要更大GPU内存）
BATCH_SIZE=256 \
NUM_WORKERS=16 \
USE_MLFLOW=1 \
bash run.sh 5

# 减小batch size（GPU内存不足时）
BATCH_SIZE=64 \
NUM_WORKERS=8 \
USE_MLFLOW=1 \
bash run.sh 5
```

### **自定义输出目录**

```bash
# 修改实验输出根目录
EXP_ROOT="outputs/experiments_v2" \
USE_MLFLOW=1 \
bash run.sh 5
```

---

## 📂 输出文件结构

训练完成后，你会看到：

```
outputs/
├── mlruns/                           # MLflow追踪数据
│   └── 0/                            # 实验ID
│       ├── meta.yaml
│       └── <run_id>/                 # 每次run的数据
│           ├── params/               # 超参数
│           ├── metrics/              # 指标时间序列
│           ├── artifacts/            # 模型和日志
│           └── tags/                 # Git信息等
│
└── transfer/
    ├── expT1_market_clean/
    │   ├── best.pth                  # 最佳模型
    │   ├── last.pth                  # 最后一个epoch的模型
    │   ├── train.log                 # 训练日志
    │   ├── training_metrics.csv      # 训练损失
    │   ├── evaluation_metrics.csv    # 评估指标
    │   └── run_config.json           # 训练配置
    │
    ├── expT4_joint_v1/
    │   └── ...
    │
    └── expT5_prcc_finetune/          # ⭐ 最终结果
        ├── best.pth                  # 你的最佳PRCC模型
        ├── evaluation_metrics.csv    # 包含官方测试集mAP
        └── ...
```

---

## 🎯 关键决策点

### **何时使用MLflow？**

| 场景 | 建议 |
|------|------|
| 快速测试/调试 | ❌ 不需要MLflow |
| 单次完整训练 | ✅ 推荐使用 |
| 超参数对比实验 | ✅✅ 强烈推荐 |
| 论文实验记录 | ✅✅ 必须使用 |

### **何时重新训练ExpT4？**

如果满足以下任一条件，建议重新训练ExpT4：

- ✅ ExpT4的PRCC-Dev mAP下降严重（如0.53→0.36）
- ✅ ExpT5起点太低（<0.25）
- ✅ 修改了ExpT4的损失函数或架构
- ✅ 使用新的数据预处理

否则，直接使用现有ExpT4检查点，只训练ExpT5即可。

---

## 🐛 常见问题

### **Q1: MLflow UI显示"No runs"**

```bash
# 检查tracking URI
ls outputs/mlruns/0/

# 确认USE_MLFLOW=1
echo $USE_MLFLOW

# 重新启动UI并指定正确路径
mlflow ui --backend-store-uri file:./outputs/mlruns
```

### **Q2: 训练中断后如何恢复？**

```bash
# 检查last.pth是否存在
ls outputs/transfer/expT5_prcc_finetune/last.pth

# 修改run.sh，添加 --resume 参数
# 或者直接使用 --pretrained-checkpoint 指向last.pth
```

### **Q3: 多次训练如何区分？**

```bash
# 使用不同的MLFLOW_RUN_NAME
USE_MLFLOW=1 \
MLFLOW_RUN_NAME="exp5_lr1e5_$(date +%m%d_%H%M)" \
bash run.sh 5

# 或使用不同的输出目录
EXP_ROOT="outputs/exp_$(date +%m%d)" bash run.sh 5
```

---

## 📝 推荐的训练工作流

### **第一次运行（验证修复）**

```bash
cd /mnt/d/Code/PedestrianDetection
conda activate reid

# 1. 验证修复
bash verify_fixes.sh

# 2. 运行ExpT5（3个epoch，约2-4小时）
USE_MLFLOW=1 \
MLFLOW_EXPERIMENT="prcc_fixes" \
MLFLOW_RUN_NAME="exp5_fixed_$(date +%m%d)" \
START_STAGE=5 \
STOP_STAGE=5 \
bash run.sh

# 3. 查看结果
tail outputs/transfer/expT5_prcc_finetune/evaluation_metrics.csv
mlflow ui --backend-store-uri file:./outputs/mlruns
```

### **如果效果好，继续优化**

```bash
# 修改run.sh中的超参数，例如：
# - 调整 --cross-clothes-contrastive-weight
# - 修改 --learning-rate
# 然后再次运行并对比

USE_MLFLOW=1 \
MLFLOW_EXPERIMENT="prcc_tuning" \
MLFLOW_RUN_NAME="exp5_lr5e5" \
bash run.sh 5
```

---

## 🎓 总结

**最简单的开始方式**：

```bash
# 在WSL中
cd /mnt/d/Code/PedestrianDetection
conda activate reid

# 启用MLflow运行ExpT5
USE_MLFLOW=1 bash run.sh 5

# 另开一个终端查看进度
mlflow ui --backend-store-uri file:./outputs/mlruns
# 浏览器打开 http://localhost:5000
```

祝训练顺利！🚀
