# ExpT4 和 ExpT5 训练策略

## 🤔 要不要重新训练ExpT4？

### **关键问题：ExpT4的质量如何？**

检查你现有的ExpT4性能：

```bash
# 查看ExpT4的训练日志
tail -50 outputs/transfer/expT4_joint_v1/evaluation_metrics.csv

# 重点关注PRCC-Dev的mAP趋势
grep "prcc_dev.*mAP" outputs/transfer/expT4_joint_v1/evaluation_metrics.csv
```

**判断标准**：

| ExpT4的PRCC-Dev mAP | 建议 | 原因 |
|-------------------|------|------|
| 稳定在 0.35-0.55 | ✅ **直接用现有ExpT4，只训练ExpT5** | ExpT4质量良好 |
| 从高到低严重下降（如0.53→0.36） | ⚠️ **需要重新训练ExpT4** | ExpT4已经退化 |
| 一直很低（<0.30） | ⚠️ **需要重新训练ExpT4** | ExpT4没训好 |

---

## 📋 三种训练场景

### **场景1: 只训练ExpT5（推荐，如果ExpT4质量OK）** ⭐

```bash
cd /mnt/d/Code/PedestrianDetection
conda activate reid

# 使用现有的ExpT4检查点，只训练ExpT5
USE_MLFLOW=1 \
MLFLOW_EXPERIMENT="prcc_optimization" \
MLFLOW_RUN_NAME="exp5_fixed_only" \
START_STAGE=5 \
STOP_STAGE=5 \
bash run.sh
```

**优点**：
- ✅ 快速（约2-4小时）
- ✅ 直接验证ExpT5的修复效果
- ✅ 节省计算资源

**适用于**：
- 你的ExpT4检查点质量良好
- 只想验证ExpT5的优化方案
- 快速迭代ExpT5的超参数

---

### **场景2: 一起训练ExpT4和ExpT5（完整流程）** 🔄

```bash
# 从ExpT3开始，重新训练ExpT4和ExpT5
USE_MLFLOW=1 \
MLFLOW_EXPERIMENT="prcc_full_retrain" \
MLFLOW_RUN_NAME="exp4_exp5_$(date +%m%d)" \
START_STAGE=4 \
STOP_STAGE=5 \
bash run.sh
```

**流程**：
1. **ExpT4训练** (~40 epochs, 约8-12小时)
   - 从ExpT3的best.pth初始化
   - 在Market+PRCC上联合训练
   - 每个epoch在PRCC-Dev验证集评估
   - 保存到 `outputs/transfer/expT4_joint_v1/best.pth`

2. **ExpT5训练** (~3 epochs, 约2-4小时)
   - **自动加载刚训练好的ExpT4**的best.pth
   - 在PRCC完整训练集微调
   - 在PRCC官方测试集评估

**优点**：
- ✅ ExpT4使用最新的修复（如果你改了ExpT4配置）
- ✅ 完整的训练流水线
- ✅ ExpT5从新鲜的ExpT4开始

**缺点**：
- ⏰ 时间长（约10-16小时）
- 💰 计算成本高

**适用于**：
- ExpT4质量很差，需要重新训练
- 修改了ExpT4的训练配置（如优化方案中建议的）
- 想要完整的实验记录

---

### **场景3: 用优化后的配置重新训练ExpT4** 🔧

如果你想应用FIXES_SUMMARY.md中的ExpT4优化建议：

#### **3.1 修改ExpT4配置**

编辑 `run.sh` 的ExpT4部分（大约在260-355行）：

```bash
# 找到train_expt4_joint_v1函数或ExpT4的配置
# 添加以下优化：

train_model \
  --mode joint \
  --epochs 40 \
  # ... 其他参数保持不变 ...
  
  # 添加以下优化参数：
  --cross-clothes-contrastive-weight 0.2 \       # 启用！
  --cross-clothes-contrastive-margin 0.5 \
  --part-triplet-weight 0.0 \                     # 禁用冲突损失
  --cloth-invariant-weight 0.0 \                  # 禁用冲突损失
  
  # 其他参数...
```

#### **3.2 运行优化后的ExpT4+ExpT5**

```bash
USE_MLFLOW=1 \
MLFLOW_EXPERIMENT="prcc_optimized" \
MLFLOW_RUN_NAME="exp4_exp5_optimized" \
START_STAGE=4 \
STOP_STAGE=5 \
bash run.sh
```

---

## 🎯 我的建议

根据你的情况，我推荐：

### **第一步：检查现有ExpT4的质量**

```bash
# 查看ExpT4的PRCC-Dev mAP趋势
cd /mnt/d/Code/PedestrianDetection
cat outputs/transfer/expT4_joint_v1/evaluation_metrics.csv | \
  grep "prcc_dev" | \
  awk -F, '{print $1, $3}' | \
  tail -10
```

### **第二步：根据结果选择策略**

#### **如果ExpT4看起来OK（mAP稳定在0.35+）**：

```bash
# 策略：只训练ExpT5
USE_MLFLOW=1 \
MLFLOW_RUN_NAME="exp5_quick_test" \
bash run.sh 5
```

#### **如果ExpT4有明显退化（mAP下降严重）**：

```bash
# 策略：重新训练ExpT4+ExpT5
USE_MLFLOW=1 \
MLFLOW_RUN_NAME="exp4_exp5_full" \
START_STAGE=4 \
STOP_STAGE=5 \
bash run.sh
```

---

## 📊 使用MLflow对比不同策略

如果你想对比两种策略的效果：

```bash
# Run 1: 使用现有ExpT4，只训练ExpT5
USE_MLFLOW=1 \
MLFLOW_EXPERIMENT="prcc_comparison" \
MLFLOW_RUN_NAME="exp5_old_exp4" \
bash run.sh 5

# Run 2: 重新训练ExpT4+ExpT5
USE_MLFLOW=1 \
MLFLOW_EXPERIMENT="prcc_comparison" \
MLFLOW_RUN_NAME="exp4_exp5_fresh" \
START_STAGE=4 \
STOP_STAGE=5 \
bash run.sh

# 然后在MLflow UI中对比
mlflow ui --backend-store-uri file:./outputs/mlruns
```

---

## ⚙️ 高级配置

### **自定义ExpT4检查点路径**

如果你想让ExpT5使用特定的ExpT4检查点：

```bash
# 方式1: 环境变量
EXP4_FOR_EXP5="/path/to/your/custom/expT4/best.pth" bash run.sh 5

# 方式2: 修改run.sh中的路径
# 编辑run.sh，找到：
# EXP4_FOR_EXP5="${EXP4_FOR_EXP5:-$DEFAULT_EXP4_FOR_EXP5}"
```

### **跳过ExpT4训练，直接用现有检查点训练ExpT5**

```bash
# 设置标志跳过ExpT4训练
TRAIN_EXPT4_JOINT_V1=0 \
EVAL_EXPT4_JOINT_V1=0 \
START_STAGE=4 \
STOP_STAGE=5 \
bash run.sh

# 这样会跳过ExpT4，但仍然会使用ExpT4的检查点初始化ExpT5
```

---

## 📈 训练时间估算

基于单GPU (如RTX 3090)：

| 训练内容 | Epochs | 预计时间 | 备注 |
|---------|--------|---------|------|
| **只ExpT5** | 3 | 2-4小时 | 快速验证 |
| **ExpT4** | 40 | 8-12小时 | 联合训练较慢 |
| **ExpT4+ExpT5** | 40+3 | 10-16小时 | 完整流程 |

如果使用多GPU（如2-4个），可以加快2-3倍。

---

## 🚀 推荐的实际操作步骤

### **Step 1: 快速验证修复（只ExpT5）**

```bash
cd /mnt/d/Code/PedestrianDetection
conda activate reid

# 先快速测试修复是否有效
USE_MLFLOW=1 \
MLFLOW_EXPERIMENT="prcc_fixes" \
MLFLOW_RUN_NAME="exp5_quick_test" \
bash run.sh 5

# 监控结果
tail -f outputs/transfer/expT5_prcc_finetune/train.log
```

**预期结果**：
- ✅ 训练不崩溃
- ✅ mAP不再下降
- ✅ mAP开始上升（即使缓慢）

### **Step 2: 如果效果好，决定是否重新训练ExpT4**

如果Step 1的ExpT5 mAP提升到0.32-0.35，说明修复有效！

**然后考虑**：
- 如果已经满意，可以继续调ExpT5的超参数
- 如果想追求更高，重新训练优化后的ExpT4+ExpT5

### **Step 3: 完整训练（如果需要）**

```bash
# 应用所有优化，重新训练ExpT4和ExpT5
USE_MLFLOW=1 \
MLFLOW_EXPERIMENT="prcc_final" \
MLFLOW_RUN_NAME="exp4_exp5_optimized" \
START_STAGE=4 \
STOP_STAGE=5 \
bash run.sh
```

---

## 💡 总结

**最省时间的方案**：
```bash
# 只训练ExpT5（2-4小时）
USE_MLFLOW=1 bash run.sh 5
```

**最完整的方案**：
```bash
# ExpT4+ExpT5一起训练（10-16小时）
USE_MLFLOW=1 START_STAGE=4 STOP_STAGE=5 bash run.sh
```

**我的建议**：先快速验证ExpT5，效果好再考虑是否重新训练ExpT4。

你想先快速验证还是直接完整训练？🤔
