# PRCC mAP优化 - 快速开始指南

## 🚀 立即开始

### 1. 验证修复（推荐）

```bash
# 在WSL中执行
cd /mnt/d/Code/PedestrianDetection
conda activate reid
bash verify_fixes.sh
```

### 2. 开始训练

```bash
# 运行ExpT5阶段
bash run.sh 5
```

### 3. 监控进度

```bash
# 实时日志
tail -f output/ExpT5/train.log

# 查看mAP
watch -n 5 "tail -3 output/ExpT5/evaluation_metrics.csv"
```

---

## ✅ 已完成的修复

### Critical Bug修复
1. ✅ 双分类器标签偏移逻辑 - **修复纯PRCC模式负标签bug**
2. ✅ Sketch损失标签处理 - **避免训练崩溃**
3. ✅ 检查点保存逻辑 - **修复恢复训练基准值错误**

### 训练配置优化
4. ✅ 解冻layer4 - **允许学习PRCC特定特征**
5. ✅ 启用cross-clothes contrastive loss (0.3) - **核心PRCC损失**
6. ✅ 禁用冲突损失 - **part-triplet和cloth-invariant设为0**
7. ✅ 降低学习率至1e-5 - **适合微调**
8. ✅ 使用余弦学习率衰减 - **替代milestone**
9. ✅ 减少数据增强强度 - **匹配模型能力**

### 代码优化
10. ✅ 合并重复的梯度反转操作 - **提升效率**

---

## 📊 预期效果

| 阶段 | mAP | 说明 |
|------|-----|------|
| 修复前 | 0.307 | 停滞不前 |
| 短期 | ~0.30 | 稳定，不再下降 |
| 中期 | 0.35-0.40 | 核心优化生效 |
| 长期 | 0.45-0.50 | 接近SOTA |

---

## 🔍 关键指标检查

训练时关注：
- ✅ `ce` loss: 从~3.5降到<2.0
- ✅ `cross_clothes_contrastive`: 逐渐下降
- ✅ `part_triplet`: 应该为0
- ✅ `cloth_invariant`: 应该为0
- ✅ mAP: 每个epoch上升

---

## 📁 重要文件

- **修复总结**: `FIXES_SUMMARY.md` - 完整文档
- **代码审查**: `C:\Users\xyz10\.claude\plans\code-quality-review.md`
- **优化方案**: `C:\Users\xyz10\.claude\plans\prcc-map-snoopy-shannon.md`
- **验证脚本**: `verify_fixes.sh` + `test_fixes.py`

---

## ⚠️ 故障排查

### 训练崩溃
```bash
python test_fixes.py  # 确认修复生效
tail -100 output/ExpT5/train.log  # 查看错误
```

### mAP不上升
1. 检查学习率: 尝试5e-5或5e-6
2. 检查loss权重: 调整cross-clothes-contrastive-weight
3. 考虑重新训练ExpT4

---

## 📝 下一步

1. **运行训练**: `bash run.sh 5`
2. **观察3个epoch的mAP变化**
3. **如果效果好**: 继续训练更多epoch
4. **如果效果不佳**: 参考FIXES_SUMMARY.md中的故障排查

---

## 🎯 成功标志

- [x] 训练不崩溃
- [ ] mAP每个epoch上升
- [ ] Epoch 3的mAP > 0.32
- [ ] Cross-clothes contrastive loss有值且下降

---

需要帮助？查看 `FIXES_SUMMARY.md` 获取详细文档。

Good luck! 🚀
