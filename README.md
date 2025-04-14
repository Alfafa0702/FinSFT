# 金融领域大语言模型微调实验报告

## 目录
- [数据准备](#数据准备)
- [实验设计](#实验设计)
- [实验结果](#实验结果)
- [结论](#结论)

## 数据准备

### 数据集获取
```bash
# 1. 下载 FinEval 数据集
mkdir financial_evaluation_dataset
cd financial_evaluation_dataset
git init
git remote add origin https://github.com/alipay/financial_evaluation_dataset.git
git config core.sparseCheckout true
echo "data/Ant/金融知识/" >> .git/info/sparse-checkout
echo "data/SUFE/" >> .git/info/sparse-checkout
git pull origin main

# 2. 下载 Kaggle 数据集的方法
# 配置 Kaggle API：https://www.kaggle.com/docs/api#authentication
# 使用 kaggle CLI 下载 zip
kaggle datasets download yixinzhou2002/nlp-hw3-output
```
然后提交到 kaggle dataset，作为 input add 到 notebook 中。

## 实验设计

### 研究目标
本实验旨在微调一个熟悉各类金融知识的模型，使其能够准确回答金融考试选择题。通过探索不同模型、微调方法及参数设置，优化模型在金融领域的表现。

### 数据来源
数据是Ant和SUFE联合制作的FinEval，通过长期客观调研总结和严格的人工筛选，利用多项选择题、主客观简答题、推理规划和检索问答等8351道多种与实际应用场景高度一致的题型，包括了金融学术知识、金融行业知识、金融安全知识以及金融智能体。目前开源数据集是金融学术知识。FinEval金融学术知识是包含高质量多项选择题的集合，涵盖金融、经济、会计和证书等领域。它包括1321条有答案的训练集，涵盖了34个不同的学术科目。相比作业提供的link，我找到了原仓库，加入了Ant制作的中文题，变成1781题，并且SUFE的dev文件夹中含有cot的少量dev数据，每个subject有5条含有cot的回答。同时，通用知识题目我选择了CHARM，覆盖了中国文化知识。

### 数据处理
1. **数据格式转换**
   - 将原始数据转换为 ShareGPT 格式的 JSON 文件
   - 划分训练集和测试集

2. **数据增强**
   - 加入 COT（Chain of Thought）解释数据，从FinEval的dev数据集中获得
   - 整合 CHARM 通用领域知识，维持领域数据与通用数据的平衡
   - 优化 prompt 指令表述

### 模型选择
1. **基础模型**
   - Qwen2-7b-instruct
   - DeepSeek-R1-Distill-Qwen-7B
   - Qwen2-7b-instruct已经采用通用数据集微调过的，能够适应问答题的指令。- DeepSeek-R1-Distill-Qwen-7B利用了从DeepSeek生成的中文推理数据集对Qwen-7B进行微调，具备了推理能力，在中文语境和领域知识理解方面具有出色表现。这样的对比实验可以研究，在处理中文文本的金融考试题目方面，推理大模型与指令大模型的表现。

2. **评估维度**
   - 未微调模型基准性能
   - 微调后模型性能提升

### 参数配置
| 参数类型 | 配置选项 | 说明 |
|---------|---------|------|
| 学习率 | 1e-5, 2e-7 | 影响模型参数更新步长 |
| 训练轮数 | 1, 5 | 5轮预期效果更好 |
| LoRA rank | 8 | 固定参数 |

### 微调方法
- **主要方法**：QLoRA（Quantized Low-Rank Adaptation）
  - 基于8bit量化模型
  - 使用低秩矩阵调整参数
  - 降低计算量和内存需求

## 实验结果

### 已完成实验概览
| 实验ID | 模型 | 数据集 | Epoch | 学习率 | 训练时长 | 最后loss | 金融准确率(SFT前) |金融准确率(SFT后) | CHARM准确率(SFT后) |
|--------|------|--------|--------|--------|---------|-----------|------------|-------------|-------------|
| 1 | Qwen2-7b-instruct | fineval | 1 | 2e-7 | 1h | 5.805 | 70.03% | 81.23% | 67.78% |
| 2 | Qwen2-7b-instruct | fineval+cot+charm | 1 | 2e-7 | 1h12m | 14.141 | 70.03% | 80.39% | 65.00% |
| 3 | Qwen2-7b-instruct | fineval+cot+charm | 5 | 2e-7 | 6h12m | 13.457 | 70.03% | 81.23% | 65.00% |
| 4 | Qwen2-7b-instruct | fineval+cot+charm | 1 | 1e-5 | 1h26m | 8.408 | 70.03% | 85.46% | 68.33% |
| 5 | DeepSeek-R1-Distill-Qwen-7B | fineval | 1 | 2e-7 | 1h21m | 17.997 | 48.18% | 48.79% | 32.78% |

### 模型响应分析

#### Qwen2-7b-instruct 表现
- 金融数据集：直接输出选项字母
- CHARM数据集：偶有格式不规范问题

#### DeepSeek 表现
- 推理过程冗长
- 难以严格遵循指令要求
- 受token限制影响较大
- 把答案写在`</think>`后面

### 性能影响因素分析

1. **模型架构影响**
   - Qwen2-7b-instruct 展现出更好的指令遵循能力
   - DeepSeek 在推理能力与指令遵从间存在权衡

2. **训练参数影响**
   - Epoch增加：性能小幅提升，loss下降趋势明显
   - 学习率调整：较大学习率有助于降低loss

3. **数据增强效果**
   - COT数据加入效果不明显
   - 通用知识整合优势未充分体现

## 结论

### 主要发现
1. 模型选择方面：
   - Qwen2-7b-instruct 表现最优（最高达85.46%）
   - DeepSeek 在指令遵循方面存在挑战

2. 参数优化方面：
   - 更多训练轮次有助于提升性能
   - 较大学习率可能带来更好效果

### 未来展望
1. 优化方向：
   - 改进模型选择策略
   - 平衡推理与指令遵循能力
   - 深入探索数据增强方法

2. 潜在改进：
   - 优化prompt工程
   - 探索混合训练策略
   - 改进评估方法
