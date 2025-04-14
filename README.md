# 下载数据
```bash
mkdir financial_evaluation_dataset
cd financial_evaluation_dataset
git init
git remote add origin https://github.com/alipay/financial_evaluation_dataset.git
git config core.sparseCheckout true
echo "data/Ant/金融知识/" >> .git/info/sparse-checkout
echo "data/SUFE/" >> .git/info/sparse-checkout
git pull origin main
```
然后提交到kaggle dataset，作为input add到notebook中。

## 下载kaggle dataset
配置kaggle API：https://www.kaggle.com/docs/api#authentication  
kaggle datasets download yixinzhou2002/nlp-hw3-output 【用kaggle CLI下载zip】


# 实验设计
研究主题是微调一个熟悉各类金融知识的模型，使其能够回答金融考试选择题。
数据处理：
    转换为示例代码中的json格式，并划分训练集和测试集。
    加入explaination的COT数据(dev训练集)
    加入通用领域知识CHARM的选择题数据
    prompt指令上的数据增强
模型选择：
    未微调的模型在测试集上的表现
    微调的模型在测试集上的表现
    Qwen2-7b-instruct
    DeepSeek-R1-Distill-Qwen-7B (支持Q6_K和Q8_0量化模型，在中文语境和领域知识理解方面表现出色，适合处理中文文本)
参数调优：
    学习率 1e-5, 2e-7
    训练轮数 5，1 应该是5更好
    #batch size 16, 32 应该越xxxx，好像没有这个选择
    #LoRA的rank 8 越低越好
微调方法+Q：
    #全参数微调
    LoRA
    #AdamW
    #P-tuning
实验结果
    在测试集上的答题准确率+案例，准确率越高越好
    Training loss收敛曲线
    非金融通用问题的对话能力+案例
    分析不同实验组件对模型性能的影响
    点评模型回答是否遵照指令，即只输出选项答案

# 目前完成的实验
(1)Qwen/Qwen2-7B-Instruct QLoRA epoch=1 lr=2e-7 【pre eval 70.03% + SFT】
(5) [version 18 NLP-hw3] Qwen/Qwen2-7b-Instruct QloRA epoch=1 lr=2e-7 【eval 81.23% 67.78%】[10m]
(2)deepseek-ai/DeepSeek-R1-Distill-Qwen-7B QLoRA epoch=1 lr=2e-7 【pre eval 48.18%】[3h45m] [187/357出现了最终<\think>选答案]
(4) [version 12] deepseek-ai/DeepSeek-R1-Distill-Qwen-7B QLoRA epoch=1 lr=2e-7 【SFT training】[1h25min]【用Transformer一张卡训练会内存不足，缺了1G】
(3) [version 11] Qwen/Qwen2-7B-Instruct 用增加的数据集(常识+CoT) 【full SFT】【llamafactory Version11】 [1h17min]【用Transformer一张卡训练数据量增大后内存不够】 
(8) [NLP-hw3]Qwen/Qwen2-7B-Instruct 用增加的数据集(常识+CoT) 【eval 80.39% 65.00%】
(6) [version 8 Alfafa] Qwen2-7b-instruct QLoRA epoch=1 lr=1e-5 【SFT】[1h31m]
(7) [version 13 llama] Qwen2-7b-instruct QLoRA epoch=5 lr=2e-7 【SFT】[6h18m]

# 正在跑的实验

deepseek-ai/DeepSeek-R1-Distill-Qwen-7B QLoRA epoch=1 lr=2e-7 【Eval】

# 未来要做的实验
DeepSeek-R1-Distill-Qwen-7B QLoRA epoch=1 lr=2e-7 发现无法避免冗长的推理过程，更改了prompt和max_length=512
Qwen2-7b-instruct QLoRA epoch=5 lr=2e-7 【Eval】
Qwen2-7b-instruct QLoRA epoch=1 lr=1e-5 【Eval】
用trainer_state.json中的log_history绘制loss曲线
非金融通用问题的对话能力+案例


# 引言
随着金融行业的发展和金融知识的广泛应用，能够准确回答金融考试选择题的智能模型具有重要意义。本实验旨在通过对特定模型进行微调，使其在金融考试选择题回答任务上表现优异，同时探索不同模型、微调方法及参数设置对模型性能的影响。

# Experiment design
（一）数据来源与数据处理
数据是Ant和SUFE联合制作的FinEval，通过长期客观调研总结和严格的人工筛选，利用多项选择题、主客观简答题、推理规划和检索问答等8351道多种与实际应用场景高度一致的题型，包括了金融学术知识、金融行业知识、金融安全知识以及金融智能体。目前开源数据集是金融学术知识。FinEval金融学术知识是包含高质量多项选择题的集合，涵盖金融、经济、会计和证书等领域。它包括1321条有答案的训练集，涵盖了34个不同的学术科目。相比作业提供的link，我找到了原仓库，加入了Ant制作的中文题，变成1781题，并且SUFE的dev文件夹中含有cot的少量dev数据，每个subject有5条含有cot的回答。同时，通用知识题目我选择了CHARM，覆盖了。。。

格式转换与数据集划分：将原始数据转换为特定的sharegpt formating的JSON文件，以便模型处理。并将数据集划分为训练集和测试集，确保训练和评估过程的独立性和有效性。
COT 数据集成：在 dev 训练集中加入带有解释（explaination）的 COT 数据，期望通过这些解释性信息帮助模型更好地理解问题解决逻辑，提升答题能力。
CHARM 数据引入：纳入通用领域知识 CHARM 的选择题数据，维持领域数据与通用数据的平衡，以确保模型在保持通用性的同时，具备领域特定能力。
数据增强：在 prompt 指令上进行数据增强，通过多样化的指令表述方式，使模型对各种提问形式更加鲁棒，提高模型在实际应用中的适应性。

Instruct模型可能是通用数据微调过的能够适应问答模式


# Code implementation
Present your code implementation, demonstrating how you translated your ideas into code.

# Result analysis
Collect and present the results obtained from your experiments, along with an analysis of the insights gained.

# Conclusion
Conclude your report by summarizing the key findings and outcomes of your research.
