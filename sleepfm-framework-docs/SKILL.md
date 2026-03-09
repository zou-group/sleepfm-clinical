---
name: sleepfm-framework
description: SleepFM 多模态睡眠基础模型框架，用于分析 PSG（多导睡眠图）数据。在处理睡眠医学 AI、生理信号处理、睡眠分期、从睡眠记录预测疾病，或将基础模型应用于临床应用时使用。涵盖模型架构、数据预处理、训练流程和临床推理工作流。
---

# SleepFM 框架

SleepFM 是一个多模态睡眠基础模型，用于分析多导睡眠图（PSG）数据以预测睡眠阶段和疾病风险。该模型发表在 Nature Medicine 上，在来自 65,000+ 名参与者的 585,000+ 小时的 PSG 数据上训练而成。

## 快速开始

### 安装与设置

```bash
# 环境要求
- Python 3.10
- CUDA 12.4
- GPU: NVIDIA A40/A100（或 RTX 2080 Ti，需减小批处理大小）
- RAM: 32+ GB

# 安装依赖
conda env create -f env.yml
conda activate sleepfm_env
pip install -r requirements.txt
```

### 核心工作流

**1. 预处理 EDF 到 HDF5**
```python
from sleepfm.preprocessing.preprocessing import EDFToHDF5Converter

converter = EDFToHDF5Converter(
    root_dir="/path/to/edf/files",
    target_dir="/path/to/output",
    resample_rate=128
)
converter.convert(edf_path, hdf5_path)
```

**2. 生成嵌入向量**
```bash
python sleepfm/pipeline/generate_embeddings.py \
    --config sleepfm/configs/config_set_transformer_contrastive.yaml
```

**3. 睡眠分期推理**
```bash
python sleepfm/pipeline/finetune_sleep_staging.py \
    --config sleepfm/configs/config_finetune_sleep_events.yaml
python sleepfm/pipeline/evaluate_sleep_staging.py
```

**4. 疾病预测**
```bash
python sleepfm/pipeline/finetune_diagnosis_coxph.py \
    --config sleepfm/configs/config_finetune_diagnosis_coxph.yaml
```

## 模型架构

### 基础模型（SetTransformer）

核心组件位于 [models.py](sleepfm/models/models.py)：

- **Tokenizer（分词器）**（基于 CNN）：将原始信号转换为 128 维嵌入向量
  - 输入：每个通道 640 个样本（128 Hz 下 5 秒）
  - 6 层 CNN，包含 BatchNorm、ELU、LayerNorm
  - 输出：Patch 嵌入向量

- **AttentionPooling（注意力池化）**：用于空间/时间聚合的多头注意力机制
  - 空间池化：模态内各通道之间
  - 时间池化：各时间步之间
  - 使用带掩码的 TransformerEncoderLayer

- **PositionalEncoding（位置编码）**：用于时间信息的正弦编码

- **TransformerEncoder**：6 层，8 个注意力头，embed_dim=128

### 微调模型

**SleepEventLSTMClassifier（睡眠事件 LSTM 分类器）**：
- 架构：SetTransformer → BiLSTM → FC（全连接层）
- 类别：5 个睡眠阶段（清醒、N1、N2、N3、REM）
- 输入：预计算的嵌入向量（BAS、RESP、EKG、EMG 模态）

**DiagnosisFinetuneFullLSTMCOXPHWithDemo（带人口统计学特征的诊断微调模型）**：
- 架构：SetTransformer → BiLSTM → 人口统计学嵌入 → CoxPH
- 类别：1065 种医学疾病（phecode 映射）
- 输入：嵌入向量 + 人口统计学特征（年龄、性别）

### 模态类型

四种 PSG 信号模态（[channel_groups.json](sleepfm/configs/channel_groups.json)）：

1. **BAS（基础）**：EEG、EOG 通道 - 脑/眼活动
2. **RESP**：呼吸、血氧饱和度、气流、打鼾
3. **EKG**：ECG/EKG 信号 - 心脏活动
4. **EMG**：肌电图 - 肌肉活动（下巴、腿部、手臂）

## 数据流程

### 输入格式

- **原始 PSG**：EDF 文件（欧洲数据格式）
- **预处理后的**：HDF5 文件，包含重采样信号（128 Hz）
- **标签**：用于睡眠阶段或疾病结果的 CSV 文件

### 预处理细节

参见 [preprocessing/preprocessing.py](sleepfm/preprocessing/preprocessing.py:1)：

1. 使用 MNE 库加载 EDF
2. 使用 channel_groups.json 将通道映射到模态组
3. 将所有信号重采样到 128 Hz
4. 以标准化结构存储为 HDF5 格式
5. 处理可变的通道数量和缺失的模态

### 数据集类

**SetTransformerDataset**：用于基础模型预训练
- 从 HDF5 文件加载 5 秒的数据块
- 返回可变长度序列的掩码张量
- 支持多模态批处理加载

**SleepEventClassificationDataset**：用于睡眠分期
- 基于上下文的窗口（默认：整夜）
- 从 CSV 加载睡眠阶段标签
- Collate 函数处理填充/掩码

**DiagnosisFinetuneFullCOXPHWithDemoDataset**：用于疾病预测
- 加载人口统计学特征（年龄、性别）
- 加载生存数据：is_event、time_to_event
- 返回 CoxPH 训练样本

## 配置系统

所有配置均为 YAML 格式（[configs/](sleepfm/configs/)）：

**config_set_transformer_contrastive.yaml**：
- 模型架构：embed_dim、num_heads、num_layers
- 训练：batch_size、lr、epochs
- 数据：modality_types、patch_size、sampling_freq

**config_finetune_sleep_events.yaml**：
- 上下文长度、max_channels
- LSTM 分类器的模型参数
- 标签路径和数据集划分

**config_finetune_diagnosis_coxph.yaml**：
- 疾病类别数：1065
- 人口统计学特征维度
- 生存损失参数

关键参数：
- `patch_size: 640`（5 秒 × 128 Hz）
- `embed_dim: 128`
- `sampling_freq: 128`
- `modality_types: ['BAS', 'RESP', 'EKG', 'EMG']`

## 训练与评估

### 预训练

```python
# sleepfm/pipeline/pretrain.py
# 用于多模态 PSG 的对比学习目标
# 损失：跨模态和时间增强的对比损失
```

关键点：
- 留一数据集验证
- 对比损失的温度参数
- WandB 日志集成
- 每 N 次迭代保存检查点

### 睡眠分期

```python
# sleepfm/pipeline/finetune_sleep_staging.py
# 在有标签的睡眠阶段数据上微调
# 损失：5 个类别的交叉熵
# 指标：每个类别的 F1 分数、混淆矩阵
```

预期性能（来自论文）：
- 平均 F1：跨数据集 0.70-0.78
- 在清醒、REM、N3 阶段表现最佳
- N1 阶段表现较弱（睡眠分期中常见）

### 疾病预测

```python
# sleepfm/pipeline/finetune_diagnosis_coxph.py
# 使用 Cox 比例风险损失微调
# 指标：生存分析的 C-index
```

主要结果：
- 130+ 种疾病的 C-index ≥0.75
- 表现最佳：痴呆症（0.85）、全因死亡率（0.84）、心肌梗死（0.81）
- 结合睡眠嵌入向量与人口统计学特征

## 推理与部署

### 嵌入向量生成

使用预训练的基础模型（[checkpoints/model_base/](sleepfm/checkpoints/model_base)）：

```python
from sleepfm.models.models import SetTransformer
import torch

model = SetTransformer(in_channels=1, patch_size=640, embed_dim=128, ...)
checkpoint = torch.load("sleepfm/checkpoints/model_base/best.pt")
model.load_state_dict(checkpoint["state_dict"])

# 前向传播返回：
# - pooled_embedding: 5 分钟聚合（用于疾病预测）
# - sequence_embeddings: 5 秒级别（用于睡眠分期）
pooled, sequence = model(modality_data, mask)
```

### 睡眠分期推理

```python
from sleepfm.models.models import SleepEventLSTMClassifier

model = SleepEventLSTMClassifier(embed_dim=128, num_classes=5, ...)
checkpoint = torch.load("sleepfm/checkpoints/model_sleep_staging/best.pth")
model.load_state_dict(checkpoint)

# 输入：(B, C, T, E) 嵌入向量
# 输出：(B, T, 5) 每个时间步的 logit
logits, mask = model(embeddings, mask)
predictions = torch.softmax(logits, dim=-1)
```

### 疾病预测推理

```python
from sleepfm.models.models import DiagnosisFinetuneFullLSTMCOXPHWithDemo

model = DiagnosisFinetuneFullLSTMCOXPHWithDemo(embed_dim=128, num_classes=1065, ...)
checkpoint = torch.load("sleepfm/checkpoints/model_diagnosis/best.pth")
model.load_state_dict(checkpoint)

# 输入：嵌入向量 + 人口统计学特征 [年龄、性别]
# 输出：(B, 1065) 风险比
hazards = model(embeddings, mask, demo_features)
```

使用 [label_mapping.csv](sleepfm/configs/label_mapping.csv) 将输出映射到疾病：
```python
import pandas as pd
labels_df = pd.read_csv("sleepfm/configs/label_mapping.csv")
labels_df["predicted_hazard"] = hazards[0].cpu().numpy()
```

## 临床应用

### 结果解释

**睡眠分期**：每个时期（30 秒）的预测
- 清醒与睡眠的区分
- REM 与 NREM 的分类
- N1、N2、N3 深度评分

**疾病预测**：受试者水平的风险评分
- 1065 种疾病的风险比
- 需要生存数据进行校准
- 与人口统计学风险因素结合

### 局限性

- 特定数据集的通道映射可能需要自定义
- 模型在特定人群上训练（斯坦福队列）
- 需要高质量的 PSG 记录（对伪迹敏感）
- 疾病预测需要在目标人群中验证

### 最佳实践

1. **通道映射**：查看 [channel_groups.json](sleepfm/configs/channel_groups.json) 并添加数据集特定的通道
2. **数据质量**：过滤信号质量差的记录
3. **微调**：即使样本量很小，也要适应目标数据集
4. **验证**：使用留一数据集法进行泛化测试
5. **临床审查**：模型预测应辅助而非替代专家评分

## 参考文件

详细信息请参阅：

- **[MODEL_ARCHITECTURE.md](references/MODEL_ARCHITECTURE.md)**：深入探讨 SetTransformer、注意力机制、损失函数
- **[PIPELINES.md](references/PIPELINES.md)**：逐步训练工作流程和超参数
- **[CLINICAL_GUIDE.md](references/CLINICAL_GUIDE.md)**：睡眠医学背景、解释预测、部署考虑

## 故障排除

**CUDA 内存不足**：减少配置中的 batch_size，或使用梯度累积

**通道数量可变**：模型通过掩码处理；确保 channel_groups.json 完整

**睡眠分期性能差**：在目标数据集上微调，即使标签有限

**疾病预测校准**：训练数据中每个疾病需要足够的事件

**通道缺失**：检查 EDF 中的通道命名与 channel_groups.json 的对比；根据需要添加映射
