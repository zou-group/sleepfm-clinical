# SleepFM 模型架构深度解析

## SetTransformer 基础模型

### 架构概览

SetTransformer 是一个基于多模态注意力机制的架构，专为处理具有可变通道数和缺失模态的 PSG 信号而设计。

### 组件详解

#### 1. Tokenizer（基于 CNN 的信号编码器）

**位置**: [models.py:12-63](sleepfm/models/models.py:12)

```python
class Tokenizer(nn.Module):
    def __init__(self, input_size=640, output_size=128)
```

**架构**:
- **输入**: 640 个样本的原始信号片段（5 秒 @ 128 Hz）
- **层结构**: 6 个 1D 卷积层，采用渐进式下采样
  - Conv1: 1→4 通道, kernel=5, stride=2 (640→320 样本)
  - Conv2: 4→8 通道, kernel=5, stride=2 (320→160)
  - Conv3: 8→16 通道, kernel=5, stride=2 (160→80)
  - Conv4: 16→32 通道, kernel=5, stride=2 (80→40)
  - Conv5: 32→64 通道, kernel=5, stride=2 (40→20)
  - Conv6: 64→128 通道, kernel=5, stride=2 (20→10)
- **归一化**: 每个卷积层后使用 BatchNorm + LayerNorm
- **激活函数**: ELU (Exponential Linear Unit)
- **池化**: AdaptiveAvgPool1d → Flatten → Linear(128, 128)
- **输出**: 128 维嵌入向量

**设计原理**:
- 分层特征提取（局部 → 全局模式）
- 通过独立处理处理可变通道数
- Stride=2 在保持信息的同时降低时间分辨率

#### 2. AttentionPooling

**位置**: [models.py:65-95](sleepfm/models/models.py:65)

```python
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads=1, dropout=0.1)
```

**功能**:
- 使用 TransformerEncoderLayer 进行自注意力计算
- 对可变长度序列应用掩码
- 通过掩码均值产生池化表示

**掩码策略**:
- `mask=0`: 有效（真实）数据
- `mask=1`: 填充（缺失）数据
- 为注意力计算反转掩码（1=有效，0=填充）

**两阶段池化**:
1. **空间池化**: 在模态内跨通道聚合
   - 输入: (B*S, C, E)，其中 S=时间步，C=通道
   - 输出: (B, S, E) 跨通道池化

2. **时间池化**: 跨时间步聚合
   - 输入: (B, S, E)
   - 输出: (B, E) 最终嵌入

#### 3. PositionalEncoding

**位置**: [models.py:98-112](sleepfm/models/models.py:98)

```python
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model)
```

**实现**: 正弦位置编码
- 使用不同频率的 sin/cos 函数
- 偶数维度: sin(pos / 10000^(2i/d_model))
- 奇数维度: cos(pos / 10000^(2i/d_model))
- 在时间 transformer 之前加到嵌入上

**目的**: 为睡眠动态编码时间顺序信息

#### 4. SetTransformer（完整模型）

**位置**: [models.py:115-152](sleepfm/models/models.py:115)

```python
class SetTransformer(nn.Module):
    def __init__(self, in_channels=1, patch_size=640, embed_dim=128,
                 num_heads=8, num_layers=6, pooling_head=4, dropout=0.1)
```

**前向传播**:

1. **Patch 嵌入** (Tokenizer)
   - 输入: (B, C, T, raw_samples)
   - 输出: (B, C, S, E)，其中 S=T/patch_size 时间步

2. **空间池化**
   - 重排: (B, C, S, E) → (B*S, C, E)
   - 跨通道池化: (B*S, E) → (B, S, E)

3. **位置编码 + LayerNorm**
   - 添加时间位置信息
   - 归一化以稳定训练

4. **时间 Transformer**
   - 6 层 TransformerEncoder
   - 每层 8 个注意力头
   - 处理时间动态

5. **时间池化**
   - 跨时间聚合
   - 输出: (B, E) 池化嵌入

6. **返回值**
   - `x`: 池化嵌入（5 分钟聚合，用于疾病预测）
   - `embedding`: 序列嵌入（5 秒级别，用于睡眠分期）

### 关键设计模式

**缺失数据的掩码处理**:
```python
# Mask shape: (B, C, T) - 与输入相同的空间维度
# 0 = 真实数据, 1 = 填充
mask = mask.to(dtype=torch.bool)
x_pooled = attention_pooling(x, key_padding_mask=mask)
```

**多模态处理**:
- 每个模态（BAS、RESP、EKG、EMG）独立处理
- 每个模态单独的前向传播
- 嵌入在下游连接或聚合

**可变长度处理**:
- 填充到批次中的最大通道/时间
- 掩码防止对填充的注意力
- 掩码均值池化用于聚合

## 微调模型

### SleepEventLSTMClassifier

**位置**: [models.py:156-216](sleepfm/models/models.py:156)

**架构**:
1. 预计算嵌入的空间池化
2. 位置编码 + LayerNorm
3. Transformer 编码器（空间建模）
4. 双向 LSTM（时间建模）
   - 隐藏层大小: embed_dim // 2 (64)
   - 双向 → 连接 → embed_dim (128)
5. 全连接层: 128 → 5 类

**输入/输出**:
- 输入: 来自 SetTransformer 的 (B, C, T, E) 嵌入
- 输出: 每个时间步的 (B, T, 5) logits
- 类别: [Wake, N1, N2, N3, REM]

**训练细节**:
- 损失: 交叉熵
- 指标: 每类 F1 分数、准确率、混淆矩阵
- 上下文: 全夜或固定窗口

### DiagnosisFinetuneFullLSTMCOXPHWithDemo

**位置**: [models.py:281-351](sleepfm/models/models.py:281)

**架构**:
1. 嵌入的空间池化
2. 用于时间序列的双向 LSTM
3. **打包序列**用于高效的可变长度处理
   - LSTM 前使用 `pack_padded_sequence`
   - LSTM 后使用 `pad_packed_sequence`
4. 有效长度上的均值池化（不包括填充）
5. **人口统计嵌入**
   - 线性层: 2 个特征（年龄、性别）→ embed_dim // 4 (32)
   - ReLU 激活
   - Dropout 正则化
6. 连接: sleep embed (128) + demo embed (32) = 160
7. 疾病头: Linear(160, 1065) → 风险比

**生存损失 (CoxPH)**:
```python
# Cox 比例风险损失
# 部分似然的负对数似然
loss = -log(exp(hazard_i) / sum(exp(hazard_j)) for j in risk_set)
```

**输入/输出**:
- 输入: 嵌入 (B, C, T, 128)，人口统计 (B, 2)，mask
- 输出: (B, 1065) 风险分数
- 标签: is_event (B, 1065), time_to_event (B, 1065)

**训练细节**:
- 每种情况的风险集构建
- 通过 is_event 标志处理删失
- C-index 评估指标

## 损失函数

### 对比损失（预训练）

**目标**: 学习能够区分以下内容的表示:
1. 不同模态（BAS vs RESP vs EKG vs EMG）
2. 不同时间窗口（增强视图）
3. 相同 vs 不同受试者

**实现** (在 pipeline/pretrain.py 中):
```python
# 带温度缩放的对比损失
# 正样本对: 相同记录，不同增强
# 负样本对: 不同记录
loss = -log(exp(sim(z_i, z_j)/tau) / sum(exp(sim(z_i, z_k)/tau)))
```

**温度**: 控制分布的锐度（默认: 0.07）

### 交叉熵损失（睡眠分期）

标准多类分类:
```python
loss = CrossEntropyLoss()(logits, targets)
# logits: (B, T, 5)
# targets: (B, T) - 类别索引 0-4
```

**类别不平衡**: 对罕见阶段（N1）使用加权损失或焦点损失

### CoxPH 损失（疾病预测）

生存分析损失:
```python
# 对于每个受试者 i 和疾病 d:
# 风险集 R: time_to_event >= time_to_event[i] 的受试者
# hazard_ratio = exp(logits[i])  # 线性预测器
# loss = -log(hazard_ratio[i] / sum(hazard_ratio[j] for j in R))
# 如果 is_event[i] == 0（删失），不在损失中
```

## 模型参数

### SetTransformer（基础模型）

```yaml
embed_dim: 128
num_heads: 8
num_layers: 6
pooling_head: 4
dropout: 0.1
patch_size: 640  # 5 秒 @ 128 Hz
max_seq_length: 128  # 用于位置编码
```

**总参数量**: ~4.44M

### SleepEventLSTMClassifier

```yaml
embed_dim: 128
num_heads: 8
num_layers: 6
num_classes: 5
pooling_head: 4
dropout: 0.1
max_seq_length: 20000
```

**总参数量**: ~1.19M

### DiagnosisFinetuneFullLSTMCOXPHWithDemo

```yaml
embed_dim: 128
num_heads: 8
num_layers: 6
num_classes: 1065
pooling_head: 4
dropout: 0.1
max_seq_length: 20000
```

**总参数量**: ~0.91M

## 训练考虑

### 多 GPU 训练

```python
model = nn.DataParallel(model)
# 自动跨 GPU 分割批次
# 跨 GPU 平均梯度
```

### 梯度累积

用于更大的有效批次大小:
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 学习率调度

```python
# StepLR 衰减
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=lr_step_period,  # 默认: 2
    gamma=0.1  # 衰减因子
)
```

### 检查点保存

```python
checkpoint = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'loss': best_loss,
}
torch.save(checkpoint, 'best.pt')
```

## 推理优化

### 无梯度上下文

```python
with torch.no_grad():
    outputs = model(inputs)
```

### 批量推理

```python
# 并行处理多个记录
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
for batch in dataloader:
    outputs = model(batch)
```

### 内存管理

```python
# 清空缓存
torch.cuda.empty_cache()

# 混合精度（如果支持）
from torch.cuda.amp import autocast
with autocast():
    outputs = model(inputs)
```
