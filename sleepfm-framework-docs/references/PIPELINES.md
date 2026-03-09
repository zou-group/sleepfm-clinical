# SleepFM 训练与评估流程

本指南提供了训练、微调和评估 SleepFM 模型的详细工作流程。

## 流程概览

```
Raw EDF Files
    ↓
Preprocessing (EDF → HDF5)
    ↓
Foundation Model Pretraining (Optional if using pretrained)
    ↓
Embedding Generation
    ↓
Fine-tuning (Sleep Staging OR Disease Prediction)
    ↓
Evaluation & Inference
```

## 1. 预处理流程

### 脚本：`sleepfm/preprocessing/preprocessing.py`

**用途**：将 EDF 文件转换为具有标准化结构的 HDF5 格式

**类**：`EDFToHDF5Converter`

**关键方法**：

#### `read_edf(edf_path)`
- Uses MNE library to load EDF
- 提取信号数据和元数据
- 识别通道名称和采样率

#### `resample_signals(signals, target_rate=128)`
- 将所有信号重采样到 128 Hz
- Uses scipy.signal.resample (FFT-based)
- 在不同的原始采样率下保持信号质量

#### `map_channels_to_modalities(channel_names, channel_groups_json)`
- 将 PSG 通道映射到 4 种模态：BAS、RESP、EKG、EMG
- Uses channel_groups.json for mappings
- 处理缺失/未识别的通道

#### `save_to_hdf5(data, hdf5_path)`
- 以 HDF5 格式保存重采样后的信号
- 结构：
  ```
  dataset.hdf5
  ├── BAS/ (C channels, T timepoints, 640 samples per chunk)
  ├── RESP/
  ├── EKG/
  └── EMG/
  ```
- 使用压缩以提高效率

**批量处理**：
```bash
# Process all EDF files in a directory
python sleepfm/preprocessing/preprocessing.py \
    --root_dir /path/to/edf/files \
    --target_dir /path/to/output/hdf5 \
    --resample_rate 128
```

**预期运行时间**：
- 单个记录：约 30 秒
- 完整数据集：取决于大小（MESA：约 10 分钟）

## 2. 基础模型预训练

### 脚本：`sleepfm/pipeline/pretrain.py`

**用途**：使用对比学习在多模态 PSG 上训练 SetTransformer

**配置文件**：`sleepfm/configs/config_set_transformer_contrastive.yaml`

**配置参数**：

```yaml
# Model Architecture
embed_dim: 128
num_heads: 8
num_layers: 6
pooling_head: 8
dropout: 0.3

# Training
batch_size: 128
epochs: 100
lr: 0.001
lr_step_period: 2
gamma: 0.1  # LR decay factor
weight_decay: 0.0
momentum: 0.9

# Data
modality_types: ['BAS', 'RESP', 'EKG', 'EMG']
BAS_CHANNELS: 10  # Max channels per modality
RESP_CHANNELS: 7
EKG_CHANNELS: 2
EMG_CHANNELS: 4
sampling_freq: 128
sampling_duration: 5  # Seconds
patch_size: 640  # 5 * 128

# Contrastive Learning
temperature: 0.07  # For contrastive loss

# Validation
mode: leave_one_out  # Cross-dataset validation
val_size: 100

# Logging
use_wandb: True
save_iter: 5000  # Save checkpoint every N iterations
eval_iter: 5000  # Evaluate every N iterations
log_interval: 100
```

**训练循环**：

```python
# Pseudo-code from pretrain.py
for epoch in range(epochs):
    for batch_idx, batch in enumerate(train_loader):
        # Load batch
        modality_data, masks, file_ids = batch

        # Forward pass for each modality
        embeddings = {}
        for modality in modality_types:
            data = modality_data[modality]
            mask = masks[modality]
            embeddings[modality] = model(data, mask)

        # Compute contrastive loss
        # - Positive pairs: same recording, different augmentations
        # - Negative pairs: different recordings
        loss = contrastive_loss(embeddings, temperature)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        if batch_idx % log_interval == 0:
            wandb.log({"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})

        # Checkpointing
        if (batch_idx % save_iter == 0):
            save_checkpoint(model, optimizer, epoch, batch_idx)

    # Validation
    if (epoch % eval_epoch == 0):
        validate(model, val_loader)
```

**验证策略**：
- **留一数据集验证（Leave-one-dataset-out）**：在 N-1 个数据集上训练，在保留的数据集上验证
- 测试在不同 PSG 导联和人群中的泛化能力
- 防止对特定队列的过拟合

**预期运行时间**（MESA 数据集）：
- 1 个 epoch：在 A40 GPU 上约 1 小时
- 完整训练：10-50 个 epoch，取决于收敛情况

**监控指标**：
- 对比损失（训练）
- 验证损失（如果有标签）
- 嵌入质量（可视化、下游任务）

## 3. 嵌入生成

### 脚本：`sleepfm/pipeline/generate_embeddings.py`

**用途**：从预训练的基础模型中提取嵌入用于微调

**输入**：
- 预处理后的 HDF5 文件
- 预训练的 SetTransformer 检查点

**输出**：
- 包含嵌入的 HDF5 文件（结构与输入相同）
- 两种类型：
  1. **5 秒嵌入**：序列级别 (B, S, 128)，用于睡眠分期
  2. **5 分钟聚合嵌入**：池化 (B, 128)，用于疾病预测

**代码示例**：

```python
from sleepfm.models.models import SetTransformer
import torch

# Load model
model = SetTransformer(in_channels=1, patch_size=640, embed_dim=128, ...)
checkpoint = torch.load("checkpoints/model_base/best.pt")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Generate embeddings
with torch.no_grad():
    for batch in dataloader:
        modality_data, masks, file_paths, chunk_starts = batch

        # Process each modality independently
        for modality_idx, modality in enumerate(['BAS', 'RESP', 'EKG', 'EMG']):
            data = modality_data[modality_idx]
            mask = masks[modality_idx]

            # Forward pass
            pooled_emb, seq_emb = model(data, mask)

            # Save embeddings
            save_embeddings(
                file_path=file_paths[i],
                modality=modality,
                pooled=pooled_emb.cpu().numpy(),
                sequence=seq_emb.cpu().numpy()
            )
```

**输出结构**：
```
output_embeddings/
├── subject_001.hdf5
│   ├── BAS/ (T, 128) - 5-second embeddings
│   ├── RESP/
│   ├── EKG/
│   └── EMG/
└── subject_5min_agg/
    ├── subject_001.hdf5
    │   ├── BAS/ (T/60, 128) - 5-minute aggregated
    │   ├── RESP/
    │   ├── EKG/
    │   └── EMG/
```

**预期运行时间**：
- MESA：在 A40 GPU 上约 5-10 分钟
- 与数据集大小呈线性关系

## 4. 睡眠分期微调

### 脚本：`sleepfm/pipeline/finetune_sleep_staging.py`

**用途**：使用预计算的嵌入在睡眠阶段标签上训练 LSTM 分类器

**配置文件**：`sleepfm/configs/config_finetune_sleep_events.yaml`

**配置**：

```yaml
# Data
context: -1  # -1 = full night, N = fixed window
max_channels: 10
channel_like: ['BAS', 'RESP', 'EKG', 'EMG']
modality_types: ['BAS', 'RESP', 'EKG', 'EMG']

# Model
model: SleepEventLSTMClassifier
model_params:
  embed_dim: 128
  num_heads: 8
  num_layers: 6
  num_classes: 5
  pooling_head: 4
  dropout: 0.1
  max_seq_length: 20000

# Training
batch_size: 8
epochs: 50
lr: 0.0001
weight_decay: 0.0

# Labels
split_path: configs/dataset_split.json
```

**数据集**：`SleepEventClassificationDataset`

**标签格式**（CSV）：
```csv
Start,Stop,StageName,StageNumber
0.0,30.0,Wake,0
30.0,60.0,N1,1
60.0,90.0,N2,2
90.0,120.0,N3,3
120.0,150.0,REM,4
```

**Collate 函数**：
- 将序列填充到批次中的最大长度
- 为填充创建掩码（0=数据，1=填充）
- 移除首次睡眠前的清醒期（常见的预处理）

**训练循环**：

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for batch in train_loader:
        embeddings, labels, mask = batch

        # Forward pass
        logits, _ = model(embeddings, mask)  # (B, T, 5)

        # Reshape for loss computation
        logits_flat = logits.reshape(-1, 5)  # (B*T, 5)
        labels_flat = labels.reshape(-1)  # (B*T,)

        # Filter out padding
        mask_flat = mask.reshape(-1) == 0  # Valid data only
        logits_flat = logits_flat[mask_flat]
        labels_flat = labels_flat[mask_flat]

        # Compute loss
        loss = criterion(logits_flat, labels_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**预期性能**（来自论文）：
- 平均 F1：0.70-0.78
- 每类 F1：
  - Wake（清醒）：0.75-0.85
  - N1：0.35-0.50（最难）
  - N2：0.70-0.80
  - N3：0.75-0.85
  - REM：0.75-0.85

**训练时间**（MESA）：
- 1 个 epoch：在 A40 GPU 上约 1 分钟
- 完整训练：20-50 个 epoch

## 5. 睡眠分期评估

### 脚本：`sleepfm/pipeline/evaluate_sleep_staging.py`

**评估指标**：
- **F1 分数**：每类和宏平均
- **准确率**：总体和每类
- **混淆矩阵**：可视化错误
- **Cohen's Kappa**：评估者间一致性

**代码示例**：

```python
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score

# Evaluate on test set
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        embeddings, labels, mask = batch
        logits, _ = model(embeddings, mask)
        preds = torch.argmax(logits, dim=-1)  # (B, T)

        # Filter padding
        valid_mask = mask == 0
        all_preds.append(preds[valid_mask].cpu().numpy())
        all_labels.append(labels[valid_mask].cpu().numpy())

# Compute metrics
f1_per_class = f1_score(all_labels, all_preds, average=None)
f1_macro = f1_score(all_labels, all_preds, average='macro')
kappa = cohen_kappa_score(all_labels, all_preds)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
```

**可视化**：
```python
import seaborn as sns
import matplotlib.pyplot as plt

class_labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
cm_norm = cm / cm.sum(axis=1, keepdims=True) * 100

plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (%)')
plt.savefig('confusion_matrix.png')
```

## 6. 疾病预测微调

### 脚本：`sleepfm/pipeline/finetune_diagnosis_coxph.py`

**用途**：训练 CoxPH 模型用于生存分析和疾病风险预测

**配置文件**：`sleepfm/configs/config_finetune_diagnosis_coxph.yaml`

**配置**：

```yaml
# Data
max_channels: 10
modality_types: ['BAS', 'RESP', 'EKG', 'EMG']
num_conditions: 1065

# Labels
labels_path: /path/to/labels/
demo_labels_path: /path/to/demographics.csv

# Model
model: DiagnosisFinetuneFullLSTMCOXPHWithDemo
model_params:
  embed_dim: 128
  num_heads: 8
  num_layers: 6
  num_classes: 1065
  pooling_head: 4
  dropout: 0.1
  max_seq_length: 20000

# Training
batch_size: 16
epochs: 100
lr: 0.0001
weight_decay: 0.0
```

**数据集**：`DiagnosisFinetuneFullCOXPHWithDemoDataset`

**标签文件**：
1. **is_event.csv**：每种疾病的二值指示器
   ```
   Study ID,Condition_0,Condition_1,...,Condition_1064
   SSC_00001,0,1,...,0
   SSC_00002,1,0,...,1
   ```

2. **time_to_event.csv**：到诊断或检查的时间
   ```
   Study ID,Condition_0,Condition_1,...,Condition_1064
   SSC_00001,3650,1825,...,NaN
   SSC_00002,730,NaN,...,3650
   ```

3. **demographics.csv**：年龄和性别
   ```
   Study ID,Age,Sex
   SSC_00001,65,1
   SSC_00002,72,0
   ```

**CoxPH 损失实现**：

```python
def coxph_loss(logits, is_event, time_to_event):
    """
    logits: (B, num_conditions) - hazard predictions
    is_event: (B, num_conditions) - event indicator
    time_to_event: (B, num_conditions) - time to event/censoring
    """
    total_loss = 0

    for condition_idx in range(num_conditions):
        condition_logits = logits[:, condition_idx]
        condition_events = is_event[:, condition_idx]
        condition_times = time_to_event[:, condition_idx]

        # Only process subjects with events
        event_mask = condition_events == 1

        if event_mask.sum() == 0:
            continue  # Skip if no events for this condition

        for i in range(len(condition_logits)):
            if not event_mask[i]:
                continue  # Censored - not in loss

            # Risk set: subjects with time_to_event >= time_to_event[i]
            risk_set = condition_times >= condition_times[i]

            # Partial likelihood
            hazard_i = condition_logits[i]
            hazard_risk_set = condition_logits[risk_set]

            # Negative log likelihood
            loss_i = -(hazard_i - torch.logsumexp(hazard_risk_set, dim=0))
            total_loss += loss_i

    return total_loss / num_conditions
```

**训练循环**：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for batch in train_loader:
        embeddings, event_time, is_event, demo_feats, mask = batch

        # Forward pass
        hazards = model(embeddings, mask, demo_feats)  # (B, 1065)

        # Compute CoxPH loss
        loss = coxph_loss(hazards, is_event, event_time)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**预期性能**（来自论文）：
- 130+ 种疾病的 C-index ≥0.75（Bonferroni 校正 p<0.01）
- 表现最佳的疾病：
  - 全因死亡率：C-index 0.84
  - 痴呆症：0.85
  - 心肌梗死：0.81
  - 心力衰竭：0.80
  - 慢性肾病：0.79
  - 中风：0.78
  - 房颤：0.78

**训练时间**（Stanford 数据集）：
- 1 个 epoch：在 A40 GPU 上约 5-10 分钟
- 完整训练：50-100 个 epoch

## 7. 疾病预测评估

**评估指标**：C-index（一致性指数）

```python
def concordance_index(risk, time_to_event, is_event):
    """
    risk: (N,) predicted risk scores
    time_to_event: (N,) time to event or censoring
    is_event: (N,) event indicator (1=event, 0=censored)
    """
    concordant = 0
    permissible = 0

    for i in range(len(risk)):
        for j in range(len(risk)):
            if i == j:
                continue

            # Permissible pairs: both events, or one event before other censoring
            if is_event[i] == 1 and is_event[j] == 1:
                permissible += 1
                if time_to_event[i] < time_to_event[j] and risk[i] > risk[j]:
                    concordant += 1
                elif time_to_event[i] > time_to_event[j] and risk[i] < risk[j]:
                    concordant += 1
                elif time_to_event[i] == time_to_event[j]:
                    # Ties - count as 0.5
                    pass
            elif is_event[i] == 1 and is_event[j] == 0:
                if time_to_event[i] <= time_to_event[j]:
                    permissible += 1
                    if risk[i] > risk[j]:
                        concordant += 1

    return concordant / permissible if permissible > 0 else 0
```

**评估**：
```python
# Evaluate on test set
all_risks = []
all_times = []
all_events = []

with torch.no_grad():
    for batch in test_loader:
        embeddings, event_time, is_event, demo_feats, mask = batch
        hazards = model(embeddings, mask, demo_feats)

        all_risks.append(hazards.cpu().numpy())
        all_times.append(event_time.cpu().numpy())
        all_events.append(is_event.cpu().numpy())

# Compute C-index per condition
c_indexes = []
for condition_idx in range(1065):
    risk = all_risks[:, condition_idx]
    time = all_times[:, condition_idx]
    event = all_events[:, condition_idx]

    c_index = concordance_index(risk, time, event)
    c_indexes.append(c_index)

# Report
print(f"Mean C-index: {np.mean(c_indexes):.3f}")
print(f"Conditions with C-index ≥0.75: {sum(np.array(c_indexes) >= 0.75)}")
```

## 8. WandB 集成

**设置**：

```python
import wandb

# Initialize
wandb.init(
    project="sleepfm",
    entity="your-username",
    config=config_dict,
    name="experiment-name"
)

# Log metrics
wandb.log({
    "train/loss": loss.item(),
    "train/lr": optimizer.param_groups[0]['lr'],
    "val/f1_macro": f1_macro,
    "val/kappa": kappa
})

# Log plots
wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})

# Finish
wandb.finish()
```

**仪表板**：
- 实时损失曲线
- 超参数比较
- 模型检查点
- 可复现性跟踪

## 最佳实践

### 数据集划分
- **训练集**：70-80% 的数据
- **验证集**：10-15% 用于超参数调优
- **测试集**：10-15% 保留用于最终评估
- **留一数据集验证**：用于跨数据集泛化能力评估

### 超参数调优
- 学习率：[1e-5, 1e-3]（对数尺度）
- 批次大小：[8, 32, 64, 128]（基于 GPU 内存）
- Dropout：[0.1, 0.3, 0.5]
- 权重衰减：[0.0, 1e-5, 1e-4]

### 早停法
```python
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(epochs):
    val_loss = validate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, 'best.pt')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break
```

### 检查点保存
```python
checkpoint = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'best_loss': best_val_loss,
    'config': config_dict
}
torch.save(checkpoint, 'checkpoint.pt')
```

### 可复现性
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
