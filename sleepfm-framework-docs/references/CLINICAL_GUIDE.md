# SleepFM 临床应用指南

本指南为解读 SleepFM 预测结果和在真实世界环境中部署模型提供临床背景。

## 睡眠医学背景

### 多导睡眠图（PSG）基础

**什么是 PSG？**
- 多参数睡眠研究
- 睡眠评估的金标准
- 记录睡眠期间的生理信号

**典型的 PSG 信号**：
1. **脑电图（EEG）**：大脑活动
   - 测量睡眠分期、觉醒、癫痫发作
   - 标准导联：F3、F4、C3、C4、O1、O2，参考乳突

2. **眼电图（EOG）**：眼动
   - 区分快速眼动睡眠（快速眼动）
   - LOC（左）、ROC（右）电极

3. **肌电图（EMG）**：肌肉活动
   - 颈部肌电图：快速眼动期肌张力缺失检测
   - 腿部肌电图：周期性肢体运动（PLMS）

4. **心电图（ECG）**：心脏节律
   - 心率变异性
   - 心律失常检测

5. **呼吸信号**：
   - 鼻/口气流（热敏电阻或压力传感器）
   - 呼吸努力（胸部/腹部带）
   - 血氧饱和度（脉搏血氧测定）

### 睡眠分期（AASM 标准）

**清醒期（Stage 0）**：
- 脑电图中的 Alpha 节律（8-13 Hz）
- 眼电图中的眨眼
- 高颈部肌电图张力

**N1 期（Stage 1）**：
- 从清醒到睡眠的过渡
- 脑电图中的 Theta 节律（4-7 Hz）
- 慢眼动
- 肌电图张力降低

**N2 期（Stage 2）**：
- 浅睡眠
- 睡眠纺锤波（11-16 Hz 爆发）
- K-复合波（先锐利负波后正波）

**N3 期（Stage 3）**：
- 深/慢波睡眠
- Delta 波（0.5-2 Hz，>75 μV）
- 难以唤醒
- 对身体恢复很重要

**快速眼动期（Stage R）**：
- 快速眼动
- 低振幅混合频率脑电图（类似于清醒/N1）
- 肌张力缺失（低颈部肌电图）
- 做梦发生
- 对记忆巩固很重要

**正常睡眠结构**：
- 每晚 4-6 个周期
- 每个周期：70-100 分钟
- 进展：N1 → N2 → N3 → N2 → REM
- 前半段更多 N3，后半段更多 REM
- 典型成人：N1（5%）、N2（50%）、N3（20%）、REM（25%）、清醒（5%）

### 常见睡眠障碍

**阻塞性睡眠呼吸暂停（OSA）**：
- 睡眠期间上气道反复塌陷
- 症状：打鼾、喘息、白天嗜睡
- 诊断：AHI ≥5 次/小时（轻度：5-15，中度：15-30，重度：>30）
- 患病率：男性约 10-17%，女性约 3-9%

**失眠症**：
- 难以入睡或维持睡眠
- 诊断：临床访谈 + 睡眠日记
- PSG 上不直接评分

**不宁腿综合征（RLS）**：
- 移动腿部的冲动，休息/晚上时加重
- 临床诊断，PSG 显示 PLMS

**发作性睡病**：
- 过度白天嗜睡
- 猝倒（突然肌肉无力）
- PSG 上的睡眠起始 REM 期（SOREMPs）

**快速眼动睡眠行为障碍（RBD）**：
- 正常 REM 肌张力缺失丧失
- 将梦境表现出来
- 神经退行性疾病的危险因素

## 解读睡眠分期预测

### 模型输出格式

**输入**：PSG 记录（6-12 小时）
**输出**：每个时段（30 秒）的预测

```
Epoch | Time  | Predicted | Probability
------|-------|-----------|------------
0     | 00:00 | Wake      | [0.92, 0.02, 0.03, 0.01, 0.02]
1     | 00:30 | Wake      | [0.88, 0.05, 0.04, 0.01, 0.02]
2     | 01:00 | N1        | [0.15, 0.62, 0.18, 0.02, 0.03]
...
```

**置信度评分**：
- 高置信度：max(probability) > 0.80
- 中置信度：0.60-0.80
- 低置信度：<0.60（考虑专家审查）

### 常见错误模式

**N1 混淆**：
- 模型经常将 N1 与清醒或 N2 混淆
- 即使在专家之间，N1 的评分者间信度也较低
- 考虑使用上下文平滑（N1 通常被 N2 包围）

**睡眠起始后清醒（WASO）**：
- 模型可能在觉醒期间过度预测清醒
- 临床规则：<15 秒的觉醒通常不评分清醒

**REM 转换**：
- 模型可能错过 REM 起始转换
- 寻找同时出现的 REM 指标：低 EMG + 眼动

**N3 碎片化**：
- 在老年人中，N3 减少且碎片化
- 模型可能在老年人群中过度预测 N2

### 睡眠指标计算

**睡眠潜伏期（SOL）**：
```
从关灯到前 5 个连续 N1/N2/N3 时段的时间
```

**总睡眠时间（TST）**：
```
所有 N1 + N2 + N3 + REM 时段的总和（不包括清醒）
```

**睡眠效率（SE）**：
```
SE = (TST / 总记录时间) × 100%
正常：>85%
```

**睡眠起始后清醒（WASO）**：
```
睡眠起始后的总清醒时间
正常：<30 分钟
```

**觉醒指数**：
```
每小时睡眠的觉醒次数
正常：<10-15 次/小时
```

**分期百分比**：
```
%N1 = (N1 时段 / TST) × 100
%N2 = (N2 时段 / TST) × 100
%N3 = (N3 时段 / TST) × 100
%REM = (REM 时段 / TST) × 100
```

### 临床决策支持

**何时信任模型**：
- 高置信度预测（>0.80）
- 清晰的睡眠结构（可见周期）
- 正常 PSG 质量（最小伪影）

**何时标记专家审查**：
- 低置信度预测
- 异常睡眠结构（例如，无 N3、REM 过多）
- 信号质量差
- 高觉醒指数
- 疑似睡眠障碍（呼吸暂停、PLMS、异态睡眠）

**质量控制检查**：
```python
def check_sleep_quality(predictions, probabilities):
    warnings = []

    # Check 1: Confidence
    low_confidence = (probabilities.max(axis=1) < 0.60).sum()
    if low_confidence > len(predictions) * 0.20:  # >20% low confidence
        warnings.append("Many low-confidence predictions - recommend expert review")

    # Check 2: Sleep architecture
    stage_percentages = calculate_stage_percentages(predictions)
    if stage_percentages['N3'] < 5:
        warnings.append("Very low N3 percentage - verify quality or age-related")
    if stage_percentages['REM'] < 10:
        warnings.append("Low REM percentage - check for REM deprivation or technical issue")

    # Check 3: Sleep continuity
    waso = calculate_waso(predictions)
    if waso > 60:  # >60 minutes
        warnings.append("High WASO - possible sleep maintenance insomnia")

    # Check 4: Sleep onset
    sol = calculate_sol(predictions)
    if sol > 30:  # >30 minutes
        warnings.append("Prolonged sleep onset latency - possible insomnia")

    return warnings
```

## 解读疾病预测

### Cox 比例风险模型

**输出**：1065 种医疗状况的风险比

**解读**：
- 更高的风险比 = 更高的风险
- **相对风险**：与人群平均水平的比较
- **非绝对风险**：不预测疾病是否/何时发生

**示例**：
```
Subject: 65-year-old male

Condition              | Hazard Ratio | Interpretation
-----------------------|--------------|------------------------
Dementia               | 2.5          | 2.5x higher risk than avg
Myocardial Infarction  | 1.8          | 1.8x higher risk than avg
Atrial Fibrillation    | 0.9          | Slightly below average risk
```

### 校准和验证

**人群特异性校准**：
- 模型在斯坦福睡眠队列上训练
- 在其他人群中表现可能不同
- 临床使用时需要本地数据重新校准

**时间-事件考虑**：
- 基于随访时间的预测（通常 5-15 年）
- 最近数据的随访时间较短
- 删失：失访或事件未发生

**混杂因素**：
- 年龄：大多数疾病的强预测因子
- 性别：男性与女性的不同风险
- 合并症：并非所有疾病都包含在模型中
- 药物：可能影响睡眠和疾病风险

### 临床用例

**筛查**：
- 识别高危患者进行针对性预防
- 示例：高痴呆风险 → 认知筛查、生活方式改变

**风险分层**：
- 将睡眠衍生风险与传统风险因素结合
- 示例：高心血管风险 + 睡眠异常 → 早期干预

**监测**：
- 通过系列 PSG 跟踪随时间的变化
- 示例：OSA 治疗 → 降低心血管风险

**研究**：
- 识别睡眠-疾病关联
- 为机制研究生成假设

### 局限性和注意事项

**非诊断性**：
- 模型预测风险，而非诊断
- 需要临床相关性
- 考虑其他风险因素和症状

**假阳性**：
- 一些高风险预测不会发展为疾病
- 避免不必要的焦虑或检查

**假阴性**：
- 一些低风险预测仍可能发展为疾病
- 继续标准筛查

**人群偏倚**：
- 训练数据：来自特定队列的约 65K 参与者
- 可能无法推广到所有种族、年龄组、地理区域

**时间有效性**：
- 医疗实践在发展
- 新治疗可能会改变疾病风险
- 模型可能需要定期重新训练

### 伦理考虑

**知情同意**：
- 向患者解释模型局限性
- 阐明：风险预测，而非诊断
- 讨论不确定性和置信区间

**健康差异**：
- 确保睡眠测试的公平获取
- 解决训练数据中的潜在偏倚
- 在不同人群中验证

**隐私**：
- 保护敏感健康数据
- 符合 HIPAA/GDPR
- PSG 和预测的安全存储

**过度治疗**：
- 避免仅基于风险预测的干预
- 考虑患者偏好和价值观
- 与临床医生共同决策

## 部署考虑

### 与临床工作流程集成

**PSG 采集**：
- 标准 PSG 设备（Compumedics、Nihon Kohden 等）
- 导出为 EDF 格式
- 自动预处理流程

**质量控制**：
- 模型输入前的信号质量检查
- 伪影检测和拒绝
- 通道验证（正确导联）

**报告生成**：
```
SLEEP STUDY REPORT
==================

Patient: [Name]
Date of Study: [Date]
Total Recording Time: [Hours]

SLEEP METRICS
- Total Sleep Time: [Hours] (Normal: >7 hours)
- Sleep Efficiency: [%] (Normal: >85%)
- Sleep Onset Latency: [Minutes] (Normal: <30)
- Wake After Sleep Onset: [Minutes] (Normal: <30)

STAGE DISTRIBUTION
- N1: [%] (Normal: 3-8%)
- N2: [%] (Normal: 45-55%)
- N3: [%] (Normal: 15-25%, age-dependent)
- REM: [%] (Normal: 20-25%)

AI MODEL CONFIDENCE
- High confidence predictions: [%]
- Low confidence predictions (flag for review): [%]

DISEASE RISK PREDICTIONS
Top 5 elevated risks:
1. [Condition]: Hazard Ratio [X.XX]
2. [Condition]: Hazard Ratio [X.XX]
...

DISCLAIMER: AI predictions are for informational purposes only and should be
reviewed by a qualified sleep specialist. Not a substitute for clinical judgment.
```

### 模型更新和再训练

**何时再训练**：
- 新队列数据可用（域偏移）
- 验证集性能下降
- 新疾病或诊断标准
- PSG 技术进步

**持续学习**：
- 监控预测与实际结果
- 收集睡眠临床医生的反馈
- 定期更新模型（例如，每年）

### 监管考虑

**FDA 分类**（如果临床使用）：
- 医疗器械软件（SaMD）
- 可能需要 FDA 清除或批准
- 取决于预期用途（决策支持 vs 诊断）

**验证**：
- 前瞻性验证研究
- 多中心试验
- 与专家评分比较

**责任**：
- 关于模型限制的明确免责声明
- 临床医生保留最终责任
- 记录模型版本和训练数据

## 未来方向

**多模态集成**：
- 将 PSG 与体动记录仪、问卷结合
- 包括可穿戴设备数据（Oura 戒指、Apple Watch）

**个性化医疗**：
- 影响睡眠的基因变异
- 睡眠药物的药物基因组学
- 个体化治疗建议

**实时应用**：
- 基于睡眠分期的自适应 PAP 治疗
- 智能家居集成（照明、温度）
- 紧急警报（严重呼吸暂停、心脏事件）

**远程医疗**：
- 人工智能分析的家庭睡眠测试
- 睡眠障碍的远程监测
- 失眠的数字疗法
