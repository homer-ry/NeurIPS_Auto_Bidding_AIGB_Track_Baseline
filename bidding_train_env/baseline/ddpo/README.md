# DDPO-CBD: Denoising Diffusion Policy Optimization for Causal Bidding

本项目实现了DDPO (Denoising Diffusion Policy Optimization) 与CBD (Causal Bidding Diffusion) 的结合，用于自动竞价策略优化。

## 方法概述

### DDPO核心思想
DDPO将扩散模型的去噪过程视为马尔可夫决策过程(MDP)：
- **状态**: 当前去噪步骤的潜在表示 $x_t$
- **动作**: 从$x_t$转移到$x_{t-1}$的噪声样本
- **策略**: 扩散模型预测的噪声分布
- **奖励**: 生成样本的质量评分

使用PPO风格的策略梯度进行优化，通过重要性采样和裁剪目标函数来稳定训练。

### CBD基线
CBD使用扩散模型+逆动力学模型(IDM)进行竞价：
1. 扩散模型生成未来状态轨迹
2. IDM根据状态序列预测当前动作
3. 支持选择性推理增强因果性

### DDPO-CBD结合
1. **预训练**: 先训练CBD模型学习策略分布
2. **奖励模型**: 训练轨迹评分模型预测未来回报
3. **DDPO微调**: 使用DDPO策略梯度微调CBD模型
4. **评估对比**: 对比CBD baseline和DDPO-CBD效果

## 目录结构

```
bidding_train_env/
├── baseline/
│   └── ddpo/
│       ├── __init__.py
│       ├── reward_model.py      # 轨迹奖励模型
│       └── ddpo_trainer.py      # DDPO训练器
├── strategy/
│   ├── cbd_bidding_strategy.py          # CBD策略
│   └── ddpo_cbd_bidding_strategy.py     # DDPO-CBD策略
run/
├── run_cbd.py                  # CBD训练脚本
├── run_cbd_ppo.py              # CBD-PPO训练脚本
├── compare_cbd_ddpo.py         # 对比评估脚本
└── run_evaluate.py             # 评估脚本
```

## 使用流程

### 1. 数据准备
确保数据已准备好：
- `data/trajectory/trajectory_data.csv` - 轨迹数据
- `data/traffic/period-7.csv` - 流量数据

### 2. 训练CBD基线模型
```bash
python -m run.run_cbd --epochs 100 --n_timesteps 10 --save_path saved_model/CBDtest
```

### 3. 训练DDPO-CBD
```bash
python -m run.run_cbd_ppo \
    --cbd_path saved_model/CBDtest/diffuser.pt \
    --save_path saved_model/DDPO-CBD \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4
```

关键参数说明：
- `--cbd_path`: 预训练CBD模型路径
- `--epochs`: 微调轮次
- `--lr`: CBD主模型学习率
- `--reward_model_lr`: 奖励模型学习率
- `--max_train_batches`: 调试时限制每轮 batch 数
- `--train_data_path`: 可切到小样本轨迹做 smoke test

### 4. 对比评估
```bash
python run/compare_cbd_ddpo.py
```

## 奖励模型

### 轨迹奖励模型
`TrajectoryRewardModel`预测轨迹的未来回报：
```python
reward_model = TrajectoryRewardModel(
    state_dim=16,      # 状态维度
    hidden_dim=128,    # 隐藏层维度
    num_layers=2       # Transformer层数
)
reward = reward_model(states, masks)  # [batch, 1]
```

### 评分计算
基于竞赛评分函数：
```
Score = Total_Reward - γ * max(0, CPA_real - CPA_constraint)
```
其中$\gamma$是CPA违规惩罚系数。

## DDPO训练

### 核心算法
```python
# PPO裁剪目标
ratio = exp(log_prob_new - log_prob_old)
surrogate1 = ratio * advantage
surrogate2 = clip(ratio, 1-ε, 1+ε) * advantage
loss = -min(surrogate1, surrogate2).mean()
```

### 训练流程
1. 从CBD策略采样轨迹
2. 奖励模型评估轨迹得分
3. 计算优势函数（归一化）
4. 多轮PPO内循环更新
5. 定期评估并保存最佳模型

## 监控指标

训练过程监控：
- `policy_loss`: DDPO策略损失
- `reward_loss`: 奖励模型损失
- `approx_kl`: 近似KL散度（监控策略偏移）
- `clipfrac`: 裁剪比例
- `mean_reward`: 平均预测奖励

收敛性判断：
- 策略损失标准差 < 0.01
- 预测奖励趋于稳定
- 评估分数不再显著提升

## 预期效果

DDPO-CBD相比CBD预期改进：
- 更高的总奖励（提升5-15%）
- 更好的CPA满足率
- 更稳定的状态轨迹生成

## 注意事项

1. **学习率**: DDPO微调使用较小学习率（1e-5），避免破坏预训练知识
2. **采样数**: 增加采样轨迹数可提高策略评估准确性
3. **裁剪范围**: 适中裁剪（0.1-0.3）平衡探索与稳定
4. **奖励模型**: 奖励模型准确性直接影响DDPO效果

## 参考文献

- DDPO论文: Black et al., "Training Diffusion Models with Reinforcement Learning" (2023)
- CBD方法: 基于CAD (Causal Auto-bidding via Diffusion) 思想
