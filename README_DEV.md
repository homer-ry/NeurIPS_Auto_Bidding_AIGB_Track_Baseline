# README_DEV

开发侧使用说明，覆盖当前仓库里可运行的离线训练、评估、指标和结果整理流程。

```
cd /Users/rongyu/Documents/develop/bidding/tianchi_bidding/NeurIPS_Auto_Bidding_AIGB_Track_Baseline/
```

## 数据准备

按 [README.md](/Users/rongyu/Documents/develop/bidding/tianchi_bidding/NeurIPS_Auto_Bidding_AIGB_Track_Baseline/README.md) 放置数据：

- `data/traffic/period-7.csv` 到 `data/traffic/period-13.csv`
- `data/trajectory/trajectory_data.csv`
- `data/trajectory/trajectory_data_extended_1.csv`
- `data/trajectory/trajectory_data_extended_2.csv`

当前仓库里：

- `data/traffic/period-7.csv` 已存在，可用于离线评估
- `data/trajectory/trajectory_data.csv` 已存在，文件较大，约 `3.9G`

## 训练脚本

### DT

```bash
python3 -m run.run_decision_transformer
```

产物：

- `saved_model/DTtest/dt.pt`
- `saved_model/DTtest/normalize_dict.pkl`

### DD

```bash
python3 -m run.run_decision_diffuser
```

产物：

- `saved_model/DDtest/diffuser.pt`

### DiT

```bash
python3 -m run.run_dit \
  --save_path saved_model/DiTtest \
  --epochs 100 \
  --batch_size 1000 \
  --n_timesteps 10 \
  --model_choice DiT1d \
  --attn_block causal
```

产物：

- `saved_model/DiTtest/diffuser.pt`
- `saved_model/DiTtest/diffuser_best.pt`

### CBD

```bash
python3 -m run.run_cbd \
  --save_path saved_model/CBDtest \
  --epochs 100 \
  --batch_size 1000 \
  --n_timesteps 10 \
  --model_choice Unet
```

产物：

- `saved_model/CBDtest/diffuser.pt`
- `saved_model/CBDtest/diffuser_best.pt`

### CBD-PPO

`run/run_cbd_ppo.py` 是当前仓库里可运行的 `CBD + reward model + DDPO-style offline fine-tuning` 脚本。

正式训练示例：

```bash
python3 -m run.run_cbd_ppo \
  --cbd_path saved_model/CBDtest/diffuser.pt \
  --save_path saved_model/DDPO-CBD \
  --epochs 10 \
  --batch_size 32 \
  --lr 1e-4 \
  --reward_model_lr 1e-4 \
  --n_timesteps 10
```

小样本 smoke test：

```bash
head -n 5001 data/trajectory/trajectory_data.csv > /tmp/trajectory_smoke.csv

MPLCONFIGDIR=/tmp/mpl python3 -m run.run_cbd_ppo \
  --cbd_path saved_model/CBDtest/diffuser.pt \
  --save_path /tmp/ddpo_smoke \
  --epochs 1 \
  --batch_size 2 \
  --pretrain_rm_epochs 1 \
  --max_train_batches 1 \
  --num_workers 0 \
  --device cpu \
  --train_data_path /tmp/trajectory_smoke.csv
```

产物：

- `saved_model/DDPO-CBD/diffuser.pt`
- `saved_model/DDPO-CBD/diffuser_best.pt`
- `saved_model/DDPO-CBD/reward_model.pt`
- `saved_model/DDPO-CBD/reward_model_best.pt`
- `saved_model/DDPO-CBD/training_history.json`

## 评估流程

### 方式一：仓库自带入口

评估入口：

```bash
python3 -m run.run_evaluate
```

注意：

- 当前 [run/run_evaluate.py](/Users/rongyu/Documents/develop/bidding/tianchi_bidding/NeurIPS_Auto_Bidding_AIGB_Track_Baseline/run/run_evaluate.py) 还是调试版
- 它默认只取 `keys[0]`
- 并且只跑前 `3` 个 timestep

所以它适合调试策略接线，不适合作为正式离线对比结果。

### 方式二：完整离线评估

正式对比时，建议使用：

- `bidding_train_env.dataloader.test_dataloader.TestDataLoader`
- `bidding_train_env.environment.offline_env.OfflineEnv`
- 目标策略类，例如：
  - `CbdBiddingStrategy`
  - `DDPOCbdBiddingStrategy`

完整评估逻辑应当：

1. 读取 `period-7.csv`
2. 按 `(deliveryPeriodIndex, advertiserNumber)` 遍历 advertiser
3. 每个 advertiser 跑满全部 `48` 个 timestep
4. 用 `OfflineEnv.simulate_ad_bidding()` 做离线拍卖模拟
5. 统计 reward / cost / CPA / score

## 指标定义

核心指标在 [run/run_evaluate.py](/Users/rongyu/Documents/develop/bidding/tianchi_bidding/NeurIPS_Auto_Bidding_AIGB_Track_Baseline/run/run_evaluate.py) 里已经实现：

- `Total Reward`
  当前 advertiser 在一个投放周期内获得的总 conversion

- `Total Cost`
  当前 advertiser 在一个投放周期内的总花费

- `CPA-real`
  `Total Cost / Total Reward`

- `CPA-constraint`
  策略实例里的 `cpa`

- `Score`
  竞赛口径：

```text
if CPA-real <= CPA-constraint:
    Score = Reward
else:
    Score = Reward * (CPA-constraint / CPA-real)^2
```

说明：

- `Score` 是当前最重要的对比指标
- 只看 `Reward` 不够，因为超 CPA 会被惩罚

## 当前离线结果

以下结果基于完整 `48` 个 timestep 的离线评估，不是 `run_evaluate.py` 那个 3-step 调试口径。

### 单 advertiser

评估对象：

- 数据：`data/traffic/period-7.csv`
- advertiser：第一个 key
- 模型：
  - `saved_model/CBDtest/diffuser.pt`
  - `saved_model/DDPO-CBD-test/diffuser.pt`

结果：

- `CBD`
  - reward: `7.0`
  - cost: `99.9961`
  - score: `0.1372`

- `DDPO-CBD-test`
  - reward: `5.0`
  - cost: `99.9699`
  - score: `0.0500`

结论：

- 在单 advertiser 上，当前 `DDPO-CBD-test` 没有优于 `CBD`

### 前 10 个 advertiser 聚合

评估对象：

- 数据：`data/traffic/period-7.csv`
- advertiser：前 `10` 个 key
- 模型：
  - `saved_model/CBDtest/diffuser.pt`
  - `saved_model/DDPO-CBD-test/diffuser.pt`

结果：

- `CBD`
  - total_reward: `46.0`
  - total_cost: `999.6460`
  - score_mean: `0.0931`
  - global_cpa: `21.7314`

- `DDPO-CBD-test`
  - total_reward: `58.0`
  - total_cost: `999.6389`
  - score_mean: `0.0989`
  - global_cpa: `17.2352`

结论：

- 在前 `10` 个 advertiser 聚合上，`DDPO-CBD-test` 略优于 `CBD`
- 当前结论是“局部有效，但提升并不稳定”

## 推荐开发流程

### 1. 先做 smoke test

先确认脚本、依赖和权重加载正常：

```bash
head -n 5001 data/trajectory/trajectory_data.csv > /tmp/trajectory_smoke.csv

MPLCONFIGDIR=/tmp/mpl python3 -m run.run_cbd_ppo \
  --cbd_path saved_model/CBDtest/diffuser.pt \
  --save_path /tmp/ddpo_smoke \
  --epochs 1 \
  --batch_size 2 \
  --pretrain_rm_epochs 1 \
  --max_train_batches 1 \
  --num_workers 0 \
  --device cpu \
  --train_data_path /tmp/trajectory_smoke.csv
```

### 2. 再跑正式离线训练

```bash
python3 -m run.run_cbd_ppo \
  --cbd_path saved_model/CBDtest/diffuser.pt \
  --save_path saved_model/DDPO-CBD \
  --epochs 10 \
  --batch_size 32 \
  --pretrain_rm_epochs 1 \
  --num_workers 0
```

### 3. 最后做完整离线评估

不要直接依赖调试版 `run.run_evaluate`。

建议至少产出两套结果：

- 单 advertiser 结果
- 多 advertiser 聚合结果，例如前 `10` 个 advertiser 或全量 advertiser

## 当前代码状态

已经完成的修正：

- `run/run_ddpo_cbd.py` 已重命名为 [run/run_cbd_ppo.py](/Users/rongyu/Documents/develop/bidding/tianchi_bidding/NeurIPS_Auto_Bidding_AIGB_Track_Baseline/run/run_cbd_ppo.py)
- `README_dev.md` 已重命名为 [README_DEV.md](/Users/rongyu/Documents/develop/bidding/tianchi_bidding/NeurIPS_Auto_Bidding_AIGB_Track_Baseline/README_DEV.md)
- `DFUSER.py` 已改成按需加载 `DiT1d`，当前环境不装 `timm` 也能跑 `CBD/CBD-PPO`
- `aigb_dataset` 已支持 `--train_data_path`，便于小样本 smoke test
- `DDPOCbdBiddingStrategy` 已兼容：
  - `saved_model/DDPO-CBD/diffuser*.pt`
  - `saved_model/DDPO-CBD-test/diffuser*.pt`

## 备注

- 全量 `trajectory_data.csv` 很大，第一次加载会比较慢
- 如果只是验证代码链路，优先用 `/tmp/trajectory_smoke.csv`
- 如果后续要做稳定结论，建议把完整离线评估脚本单独固化到 `run/` 目录，而不是继续复用调试版 `run_evaluate.py`
