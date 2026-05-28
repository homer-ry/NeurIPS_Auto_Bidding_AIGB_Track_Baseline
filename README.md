# 项目概览
本项目是一个自动出价训练与评估框架，用于实现、训练和离线评估不同自动出价策略。
框架包含三个主要模块：数据处理、策略训练和离线评估。
项目内置了多种生成式模型基线策略，例如 Decision Transformer、Decision Diffuser、DiT、CBD 和 CBD-PPO。
参赛者或研究者可以基于公开训练数据训练自动出价策略，并通过本项目提供的离线环境进行基础评估。
由于离线训练阶段无法直接接入真实拍卖系统，本框架的离线评估可用于验证策略实现是否满足竞赛接口和基本效果要求。



## 依赖环境
```
conda create -n nips-bidding-env python=3.9.12 pip=23.0.1
conda activate nips-bidding-env
pip install -r requirements.txt 
```

# 可复现离线基线实验流程

本节是本项目推荐的端到端离线实验流程，用于产出所有基线方法的离线对比结果表格。流程覆盖公开数据集构建、模型训练、统一离线评估和结果表格导出。

支持的基线方法：

| 方法 | 类型 | 训练入口 | 默认 checkpoint |
| --- | --- | --- | --- |
| Heuristic | 规则基线 | 无需训练 | 内置 `PlayerBiddingStrategy` |
| DT | Decision Transformer | `run/run_decision_transformer.py` | `saved_model/DTtest/dt.pt` |
| DD | Decision Diffuser | `run/run_decision_diffuser.py` | `saved_model/DDtest/diffuser.pt` |
| DiT | Diffusion Transformer | `run/run_dit.py` | `saved_model/DiTtest/diffuser.pt` |
| CBD | Causal Bidding Diffusion, U-Net | `run/run_cbd.py` | `saved_model/CBDtest/diffuser.pt` |
| CBD-PPO | 基于 DDPO/PPO 微调的 CBD | `run/run_cbd_ppo.py` | `saved_model/DDPO-CBD/diffuser.pt` |

最终推荐使用 `run/run_open_source_benchmark.py` 进行统一评估。该脚本会在公开 traffic period 上评估所选策略，并在 `results/open_source_benchmark/` 下生成 Markdown 和 JSON 两种格式的对比表。

## 1. 环境准备

```bash
conda create -n nips-bidding-env python=3.9.12 pip=23.0.1
conda activate nips-bidding-env
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

启动长时间训练前，建议先确认 CUDA 是否可用：

```bash
python - <<'PY'
import torch
print("cuda_available =", torch.cuda.is_available())
print("device_count =", torch.cuda.device_count())
PY
```

## 2. 数据集构建

将公开 traffic 数据和 trajectory 数据下载到 `data/downloads/`：

```bash
mkdir -p data/downloads data/traffic data/trajectory

wget -P data/downloads https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_period_7-8.zip
wget -P data/downloads https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_period_9-10.zip
wget -P data/downloads https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_period_11-12.zip
wget -P data/downloads https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_period_13.zip
wget -P data/downloads https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_trajectory_data.zip
wget -P data/downloads https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_trajectory_data_extended_1.zip
wget -P data/downloads https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_trajectory_data_extended_2.zip
```

解压并放置到项目约定目录：

```bash
unzip -o data/downloads/autoBidding_aigb_track_data_period_7-8.zip -d data/traffic
unzip -o data/downloads/autoBidding_aigb_track_data_period_9-10.zip -d data/traffic
unzip -o data/downloads/autoBidding_aigb_track_data_period_11-12.zip -d data/traffic
unzip -o data/downloads/autoBidding_aigb_track_data_period_13.zip -d data/traffic
unzip -o data/downloads/autoBidding_aigb_track_data_trajectory_data.zip -d data/trajectory
unzip -o data/downloads/autoBidding_aigb_track_data_trajectory_data_extended_1.zip -d data/trajectory
unzip -o data/downloads/autoBidding_aigb_track_data_trajectory_data_extended_2.zip -d data/trajectory
```

期望目录结构：

```text
data/
  traffic/
    period-7.csv
    period-8.csv
    period-9.csv
    period-10.csv
    period-11.csv
    period-12.csv
    period-13.csv
  trajectory/
    trajectory_data.csv
    trajectory_data_extended_1.csv
    trajectory_data_extended_2.csv
```

第一次训练或评估时会自动构建本地缓存：

| 缓存文件 | 生成位置 | 作用 |
| --- | --- | --- |
| `data/trajectory/trajectory_data.npz` | `bidding_train_env/baseline/dit/dataset.py` | 加速 CBD/DiT/CBD-PPO 的 trajectory 加载 |
| `data/traffic/period-*.csv.episodes.pkl` | `run/run_open_source_benchmark.py` | 加速公开 traffic episode 评估 |

如果源 CSV 更新，需要删除对应缓存后重新运行命令。

## 3. 训练基线模型

以下命令均默认在项目根目录执行。

### DT

推荐的公开数据训练命令：

```bash
mkdir -p logs
nohup python -u run/run_decision_transformer.py \
  --epochs 200 \
  --batch_size 32 \
  --save_path saved_model/DTtest \
  > logs/run_dt_100ep.log 2>&1 &
```

输出文件：

```text
saved_model/DTtest/dt.pt
saved_model/DTtest/normalize_dict.pkl
logs/run_dt_100ep.log
```

### DD

推荐的公开数据训练命令：

```bash
mkdir -p logs
nohup python -u run/run_decision_diffuser.py \
  --epochs 200 \
  --batch_size 1000 \
  --save_path saved_model/DDtest \
  > logs/run_dd_100ep.log 2>&1 &
```

输出文件：

```text
saved_model/DDtest/diffuser.pt
logs/run_dd_100ep.log
```

### DiT

推荐的公开数据训练命令：

```bash
mkdir -p logs
nohup python -u run/run_dit.py \
  --epochs 200 \
  --batch_size 1000 \
  --n_timesteps 10 \
  --save_path saved_model/DiTtest \
  --save_every 20 \
  > logs/run_dit_100ep.log 2>&1 &
```

输出文件：

```text
saved_model/DiTtest/diffuser_best.pt
saved_model/DiTtest/diffuser.pt
logs/run_dit_100ep.log
```

### CBD

推荐的公开数据训练命令：

```bash
mkdir -p logs
nohup python -u run/run_cbd.py \
  --epochs 200 \
  --batch_size 1000 \
  --n_timesteps 10 \
  --save_path saved_model/CBDtest \
  --save_every 20 \
  > logs/run_cbd_100ep.log 2>&1 &
```

输出文件：

```text
saved_model/CBDtest/diffuser_best.pt
saved_model/CBDtest/diffuser.pt
logs/run_cbd_100ep.log
```

训练日志会记录设备和训练指标。A10 机器上如果 PyTorch 可以识别 CUDA，`run/run_cbd.py` 会自动选择 `cuda:0`，日志中会出现 `Using device: cuda:0`。CBD 训练过程中第 1 个 epoch 和每 10 个 epoch 会输出一次 `loss`、`diff`、`inv`：

```text
Epoch 1: loss=..., diff=..., inv=...
Epoch 10: loss=..., diff=..., inv=...
```

### CBD-PPO / DDPO-CBD

CBD-PPO 分为两个阶段：先按上一节训练基础 CBD checkpoint；然后进入 CBD-PPO 阶段，先预训练 trajectory reward model，再冻结 reward model 和 inverse dynamics model，最后只通过 PPO/DDPO 继续训练 CBD diffusion policy。

```bash
mkdir -p logs
nohup python -u run/run_cbd_ppo.py \
  --cbd_path saved_model/CBDtest/diffuser.pt \
  --save_path saved_model/DDPO-CBD \
  --epochs 200 \
  --batch_size 32 \
  --pretrain_rm_epochs 2 \
  --ppo_update_iters 4 \
  --clip_param 0.2 \
  --kl_coef 0.1 \
  --condition_steps 24 \
  --reward_eval_freq 10 \
  > logs/run_cbd_ppo_100ep.log 2>&1 &
```

输出文件：

```text
saved_model/DDPO-CBD/diffuser_best.pt
saved_model/DDPO-CBD/diffuser.pt
saved_model/DDPO-CBD/reward_model_best.pt
saved_model/DDPO-CBD/reward_model.pt
saved_model/DDPO-CBD/training_history.json
logs/run_cbd_ppo_100ep.log
```

CBD-PPO 日志中会先出现 `RM Pretrain`，表示先训练 RM；随后出现 `Starting DDPO/PPO fine-tuning` 和 `CBD-PPO Epoch`，表示开始训练 CBD policy。训练完成后，`training_history.json` 会记录 policy 参数更新和 inverse dynamics 冻结检查。

`training_history.json` 中建议重点检查：

| 字段 | 期望含义 |
| --- | --- |
| `diffuser_param_delta` | PPO 更新后应大于 0，表示 diffusion policy 参数被更新 |
| `inv_param_delta` | 应保持为 0，因为 inverse dynamics 已冻结 |
| `policy_loss`, `kl_penalty`, `ratio_mean`, `clip_fraction` | PPO/DDPO 训练诊断指标 |

## 4. 统一离线评估

推荐使用 `run/run_open_source_benchmark.py` 做统一离线评估。该脚本会回放公开 PV 日志，在每个 timestep 调用 agent 出价，然后基于日志中的 `leastWinningCost` 模拟参竞过程，最终汇总 reward、cost、CPA、score、预算利用率和零转化比例等指标。

在全部公开 period 上评估全部基线方法：

```bash
python run/run_open_source_benchmark.py \
  --traffic_dir data/traffic \
  --periods 7 8 9 10 11 12 13 \
  --models Heuristic DT DD DiT CBD CBD-PPO \
  --repeats 1 \
  --output_dir results/open_source_benchmark \
  --dit_model_path saved_model/DiTtest/diffuser_best.pt \
  --cbd_model_path saved_model/CBDtest/diffuser_best.pt \
  --cbd_ppo_model_path saved_model/DDPO-CBD/diffuser_best.pt
```

如果只想快速验证流程，可以限制每个 period 的 advertiser episode 数量：

```bash
python run/run_open_source_benchmark.py \
  --traffic_dir data/traffic \
  --periods 7 \
  --models Heuristic DT DD DiT CBD CBD-PPO \
  --max_episodes_per_period 2 \
  --output_dir results/open_source_benchmark_smoke
```

输出文件：

```text
results/open_source_benchmark/benchmark_<timestamp>.md
results/open_source_benchmark/benchmark_<timestamp>.json
```

Markdown 报告会包含最终需要的结果表：

```markdown
## Overall

| Model | Mean Score | Mean Reward | Mean CPA | Mean Budget Util. | Zero-Conv Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic | ... | ... | ... | ... | ... |
| DT | ... | ... | ... | ... | ... |
| DD | ... | ... | ... | ... | ... |
| DiT | ... | ... | ... | ... | ... |
| CBD | ... | ... | ... | ... | ... |
| CBD-PPO | ... | ... | ... | ... | ... |

## Per Period

### Period-7

| Model | Mean Score | Mean Reward | Mean CPA | Mean Budget Util. |
| --- | ---: | ---: | ---: | ---: |
```

## 5. 可选 RM Alignment 评估

`run/run_public_rm_alignment.py` 使用离线 reward model 评估生成轨迹质量。该指标不是主要竞赛式 score，但可用于检查 CBD-PPO 在 learned RM 下是否提升了生成轨迹 reward。

```bash
python run/run_public_rm_alignment.py \
  --train_data_path data/trajectory/trajectory_data.csv \
  --cbd_path saved_model/CBDtest/diffuser_best.pt \
  --cbd_ppo_path saved_model/DDPO-CBD/diffuser_best.pt \
  --rm_epochs 2 \
  --batch_size 32 \
  --output_dir results/public_rm_alignment
```

输出文件：

```text
results/public_rm_alignment/public_rm_alignment_<timestamp>.md
results/public_rm_alignment/public_rm_alignment_<timestamp>.json
```

报告表格：

```markdown
| Model | RM Score | RM Std | Relative Lift vs CBD |
| --- | ---: | ---: | ---: |
| CBD | ... | ... | -- |
| CBD-PPO | ... | ... | ... |
```

## 6. 旧版单策略评估

`run/run_evaluate.py` 会在 `data/traffic/period-7.csv` 上评估 `bidding_train_env/strategy/__init__.py` 中导出的 `PlayerBiddingStrategy`。该路径适合单策略调试，但不推荐用于产出最终 all-baseline 表格。

```bash
python -m run.run_evaluate
```

旧版对比脚本：

```bash
python run/compare_evaluate.py
python run/compare_dit_dt_dd.py
python run/compare_cbd_dt_dd_dit.py
python run/compare_cbd_ddpo.py
```

这些脚本会通过改写 `bidding_train_env/strategy/__init__.py` 来切换策略。若要产出可复现的多模型表格，建议使用 `run/run_open_source_benchmark.py`，因为它会直接实例化策略，不需要改写 import。

## 7. 推荐完整复现顺序

```bash
# 1. 环境
conda activate nips-bidding-env
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 2. 检查数据文件
ls data/traffic/period-{7,8,9,10,11,12,13}.csv
ls data/trajectory/trajectory_data.csv

# 3. 训练基线
mkdir -p logs
nohup python -u run/run_decision_transformer.py --epochs 200 --batch_size 32 --save_path saved_model/DTtest > logs/run_dt_100ep.log 2>&1 &
nohup python -u run/run_decision_diffuser.py --epochs 200 --batch_size 1000 --save_path saved_model/DDtest > logs/run_dd_100ep.log 2>&1 &
nohup python -u run/run_dit.py --epochs 200 --batch_size 1000 --n_timesteps 10 --save_path saved_model/DiTtest --save_every 20 > logs/run_dit_100ep.log 2>&1 &
nohup python -u run/run_cbd.py --epochs 200 --batch_size 1000 --n_timesteps 10 --save_path saved_model/CBDtest --save_every 20 > logs/run_cbd_100ep.log 2>&1 &

# CBD-PPO 阶段：读取上一步 CBD checkpoint，先训练 RM，再冻结 RM/IDM 并用 DDPO/PPO 微调 CBD policy
nohup python -u run/run_cbd_ppo.py --cbd_path saved_model/CBDtest/diffuser.pt --save_path saved_model/DDPO-CBD --epochs 200 --batch_size 32 --pretrain_rm_epochs 2 --ppo_update_iters 4 --clip_param 0.2 --kl_coef 0.1 --condition_steps 24 > logs/run_cbd_ppo_100ep.log 2>&1 &

# 4. 产出最终离线对比表
python run/run_open_source_benchmark.py \
  --traffic_dir data/traffic \
  --periods 7 8 9 10 11 12 13 \
  --models Heuristic DT DD DiT CBD CBD-PPO \
  --repeats 1 \
  --output_dir results/open_source_benchmark \
  --dit_model_path saved_model/DiTtest/diffuser_best.pt \
  --cbd_model_path saved_model/CBDtest/diffuser_best.pt \
  --cbd_ppo_model_path saved_model/DDPO-CBD/diffuser_best.pt

# 5. 可选 RM alignment 表
python run/run_public_rm_alignment.py \
  --train_data_path data/trajectory/trajectory_data.csv \
  --cbd_path saved_model/CBDtest/diffuser_best.pt \
  --cbd_ppo_path saved_model/DDPO-CBD/diffuser_best.pt \
  --output_dir results/public_rm_alignment
```

## 8. 需要汇报的结果文件

撰写论文或实验报告时，优先引用以下文件：

| 用途 | 文件路径模式 |
| --- | --- |
| 全部基线离线结果表 | `results/open_source_benchmark/benchmark_<timestamp>.md` |
| 全部基线机器可读指标 | `results/open_source_benchmark/benchmark_<timestamp>.json` |
| RM alignment 表 | `results/public_rm_alignment/public_rm_alignment_<timestamp>.md` |
| CBD-PPO 训练诊断 | `saved_model/DDPO-CBD/training_history.json` |

# 原始 Baseline 说明

以下内容是竞赛 starter kit 中保留的原始 baseline 使用说明。若目标是产出完整对比表，请优先使用上面的“可复现离线基线实验流程”。

## 使用说明
## 数据集链接
由于数据文件较大，公开数据被拆分为多个压缩包下载。

https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_period_7-8.zip

https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_period_9-10.zip

https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_period_11-12.zip

https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_period_13.zip
<br>
<br>
https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_trajectory_data.zip

https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_trajectory_data_extended_1.zip

https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_trajectory_data_extended_2.zip


## 数据处理
项目提供 traffic 粒度数据以及用于模型训练的 trajectory 数据。
下载 traffic 数据和 trajectory 数据后，将其放置到项目的 `data/` 目录下。
`data/` 目录结构应为：
```
NeurIPS_Auto_Bidding_AIGB_Track_Baseline
|── data
    |── traffic
        |── period-7.csv
        |── period-8.csv
        |── period-9.csv
        |── period-10.csv
        |── period-11.csv
        |── period-12.csv
        |── period-13.csv
    |── trajectory
        |── trajectory_data.csv
        |── trajectory_data_extended_1.csv
        |── trajectory_data_extended_2.csv
```

## 训练模型
### Decision-Transformer
加载训练数据并训练 DT 出价策略。
```
python main/main_decision_transformer.py 
```
评估时将 `DtBiddingStrategy` 作为 `PlayerBiddingStrategy`。
```
bidding_train_env/strategy/__init__.py
from .dt_bidding_strategy import DtBiddingStrategy as PlayerBiddingStrategy
```

### Decision-Diffusion
加载训练数据并训练 DD 出价策略。
```
python main/main_decision_diffuser.py
```
评估时将 `DdBiddingStrategy` 作为 `PlayerBiddingStrategy`。
```
bidding_train_env/strategy/__init__.py
from .dd_bidding_strategy import DdBiddingStrategy as PlayerBiddingStrategy
```


## 离线评估
加载训练数据并构建离线评估环境，用于评估出价策略效果。
```
python main/main_test.py
```

# 附录

## Traffic 粒度数据格式
The training dataset is derived from advertising delivery data generated via the auction system where multiple advertisers compete against each other. Participants can use this dataset to recreate the historical delivery process of all advertisers across all impression opportunities. The training dataset includes 7 delivery periods. Each delivery period contains approximately 500,000 impression opportunities and is divided into 48 steps. There are 48 advertisers competing for these opportunities. The dataset consists of approximately 170 million records, with a total size of 25G. The specific data format is as follows:

* **(c1) deliveryPeriodIndex**: Represents the index of the current delivery period.
* **(c2) advertiserNumber**: Represents the unique identifier of the advertiser.
* **(c3) advertiserCategoryIndex**: Represents the index of the advertiser's industry category.
* **(c4) budget**: Represents the advertiser's budget for a delivery period.
* **(c5) CPAConstraint**: Represents the CPA constraint of the advertiser.
* **(c6) timeStepIndex**: Represents the index of the current decision time step.
* **(c7) remainingBudget**: Represents the advertiser's remaining budget before the current step.
* **(c8) pvIndex**: Represents the index of the impression opportunity.
* **(c9) pValue**: Represents the conversion action probability when the advertisement is exposed to the customer.
* **(c10) pValueSigma**: Invalid variable, Constantly zero, please ignore.
* **(c11) bid**: Represents the advertiser's bid for the impression opportunity.
* **(c12) xi**: Represents the winning status of the advertiser for the impression opportunity, where 1 implies winning the opportunity and 0 suggests not winning the opportunity.
* **(c13) adSlot**: Represents the won ad slot. The value ranges from 1 to 3, with 0 indicating not winning the opportunity .
* **(c14) cost**: Represents the cost that the advertiser needs to pay if the ad is exposed to the customer.
* **(c15) isExposed**: Represents whether the ad in the slot was displayed to the customer, where 1 implies the ad is exposed and 0 suggests not exposed.
* **(c16) conversionAction**: Represents whether the conversion action has occurred, where 1 implies the occurrence of the conversion action and 0 suggests that it has not occurred.
* **(c17) leastWinningCost**: Represents the minimum cost to win the impression opportunity,i.e., the 4-th highest bid of the impression opportunity.
* **(c18) isEnd**: Represents the completion status of the advertising period, where 1 implies either the final decision step of the delivery period or the advertiser's remaining budget falling below the system-set minimum remaining budget.


## 训练数据示例
### example-1

| c1 |  c2 | c3 |   c4   |  c5  | c6 |   c7   |  c8   |   c9    |  c10   |  c11  | c12 | c13 |  c14  | c15 | c16 |  c17  | c18 |
|----|-----|----|--------|------|----|--------|-------|---------|--------|-------|-----|-----|-------|-----|-----|-------|-----|
|  1 |  31 |  2 | 6500.00| 27.00|  5 | 5962.49| 101000| 0.0103542| 0 | 0.2845 |  1  |  1  | 0.2702|  1  |  0  | 0.1832|  0  |
|  1 |  22 |  6 | 7000.00| 38.00|  5 | 5988.25| 101000| 0.0070297| 0 | 0.2702 |  1  |  2  | 0.2154|  1  |  1  | 0.1832|  0  |
|  1 |  15 |  7 | 7000.00| 42.00|  5 | 6132.52| 101000| 0.0051392| 0 | 0.2154 |  1  |  3  | 0.1832|  0  |  0  | 0.1832|  0  |
|  1 |  39 |  3 | 6000.00| 30.00|  5 | 5443.27| 101000| 0.0062134| 0 | 0.1832 |  0  |  0  | 0     |  0  |  0  | 0.1832|  0  |
|  1 |  43 |  9 | 7500.00| 25.00|  5 | 6421.81| 101000| 0.0045392| 0 | 0.1099 |  0  |  0  | 0     |  0  |  0  | 0.1832|  0  |

This example presents an impression opportunity involving the top five advertisers. The top three advertisers, numbered 31, 22, and 15, won the impression opportunity with the highest bids and were allocated to ad slots 1, 2, and 3, respectively. During this impression, slots 1 and 2 were exposed to the customer, while slot 3 remained unexposed. Consequently, ads in slots 1 and 2 need to pay 0.2702 and 0.2154, respectively. Additionally, the customer engaged in a conversion action with the ad in slot 2.


### example-2

| c1 | c2 | c3 | c4     | c5   | c6 | c7     | c8   | c9        | c10       | c11   | c12 | c13 | c14   | c15 | c16 | c17   | c18 |
|----|----|----|--------|------|----|--------|------|-----------|-----------|-------|-----|-----|-------|-----|-----|-------|-----|
| 3  | 48 | 6  | 7500.00| 40.00| 1  | 7500.00| 1    | 0.0032157 | 0 | 0.1345| 0   | 0   | 0     | 0   | 0   | 0.1628| 0   |
| 3  | 48 | 6  | 7500.00| 40.00| 1  | 7500.00| 2    | 0.0146256 | 0 | 0.5852| 0   | 0   | 0     | 0   | 0   | 0.6421| 0   |
| 3  | 48 | 6  | 7500.00| 40.00| 1  | 7500.00| 3    | 0.0054324 | 0 | 0.1924| 1   | 1   | 0.1673 | 1   | 1   | 0.1454| 0   |
| 3  | 48 | 6  | 7500.00| 40.00| 1  | 7500.00| 4    | 0.0073145 | 0 | 0.2786| 0   | 0   | 0     | 0   | 0   | 0.2862| 0   |
| …  |
| 3  | 48 | 6  | 7500.00| 40.00| 2  | 7341.25| 20901| 0.0076453 | 0 | 0.2856| 0   | 0   | 0     | 0   | 0   | 0.3245| 0   |
| 3  | 48 | 6  | 7500.00| 40.00| 2  | 7341.25| 20902| 0.0139234 | 0 | 0.5629| 1   | 2   | 0     | 0   | 0   | 0.6782| 0   |
| 3  | 48 | 6  | 7500.00| 40.00| 2  | 7341.25| 20903| 0.0077212 | 0 | 0.3045| 0   | 0   | 0     | 0   | 0   | 0.3122| 0   |
| 3  | 48 | 6  | 7500.00| 40.00| 2  | 7341.25| 20904| 0.0021341 | 0 | 0.0926| 0   | 0   | 0     | 0   | 0   | 0.1151| 0   |
| …  |
| 3  | 48 | 6  | 7500.00| 40.00| 43 | 0.00   | 895201| 0.0065274 | 0 | 0.0000| 0   | 0   | 0     | 0   | 0   | 0.1243| 1   |
| 3  | 48 | 6  | 7500.00| 40.00| 43 | 0.00   | 895202| 0.0032125 | 0 | 0.0000| 0   | 0   | 0     | 0   | 0   | 0.2986| 1   |
| 3  | 48 | 6  | 7500.00| 40.00| 43 | 0.00   | 895203| 0.0112986 | 0 | 0.0000| 0   | 0   | 0     | 0   | 0   | 0.0932| 1   |
| 3  | 48 | 6  | 7500.00| 40.00| 43 | 0.00   | 895204| 0.0051678 | 0 | 0.0000| 0   | 0   | 0     | 0   | 0   | 0.1687| 1   |

This example presents a data sample illustrating an advertiser's bidding process across time steps within a delivery period. The advertiser has a budget of 7500, a CPA constraint of 40, and belongs to industry category 6. Throughout different time steps, the advertiser engages in bidding for every available impression and obtains the corresponding results. During this period, the advertiser's remaining budget decreases correspondingly. Additionally, the advertiser adjusts their bidding strategy based on prior performance, although this adjustment will not be directly evident in the data.

## Trajectory 数据格式

Trajectory data is converted from traffic granularity data. It records information for multiple advertisers over different time steps across multiple periods as (s, a, r, s').
you can refer to the code provided below.
```
python  bidding_train_env/dataloader/rl_data_generator.py
```


* **(c1) deliveryPeriodIndex**: Represents the index of the current delivery period.
* **(c2) advertiserNumber**: Represents the unique identifier of the advertiser.
* **(c3) advertiserCategoryIndex**: Represents the index of the advertiser's industry category.
* **(c4) budget**: Represents the advertiser's budget for a delivery period.
* **(c5) CPAConstraint**: Represents the CPA constraint of the advertiser.
* **(c6) realAllCost**: Represents the cost of the advertiser during the entire period.
* **(c7) realAllConversion**: Represents the conversions of the advertiser during the entire period.
* **(c8) timeStepIndex**: Represents the index of the current decision time step.
* **(c9) state**: Represents the advertiser's state in this timeStep.
* **(c10) action**: Represents the advertiser's action in this timeStep.
* **(c11) reward**: Represents the advertiser's sparse reward(total conversion) in this timeStep.
* **(c12) reward_continuous**: Represents the advertiser's continuous reward(The sum of the pValues of all exposed traffic) in this timeStep.
* **(c13) done**: Represents the completion status of the advertising period, where 1 implies either the final decision step of the delivery period or the advertiser's remaining budget falling below the system-set minimum remaining budget.
* **(c14) next_state**: Represents the advertiser's next state in this timeStep.



# 参考资料
Decision Transformer 实现参考：
https://github.com/kzl/decision-transformer

Decision Diffusion 实现参考：
https://github.com/anuragajay/decision-diffuser/tree/main/code


# dev
```
cd /share/rongyu03/rl/wentou/Bidding_CBD_PPO

```
