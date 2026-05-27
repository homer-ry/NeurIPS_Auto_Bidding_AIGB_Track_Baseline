from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from zipfile import BadZipFile



class aigb_dataset(Dataset):
    def __init__(self, step_len, **kwargs) -> None:
        super().__init__()
        train_data_path = kwargs.get("train_data_path", "data/trajectory/trajectory_data.csv")
        self.return_transform = kwargs.get("return_transform", "log1p")
        self.return_clip_quantile = kwargs.get("return_clip_quantile", 0.99)
        states, actions, rewards, terminals = load_local_data_nips(
            train_data_path=train_data_path)
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.step_len = step_len
        self.num_of_states = states.shape[1]
        self.num_of_actions = actions.shape[1]

        # 分割序列
        # 每个序列的开头
        self.candidate_pos = (self.terminals == 0).nonzero()[0]
        self.candidate_pos += 1
        self.candidate_pos = [0] + self.candidate_pos.tolist()[:-1]
        # 后面再加上序列的结尾
        self.candidate_pos = self.candidate_pos + [self.states.shape[0]]
        self.episode_returns = self._build_episode_returns()
        self.return_scale = self._build_return_scale()

    def _build_episode_returns(self):
        returns = []
        for index in range(len(self.candidate_pos) - 1):
            start = self.candidate_pos[index]
            end = self.candidate_pos[index + 1]
            returns.append(float(self.rewards[start:end, :].sum()))
        return np.asarray(returns, dtype=np.float32)

    def _build_return_scale(self):
        if len(self.episode_returns) == 0:
            return 1.0
        clipped = float(np.quantile(self.episode_returns, self.return_clip_quantile))
        return max(clipped, 1.0)

    def _normalize_return(self, raw_return):
        if self.return_transform == "sigmoid":
            return torch.sigmoid(torch.tensor(raw_return, dtype=torch.float32))
        if self.return_transform == "log1p":
            normalized = np.log1p(raw_return) / np.log1p(self.return_scale)
            return torch.tensor(min(max(normalized, 0.0), 1.0), dtype=torch.float32)
        if self.return_transform == "linear":
            normalized = raw_return / self.return_scale
            return torch.tensor(min(max(normalized, 0.0), 1.0), dtype=torch.float32)
        raise ValueError(f"Unsupported return_transform: {self.return_transform}")

    def __len__(self):
        return len(self.candidate_pos) - 1

    def __getitem__(self, index):
        # 获取序列
        state = torch.tensor(self.states[self.candidate_pos[index]:self.candidate_pos[index + 1], :],
                             dtype=torch.float32)
        action = torch.tensor(self.actions[self.candidate_pos[index]:self.candidate_pos[index + 1], :],
                              dtype=torch.float32)
        reward = torch.tensor(self.rewards[self.candidate_pos[index]:self.candidate_pos[index + 1], :],
                              dtype=torch.float32)
        action = action - 1
        # 当前序列的长度
        len_state = len(state)
        # 进行padding
        state = torch.nn.functional.pad(state, (0, 0, 0, self.step_len - len(state)), "constant", 0)
        action = torch.nn.functional.pad(action, (0, 0, 0, self.step_len - len(action)), "constant", 0)
        # 计算returns
        returns = self._normalize_return(self.episode_returns[index]).reshape(1)
        # 计算masks
        masks = torch.zeros(self.step_len)
        masks[:len_state] = 1
        masks = masks.bool()
        # 返回
        return state, action, returns, masks


# 加载本地数据
def load_local_data(data_version):
    states = pd.read_csv("simulation_platform/data/offline_trajectory/" + data_version + "/states.csv").values[:,
             0::]
    actions = pd.read_csv("simulation_platform/data/offline_trajectory/" + data_version + "/actions.csv").values[:,
              0::]
    rewards = pd.read_csv("simulation_platform/data/offline_trajectory/" + data_version + "/rewards.csv").values[:,
              0::]
    terminals = pd.read_csv("simulation_platform/data/offline_trajectory/" + data_version + "/terminal.csv").values[
                :,
                0::]
    return states, actions, rewards, terminals


def load_local_data_nips(train_data_path="data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"):
    cache_path = Path(train_data_path).with_suffix(".npz")
    if cache_path.exists() and cache_path.stat().st_mtime >= Path(train_data_path).stat().st_mtime:
        try:
            cached = np.load(cache_path, allow_pickle=False)
            return (
                cached["states"],
                cached["actions"],
                cached["rewards"],
                cached["terminals"],
            )
        except (BadZipFile, ValueError, OSError):
            cache_path.unlink(missing_ok=True)

    training_data = pd.read_csv(
        train_data_path,
        usecols=["state", "timeStepIndex", "action", "reward"],
    )

    def parse_state(val):
        if pd.isna(val):
            return np.zeros(16, dtype=np.float32)
        stripped = str(val).strip().strip("()[]")
        return np.fromstring(stripped, sep=",", dtype=np.float32)

    # 使用apply方法应用上述函数
    training_data["state"] = training_data["state"].apply(parse_state)
    training_data["terminal"] = training_data["timeStepIndex"] != 47
    training_data["terminal"] = training_data["terminal"].astype(int)
    states = np.array(training_data['state'].tolist())
    actions = training_data["action"].to_numpy().reshape(-1, 1)
    rewards = training_data["reward"].to_numpy().reshape(-1, 1)
    terminals = training_data["terminal"].to_numpy().reshape(-1, 1)
    np.savez_compressed(
        cache_path,
        states=states.astype(np.float32, copy=False),
        actions=actions.astype(np.float32, copy=False),
        rewards=rewards.astype(np.float32, copy=False),
        terminals=terminals.astype(np.int8, copy=False),
    )
    return states, actions, rewards, terminals
