"""
Reward Model for Trajectory Evaluation in Bidding

基于轨迹的未来回报预测模型，用于DDPO训练时的reward信号
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class TrajectoryRewardModel(nn.Module):
    """
    Reward Model that predicts the return of a trajectory
    用于DDPO的reward预测，输入状态序列，输出预测的return
    """
    
    def __init__(self, state_dim=16, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.input_scale = 100.0
        
        # State embedding
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        
        # Positional encoding for time steps
        self.pos_embed = PositionalEncoding(hidden_dim, dropout=dropout, max_len=48)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Reward prediction head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
    def forward(self, states, masks=None):
        """
        Args:
            states: [batch, horizon, state_dim]
            masks: [batch, horizon] - valid trajectory mask
        Returns:
            reward: [batch, 1] - predicted return
        """
        states = torch.nan_to_num(states, nan=0.0, posinf=self.input_scale, neginf=-self.input_scale)
        states = torch.asinh(states.clamp(min=-1e6, max=1e6)) / math.asinh(self.input_scale)

        # Embed states
        x = self.state_embed(states)  # [batch, horizon, hidden_dim]
        
        # Add positional encoding
        x = self.pos_embed(x)
        
        # Apply transformer
        if masks is not None:
            # Create attention mask for padding
            key_padding_mask = ~(masks.bool())  # True for padded positions
            x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        else:
            x = self.transformer(x)
        
        # Global average pooling over valid positions
        if masks is not None:
            masks_float = masks.float()
            x = (x * masks_float.unsqueeze(-1)).sum(dim=1) / masks_float.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        # Predict reward
        reward = self.reward_head(x)
        return reward


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=48):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.dim // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class GroupNorm1d(nn.Module):
    def __init__(self, dim, num_groups=8, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = max(1, min(num_groups, dim // min_channels_per_group))
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x.unsqueeze(2),
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x.squeeze(2)


def get_norm(dim: int, norm_type: str = "groupnorm"):
    if norm_type == "groupnorm":
        return GroupNorm1d(dim)
    if norm_type == "layernorm":
        return nn.GroupNorm(1, dim)
    return nn.Identity()


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, emb_dim: int, kernel_size: int = 3, norm_type: str = "groupnorm"):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size // 2),
            get_norm(out_dim, norm_type),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size // 2),
            get_norm(out_dim, norm_type),
            nn.Mish(),
        )
        self.emb_mlp = nn.Sequential(nn.Mish(), nn.Linear(emb_dim, out_dim))
        self.residual_conv = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, emb):
        out = self.conv1(x) + self.emb_mlp(emb).unsqueeze(-1)
        out = self.conv2(out)
        return out + self.residual_conv(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class UNetTrajectoryRewardModel(nn.Module):
    """
    Half-Janner 1D U-Net return model adapted to this project's AIGB trajectories.

    The external wentou implementation predicts a trajectory-level return from
    (state[, action]) sequences. This version keeps the same downsampling U-Net
    idea, but supports 48-step padded trajectories, 16-dim states, masks, and
    stable input compression for the large raw bidding features.
    """

    def __init__(
        self,
        state_dim=16,
        horizon=48,
        hidden_dim=128,
        emb_dim=128,
        out_dim=1,
        kernel_size=3,
        dim_mult: Tuple[int, ...] = (1, 2, 2, 2),
        norm_type="groupnorm",
        traj_add_a=False,
        input_scale=100.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.traj_add_a = traj_add_a
        self.input_scale = input_scale
        in_dim = state_dim + 1 if traj_add_a else state_dim

        dims = [in_dim] + [hidden_dim * m for m in torch.tensor(dim_mult).cumprod(0).tolist()]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.map_noise = PositionalEmbedding(emb_dim)
        self.map_emb = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.downs = nn.ModuleList([])
        current_horizon = horizon
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index >= len(in_out) - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualBlock(dim_in, dim_out, hidden_dim, kernel_size, norm_type),
                        ResidualBlock(dim_out, dim_out, hidden_dim, kernel_size, norm_type),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
            if not is_last:
                current_horizon = (current_horizon + 1) // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        self.mid_block1 = nn.ModuleList(
            [
                ResidualBlock(mid_dim, mid_dim_2, hidden_dim, kernel_size=5, norm_type=norm_type),
                Downsample1d(mid_dim_2),
            ]
        )
        current_horizon = (current_horizon + 1) // 2
        fc_dim = mid_dim_2 * max(current_horizon, 1)
        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + hidden_dim, max(fc_dim // 2, out_dim)),
            nn.Mish(),
            nn.Linear(max(fc_dim // 2, out_dim), out_dim),
            nn.Sigmoid(),
        )

    def _prepare_input(self, states, actions=None, masks=None):
        states = torch.nan_to_num(states, nan=0.0, posinf=self.input_scale, neginf=-self.input_scale)
        states = torch.asinh(states.clamp(min=-1e6, max=1e6)) / math.asinh(self.input_scale)
        if self.traj_add_a:
            if actions is None:
                action_part = torch.zeros(states.shape[0], states.shape[1], 1, device=states.device, dtype=states.dtype)
            else:
                action_part = torch.nan_to_num(actions, nan=0.0, posinf=self.input_scale, neginf=-self.input_scale)
                action_part = torch.asinh(action_part.clamp(min=-1e6, max=1e6)) / math.asinh(self.input_scale)
            states = torch.cat([action_part, states], dim=-1)
        if masks is not None:
            states = states * masks.float().unsqueeze(-1)
        return states

    def forward(self, states, masks=None, actions=None, t: Optional[torch.Tensor] = None, condition=None):
        x = self._prepare_input(states, actions=actions, masks=masks).permute(0, 2, 1)
        if t is None:
            t = torch.full((x.shape[0],), 1000, device=x.device, dtype=torch.long)
        emb = self.map_noise(t)
        if condition is not None:
            emb = emb + condition
        emb = self.map_emb(emb)

        for resnet1, resnet2, downsample in self.downs:
            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = downsample(x)

        x = self.mid_block1[0](x, emb)
        x = self.mid_block1[1](x)
        x = x.flatten(1)
        return self.final_block(torch.cat([x, emb], dim=-1))


class CPAScoreModel(nn.Module):
    """
    CPA-aware Score Model
    结合CPA约束的评分模型，考虑CPA是否满足约束
    """
    
    def __init__(self, state_dim=16, hidden_dim=128, cpa_dim=1):
        super().__init__()
        self.state_dim = state_dim
        
        # Base reward model
        self.reward_model = TrajectoryRewardModel(state_dim, hidden_dim)
        
        # CPA constraint encoder
        self.cpa_embed = nn.Linear(cpa_dim, hidden_dim)
        
        # Combined score head
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, states, cpa_constraint, masks=None):
        """
        Args:
            states: [batch, horizon, state_dim]
            cpa_constraint: [batch] - CPA constraint value
            masks: [batch, horizon]
        Returns:
            score: [batch, 1] - combined score considering CPA
        """
        # Get base reward
        reward_embed = self.reward_model.reward_head[
            :-1  # Remove final sigmoid
        ](self._get_reward_embed(states, masks))
        
        # CPA embedding
        cpa_embed = self.cpa_embed(cpa_constraint.unsqueeze(-1))
        
        # Combine
        combined = torch.cat([reward_embed, cpa_embed], dim=-1)
        score = self.score_head(combined)
        
        return score
    
    def _get_reward_embed(self, states, masks):
        x = self.reward_model.state_embed(states)
        x = self.reward_model.pos_embed(x)
        if masks is not None:
            key_padding_mask = ~masks
            x = self.reward_model.transformer(x, src_key_padding_mask=key_padding_mask)
        else:
            x = self.reward_model.transformer(x)
        if masks is not None:
            x = (x * masks.unsqueeze(-1)).sum(dim=1) / masks.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return x


class MLPStateEncoder(nn.Module):
    """Simple MLP-based state encoder for ablation"""
    
    def __init__(self, state_dim=16, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, states, masks=None):
        """
        Args:
            states: [batch, horizon, state_dim]
            masks: [batch, horizon]
        Returns:
            reward: [batch, 1]
        """
        # Encode each state
        encoded = self.encoder(states)  # [batch, horizon, hidden_dim//2]
        
        # Pool over time
        if masks is not None:
            encoded = (encoded * masks.unsqueeze(-1)).sum(dim=1) / masks.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            encoded = encoded.mean(dim=1)
        
        # Predict reward
        return self.reward_head(encoded)


def compute_score_from_trajectory(states, rewards, cpa_constraint, cpa_real, gamma=0.99):
    """
    Compute the score for a trajectory based on the scoring function from the competition
    
    Score = reward - gamma * max(0, cpa_real - cpa_constraint)
    
    Args:
        states: [batch, horizon, state_dim] - state trajectory
        rewards: [batch, horizon, 1] - reward trajectory  
        cpa_constraint: [batch] - CPA constraint
        cpa_real: [batch] - actual CPA achieved
        gamma: penalty coefficient
    Returns:
        score: [batch, 1] - final score
    """
    total_reward = rewards.sum(dim=1)
    cpa_penalty = gamma * torch.clamp(cpa_real - cpa_constraint, min=0)
    score = total_reward - cpa_penalty.unsqueeze(-1)
    return score
