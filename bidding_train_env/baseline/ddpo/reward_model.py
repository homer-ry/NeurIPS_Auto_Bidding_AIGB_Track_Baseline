"""
Reward Model for Trajectory Evaluation in Bidding

基于轨迹的未来回报预测模型，用于DDPO训练时的reward信号
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TrajectoryRewardModel(nn.Module):
    """
    Reward Model that predicts the return of a trajectory
    用于DDPO的reward预测，输入状态序列，输出预测的return
    """
    
    def __init__(self, state_dim=16, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
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
