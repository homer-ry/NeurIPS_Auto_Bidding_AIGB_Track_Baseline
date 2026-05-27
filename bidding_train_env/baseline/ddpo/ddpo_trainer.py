"""
DDPO (Denoising Diffusion Policy Optimization) for CBD

基于DDPO论文的方法，将CBD扩散模型视为策略，
使用PPO风格的策略梯度进行微调
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
import random


def extract(a, t, x_shape):
    """Extract values from tensor a at indices t and reshape"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, 1, 1)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule for beta values"""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.tensor(np.clip(betas, a_min=0, a_max=0.999), dtype=torch.float32)


class DDPOBuffer:
    """Buffer for storing DDPO training samples"""
    
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, sample: Dict):
        """Add a sample to the buffer"""
        self.buffer.append(sample)
        
    def sample(self, batch_size: int) -> Dict:
        """Sample a batch from the buffer"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return {
            key: torch.stack([b[key] for b in batch])
            for key in batch[0].keys()
        }
    
    def __len__(self):
        return len(self.buffer)


class DDPOTrainer:
    """
    DDPO Trainer for fine-tuning CBD diffusion model
    
    Key idea from DDPO paper:
    1. Treat diffusion sampling as a policy
    2. Use importance sampling with old policy log probs
    3. Apply PPO-style clipped objective
    """
    
    def __init__(
        self,
        cbd_model,
        reward_model,
        lr=1e-5,
        clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=1.0,
        n_timesteps=10,
        device='cuda'
    ):
        self.cbd_model = cbd_model
        self.reward_model = reward_model
        self.device = device
        
        # DDPO hyperparameters
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_timesteps = n_timesteps
        
        # Optimizers
        self.cbd_optimizer = Adam(cbd_model.parameters(), lr=lr)
        self.rm_optimizer = Adam(reward_model.parameters(), lr=lr)
        
        # Buffer for experience
        self.buffer = DDPOBuffer()
        
        # Statistics tracking
        self.stats = {
            'policy_loss': [],
            'reward_loss': [],
            'approx_kl': [],
            'clipfrac': [],
            'reward_mean': [],
            'reward_std': []
        }
        
        # Setup diffusion schedule
        self._setup_diffusion_schedule()
        
    def _setup_diffusion_schedule(self):
        """Setup noise schedule for diffusion"""
        betas = cosine_beta_schedule(self.n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # For sampling
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        
        # For posterior
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', 
                            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
    def register_buffer(self, name, tensor):
        """Register a buffer (similar to nn.Module)"""
        setattr(self, name, tensor.to(self.device))
        
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process: add noise to x_start at timestep t"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(t.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(t.device)
        
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def compute_log_prob(self, x_t, x_t_minus_1, t, model_output):
        """
        Compute log probability of transition from x_t to x_t_minus_1
        This is needed for importance sampling in DDPO
        
        Args:
            x_t: [batch, horizon, dim] noisy trajectory at step t
            x_t_minus_1: [batch, horizon, dim] trajectory at step t-1
            t: [batch] timestep
            model_output: model prediction
        """
        batch_size = x_t.shape[0]
        
        # Predict x_0 from model output
        if self.cbd_model.diffuser.predict_epsilon:
            sqrt_recip = self.sqrt_alphas_cumprod.reciprocal().to(t.device)
            sqrt_recip_m1 = self.sqrt_one_minus_alphas_cumprod.reciprocal().to(t.device)
            x_0_pred = (
                extract(sqrt_recip, t, x_t.shape) * x_t -
                extract(sqrt_recip_m1, t, x_t.shape) * model_output
            )
        else:
            x_0_pred = model_output
            
        # Clamp prediction
        x_0_pred = x_0_pred.clamp(-1, 1)
        
        # Get coefficients for posterior mean
        betas_t = self.betas.to(t.device)[t.long()]
        alphas_cumprod_t = self.alphas_cumprod.to(t.device)[t.long()]
        alphas_cumprod_prev_t = self.alphas_cumprod_prev.to(t.device)[t.long()]
        alphas_t = self.alphas.to(t.device)[t.long()]
        
        # Reshape for broadcasting: [batch, 1, 1]
        betas_t = betas_t.view(batch_size, 1, 1)
        alphas_cumprod_t = alphas_cumprod_t.view(batch_size, 1, 1)
        alphas_cumprod_prev_t = alphas_cumprod_prev_t.view(batch_size, 1, 1)
        alphas_t = alphas_t.view(batch_size, 1, 1)
        
        # Compute posterior mean
        posterior_mean = (
            betas_t * torch.sqrt(alphas_cumprod_prev_t) / (1 - alphas_cumprod_t) * x_0_pred +
            (1 - alphas_cumprod_prev_t) * torch.sqrt(alphas_t) / (1 - alphas_cumprod_t) * x_t
        )
        
        # Compute log prob under Gaussian
        log_var = self.posterior_log_variance_clipped.to(t.device)[t.long()]
        log_var = log_var.view(batch_size, 1, 1)
        
        log_prob = -0.5 * ((x_t_minus_1 - posterior_mean) ** 2 / log_var.exp() + log_var + np.log(2 * np.pi))
        
        return log_prob.sum(dim=-1)  # Sum over state dimensions
    
    def collect_samples(self, states: torch.Tensor, num_samples: int = 16):
        """
        Collect samples from the current CBD policy
        
        Args:
            states: [batch, state_dim] initial states
            num_samples: number of trajectories to sample per state
            
        Returns:
            samples: Dict with trajectories and log probs
        """
        batch_size = states.shape[0]
        
        # Expand states for multiple samples
        states_expanded = states.unsqueeze(1).expand(-1, num_samples, -1)  # [batch, num_samples, state_dim]
        states_expanded = states_expanded.reshape(-1, states.shape[-1])  # [batch*num_samples, state_dim]
        
        # Sample from CBD
        with torch.no_grad():
            returns = torch.ones(states_expanded.shape[0], 1, device=self.device)
            
            # Get predicted trajectories
            if hasattr(self.cbd_model, 'forward'):
                trajectories = self.cbd_model.diffuser.conditional_sample(
                    cond=states_expanded,
                    returns=returns,
                    horizon=48
                )
            else:
                # Fallback for different model interface
                trajectories = self.cbd_model.sample(states_expanded, returns)
        
        # Compute rewards from reward model
        with torch.no_grad():
            # Extract state trajectory
            if trajectories.dim() == 3 and trajectories.shape[-1] > states.shape[-1]:
                state_traj = trajectories[:, :, 1:]  # Skip action dimension if present
            else:
                state_traj = trajectories
                
            predicted_rewards = self.reward_model(state_traj)
        
        return {
            'states': states_expanded,
            'trajectories': trajectories,
            'rewards': predicted_rewards,
            'returns': returns
        }
    
    def compute_advantages(self, rewards: torch.Tensor, values: Optional[torch.Tensor] = None):
        """Compute advantages using GAE or simple normalization"""
        if values is not None:
            advantages = rewards - values
        else:
            advantages = rewards
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def policy_gradient_step(self, batch: Dict):
        """
        Single DDPO policy gradient step
        
        This implements the PPO-style clipped objective for diffusion policy
        """
        states = batch['states']
        old_log_probs = batch['log_probs']
        rewards = batch['rewards']
        old_trajectories = batch['trajectories']
        
        # Compute advantages
        advantages = self.compute_advantages(rewards)
        
        # Forward pass through CBD model
        batch_size = states.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device).long()
        
        # Get model predictions
        noise = torch.randn_like(old_trajectories)
        noisy_trajectories = self.q_sample(old_trajectories, t, noise)
        
        if hasattr(self.cbd_model, 'diffuser'):
            model_output = self.cbd_model.diffuser.model(noisy_trajectories, t, batch['returns'])
        else:
            model_output = self.cbd_model(noisy_trajectories, t, batch['returns'])
        
        # Compute new log probs
        new_log_probs = self.compute_log_prob(noisy_trajectories, old_trajectories, t, model_output)
        
        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus (encourages exploration)
        entropy = -(new_log_probs.exp() * new_log_probs).sum(dim=-1).mean()
        policy_loss = policy_loss - self.entropy_coef * entropy
        
        # Compute approx KL for logging
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
        clipfrac = ((ratio - 1).abs() > self.clip_range).float().mean()
        
        return policy_loss, {
            'approx_kl': approx_kl.item(),
            'clipfrac': clipfrac.item(),
            'entropy': entropy.item()
        }
    
    def train_reward_model(self, batch: Dict):
        """Train the reward model on collected samples"""
        trajectories = batch['trajectories']
        target_rewards = batch['rewards']
        
        # Extract states from trajectories
        if trajectories.dim() == 3 and trajectories.shape[-1] > 16:
            state_traj = trajectories[:, :, 1:]  # Skip action dimension
        else:
            state_traj = trajectories
            
        predicted_rewards = self.reward_model(state_traj)
        
        # MSE loss
        reward_loss = F.mse_loss(predicted_rewards, target_rewards)
        
        self.rm_optimizer.zero_grad()
        reward_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.max_grad_norm)
        self.rm_optimizer.step()
        
        return reward_loss.item()
    
    def train_step(self, batch: Dict):
        """Combined training step for DDPO"""
        # Train reward model
        reward_loss = self.train_reward_model(batch)
        
        # Train policy with DDPO
        policy_loss, info = self.policy_gradient_step(batch)
        
        self.cbd_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cbd_model.parameters(), self.max_grad_norm)
        self.cbd_optimizer.step()
        
        # Update stats
        self.stats['policy_loss'].append(policy_loss.item())
        self.stats['reward_loss'].append(reward_loss)
        self.stats['approx_kl'].append(info['approx_kl'])
        self.stats['clipfrac'].append(info['clipfrac'])
        
        return {
            'policy_loss': policy_loss.item(),
            'reward_loss': reward_loss,
            **info
        }
    
    def save(self, path: str):
        """Save trainer state"""
        torch.save({
            'cbd_model': self.cbd_model.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            'cbd_optimizer': self.cbd_optimizer.state_dict(),
            'rm_optimizer': self.rm_optimizer.state_dict(),
            'stats': self.stats
        }, path)
        
    def load(self, path: str):
        """Load trainer state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.cbd_model.load_state_dict(checkpoint['cbd_model'])
        self.reward_model.load_state_dict(checkpoint['reward_model'])
        self.cbd_optimizer.load_state_dict(checkpoint['cbd_optimizer'])
        self.rm_optimizer.load_state_dict(checkpoint['rm_optimizer'])
        self.stats = checkpoint.get('stats', self.stats)


class DDPOConfig:
    """Configuration for DDPO training"""
    
    def __init__(
        self,
        lr: float = 1e-5,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        n_timesteps: int = 10,
        batch_size: int = 64,
        n_epochs: int = 4,
        n_samples_per_state: int = 16,
        reward_update_freq: int = 10,
        device: str = 'cuda'
    ):
        self.lr = lr
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_timesteps = n_timesteps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_samples_per_state = n_samples_per_state
        self.reward_update_freq = reward_update_freq
        self.device = device
