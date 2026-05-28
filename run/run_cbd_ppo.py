"""
CBD-PPO Fine-tuning Script

This script adapts the external CBD-DDPO reference to this repository's AIGB
CBD implementation:

1. Load a pretrained CBD diffusion model.
2. Pretrain a trajectory reward model on offline trajectories.
3. Freeze the reward model and CBD inverse-dynamics model.
4. Sample CBD trajectories with per-step log probabilities.
5. Update only the CBD diffusion policy with PPO clipped objective and KL penalty.
6. Track policy/inverse parameter deltas to verify the intended module is updated.

The result keeps the local DFUSER / strategy / evaluation stack compatible while
matching the source project's DDPO/PPO training logic more closely.
"""

import argparse
import copy
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidding_train_env.baseline.ddpo.reward_model import TrajectoryRewardModel, UNetTrajectoryRewardModel
from bidding_train_env.baseline.dit.DFUSER import DFUSER
from bidding_train_env.baseline.dit.dataset import aigb_dataset
from run.training_log_utils import BatchMeanMeter, add_scalar, create_summary_writer, parse_bool_arg


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _l2_delta(before: Dict[str, torch.Tensor], module: torch.nn.Module) -> float:
    delta = 0.0
    for name, param in module.named_parameters():
        if name in before:
            delta += torch.sum((param.detach().cpu() - before[name]) ** 2).item()
    return float(delta ** 0.5)


class CBDDDPOTrainer:
    """DDPO/PPO trainer adapted to the local CBD diffusion implementation."""

    def __init__(
        self,
        cbd_model: DFUSER,
        reward_model: TrajectoryRewardModel,
        device: torch.device,
        lr: float,
        ppo_update_iters: int,
        clip_param: float,
        kl_coef: float,
        max_grad_norm: float,
        condition_steps: int,
        sample_value_clip: float,
    ):
        self.cbd_model = cbd_model
        self.diffuser = cbd_model.diffuser
        self.policy_model = self.diffuser.model
        self.reward_model = reward_model
        self.device = device
        self.ppo_update_iters = ppo_update_iters
        self.clip_param = clip_param
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm
        self.condition_steps = condition_steps
        self.sample_value_clip = sample_value_clip

        for param in self.reward_model.parameters():
            param.requires_grad = False
        for param in self.diffuser.inv_model.parameters():
            param.requires_grad = False

        self.ref_model = copy.deepcopy(self.policy_model).to(device).eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=lr)

    def _sanitize_sample(self, value: torch.Tensor) -> torch.Tensor:
        if self.sample_value_clip and self.sample_value_clip > 0:
            return torch.nan_to_num(
                value,
                nan=0.0,
                posinf=self.sample_value_clip,
                neginf=-self.sample_value_clip,
            ).clamp(min=-self.sample_value_clip, max=self.sample_value_clip)
        return torch.nan_to_num(value, nan=0.0)

    def _prefix_len(self, masks: torch.Tensor) -> torch.Tensor:
        valid_len = masks.long().sum(dim=1).clamp(min=1)
        if self.condition_steps > 0:
            prefix = torch.full_like(valid_len, self.condition_steps)
            return torch.minimum(prefix, valid_len)
        # Match the source project's Completer idea: keep a query prefix and optimize the answer suffix.
        return torch.clamp(valid_len // 2, min=1)

    def _apply_prefix(self, x: torch.Tensor, real_states: torch.Tensor, prefix_len: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        for row, cut in enumerate(prefix_len.tolist()):
            x[row, :cut, :] = real_states[row, :cut, :]
        return x

    def _answer_mask(self, masks: torch.Tensor, prefix_len: torch.Tensor) -> torch.Tensor:
        answer_mask = masks.float().clone()
        for row, cut in enumerate(prefix_len.tolist()):
            answer_mask[row, :cut] = 0.0
        return answer_mask[:, :, None]

    def _transition_log_prob(
        self,
        x_tm1: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor,
        log_variance: torch.Tensor,
        answer_mask: torch.Tensor,
    ) -> torch.Tensor:
        # p_sample uses 0.5 * randn_like, so the effective posterior variance is scaled by 0.25.
        effective_variance = (variance * 0.25).clamp(min=1e-10)
        effective_log_variance = torch.log(effective_variance)
        log_prob = -0.5 * (
            ((x_tm1 - mean) ** 2) / effective_variance
            + effective_log_variance
            + np.log(2 * np.pi)
        )
        log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=100.0, neginf=-100.0)
        return (log_prob * answer_mask).sum(dim=(1, 2))

    @torch.no_grad()
    def sample_with_log_probs(
        self,
        states: torch.Tensor,
        returns: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]], torch.Tensor]:
        batch_size, horizon, state_dim = states.shape
        prefix_len = self._prefix_len(masks)
        answer_mask = self._answer_mask(masks, prefix_len)
        x = 0.5 * torch.randn(batch_size, horizon, state_dim, device=self.device)
        x = self._apply_prefix(x, states, prefix_len)
        x = self._sanitize_sample(x)

        total_log_prob = torch.zeros(batch_size, device=self.device)
        steps_data: List[Dict[str, torch.Tensor]] = []

        for step in range(self.diffuser.n_timesteps - 1, -1, -1):
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            mean, variance, log_variance = self.diffuser.p_mean_variance(
                x=x,
                cond=states,
                t=t,
                returns=returns,
            )
            noise = 0.5 * torch.randn_like(x)
            nonzero_mask = (1 - (t == 0).float()).reshape(batch_size, 1, 1)
            x_next = mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise
            x_next = self._apply_prefix(x_next, states, prefix_len)
            x_next = self._sanitize_sample(x_next)

            if step > 0:
                total_log_prob += self._transition_log_prob(
                    x_tm1=x_next,
                    mean=mean,
                    variance=variance,
                    log_variance=log_variance,
                    answer_mask=answer_mask,
                )

            steps_data.append(
                {
                    "x_t": x.detach(),
                    "x_tm1": x_next.detach(),
                    "t": t.detach(),
                    "answer_mask": answer_mask.detach(),
                }
            )
            x = x_next

        return x, total_log_prob, steps_data, prefix_len

    def compute_log_prob_from_steps(
        self,
        steps_data: List[Dict[str, torch.Tensor]],
        states: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = states.shape[0]
        log_probs = torch.zeros(batch_size, device=self.device)
        for step_data in steps_data:
            t = step_data["t"]
            if bool((t == 0).all()):
                continue
            mean, variance, log_variance = self.diffuser.p_mean_variance(
                x=step_data["x_t"],
                cond=states,
                t=t,
                returns=returns,
            )
            mean = self._sanitize_sample(mean)
            log_probs = log_probs + self._transition_log_prob(
                x_tm1=step_data["x_tm1"],
                mean=mean,
                variance=variance,
                log_variance=log_variance,
                answer_mask=step_data["answer_mask"],
            )
        return log_probs

    @torch.no_grad()
    def compute_rewards(self, trajectories: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        rewards = self.reward_model(trajectories, masks)
        rewards = torch.nan_to_num(rewards.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
        return rewards

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        advantages = rewards - rewards.mean()
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        if advantages.numel() > 1:
            std = advantages.std(unbiased=False)
            if torch.isfinite(std) and std > 1e-8:
                advantages = advantages / (std + 1e-8)
        return advantages

    def compute_kl_penalty(
        self,
        trajectories: torch.Tensor,
        returns: torch.Tensor,
        masks: torch.Tensor,
        prefix_len: torch.Tensor,
    ) -> torch.Tensor:
        if self.kl_coef <= 0:
            return torch.tensor(0.0, device=self.device)
        batch_size = trajectories.shape[0]
        t = torch.randint(0, self.diffuser.n_timesteps, (batch_size,), device=self.device).long()
        noisy = self.diffuser.q_sample(trajectories.detach(), t)
        noisy = self._apply_prefix(noisy, trajectories.detach(), prefix_len)
        noisy = self._sanitize_sample(noisy)
        pred_current = self.policy_model(noisy, t, returns)
        with torch.no_grad():
            pred_ref = self.ref_model(noisy, t, returns)
        pred_current = torch.nan_to_num(pred_current, nan=0.0, posinf=0.0, neginf=0.0)
        pred_ref = torch.nan_to_num(pred_ref, nan=0.0, posinf=0.0, neginf=0.0)
        return F.mse_loss(pred_current * masks[:, :, None].float(), pred_ref * masks[:, :, None].float())

    def train_step(
        self,
        states: torch.Tensor,
        returns: torch.Tensor,
        masks: torch.Tensor,
    ) -> Dict[str, float]:
        self.policy_model.train()
        self.reward_model.eval()
        self.diffuser.inv_model.eval()

        with torch.no_grad():
            trajectories, old_log_probs, steps_data, prefix_len = self.sample_with_log_probs(
                states=states,
                returns=returns,
                masks=masks,
            )
            rewards = self.compute_rewards(trajectories, masks)
            advantages = self.compute_advantages(rewards)

        if not (
            torch.isfinite(trajectories).all()
            and torch.isfinite(old_log_probs).all()
            and torch.isfinite(rewards).all()
            and torch.isfinite(advantages).all()
        ):
            logger.warning(
                "Skipping PPO batch because sampled tensors are non-finite: traj=%s old_log_prob=%s reward=%s adv=%s",
                torch.isfinite(trajectories).all().item(),
                torch.isfinite(old_log_probs).all().item(),
                torch.isfinite(rewards).all().item(),
                torch.isfinite(advantages).all().item(),
            )
            return {
                "policy_loss": 0.0,
                "kl_penalty": 0.0,
                "total_loss": 0.0,
                "reward_mean": 0.0,
                "reward_std": 0.0,
                "advantage_mean": 0.0,
                "advantage_std": 0.0,
                "ratio_mean": 1.0,
                "ratio_std": 0.0,
                "clip_fraction": 0.0,
                "condition_steps_mean": prefix_len.float().mean().item(),
                "skipped_update": 1.0,
            }

        total_policy_loss = 0.0
        total_kl_penalty = 0.0
        ratio_means = []
        ratio_stds = []
        clip_fraction = 0.0
        valid_updates = 0
        skipped_updates = 0

        for _ in range(self.ppo_update_iters):
            new_log_probs = self.compute_log_prob_from_steps(steps_data, states, returns)
            log_ratio = torch.clamp(new_log_probs - old_log_probs.detach(), min=-20.0, max=20.0)
            ratio = torch.exp(log_ratio)

            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2).mean()
            kl_penalty = self.compute_kl_penalty(trajectories, returns, masks, prefix_len)
            loss = policy_loss + self.kl_coef * kl_penalty
            if not torch.isfinite(loss):
                logger.warning(
                    "Skipping PPO update because loss is non-finite: policy=%s kl=%s total=%s",
                    policy_loss.detach().item() if bool(torch.isfinite(policy_loss).item()) else policy_loss,
                    kl_penalty.detach().item() if bool(torch.isfinite(kl_penalty).item()) else kl_penalty,
                    loss,
                )
                skipped_updates += 1
                continue

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            if not torch.isfinite(grad_norm):
                logger.warning("Skipping PPO update because grad_norm is non-finite: %s", grad_norm)
                self.optimizer.zero_grad(set_to_none=True)
                skipped_updates += 1
                continue
            self.optimizer.step()
            valid_updates += 1

            total_policy_loss += policy_loss.item()
            total_kl_penalty += kl_penalty.item()
            ratio_means.append(ratio.detach().mean().item())
            ratio_stds.append(ratio.detach().std(unbiased=False).item())
            clip_fraction = (
                ((ratio.detach() < 1.0 - self.clip_param) | (ratio.detach() > 1.0 + self.clip_param))
                .float()
                .mean()
                .item()
            )

        update_denominator = max(valid_updates, 1)
        mean_policy_loss = total_policy_loss / update_denominator
        mean_kl_penalty = total_kl_penalty / update_denominator
        return {
            "policy_loss": mean_policy_loss,
            "kl_penalty": mean_kl_penalty,
            "total_loss": mean_policy_loss + self.kl_coef * mean_kl_penalty,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std(unbiased=False).item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std(unbiased=False).item(),
            "ratio_mean": float(np.mean(ratio_means)) if ratio_means else 1.0,
            "ratio_std": float(np.mean(ratio_stds)) if ratio_stds else 0.0,
            "clip_fraction": clip_fraction,
            "condition_steps_mean": prefix_len.float().mean().item(),
            "skipped_update": skipped_updates / max(self.ppo_update_iters, 1),
        }


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        if device_arg.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU")
            return torch.device("cpu")
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_cbd_model(
    cbd_path: str,
    n_timesteps: int,
    device: torch.device,
    cond_obs_training: bool = True,
) -> DFUSER:
    logger.info("Loading CBD checkpoint from %s", cbd_path)
    logger.info("cond_obs_training: %s", cond_obs_training)
    model = DFUSER(
        dim_obs=16,
        n_timesteps=n_timesteps,
        model_choice="Unet",
        attn_block="vanilla",
        predict_epsilon=False,
        cond_obs_training=cond_obs_training,
        pred_one_step=False,
        traj_add_a=False,
    )
    model.load_net(cbd_path, device=str(device))
    model = model.to(device)
    model.use_cuda = device.type == "cuda"
    return model


def build_dataloader(
    step_len: int,
    batch_size: int,
    num_workers: int,
    train_data_path: str,
    return_transform: str,
    return_clip_quantile: float,
    top_return_quantile: float,
    val_ratio: float = 0.05,
    seed: int = 2026,
) -> Tuple[DataLoader, DataLoader]:
    dataset = aigb_dataset(
        step_len,
        load_preprocessed_tain_data=True,
        sparse_data=False,
        simplify_state=False,
        rtg_preference="score",
        train_data_path=train_data_path,
        return_transform=return_transform,
        return_clip_quantile=return_clip_quantile,
    )
    if top_return_quantile > 0:
        threshold = float(np.quantile(dataset.episode_returns, top_return_quantile))
        kept_indices = np.where(dataset.episode_returns >= threshold)[0].tolist()
        logger.info(
            "Filtering dataset by top_return_quantile=%.2f, threshold=%.4f, kept=%s/%s",
            top_return_quantile,
            threshold,
            len(kept_indices),
            len(dataset),
        )
        dataset = Subset(dataset, kept_indices)
    if val_ratio > 0 and len(dataset) > 1:
        val_size = max(1, int(len(dataset) * val_ratio))
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset = dataset
        val_dataset = dataset
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available() and num_workers > 0),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available() and num_workers > 0),
    )
    logger.info(
        "Dataset size: %s, train=%s batches=%s, val=%s batches=%s",
        len(dataset),
        len(train_dataset),
        len(dataloader),
        len(val_dataset),
        len(val_dataloader),
    )
    return dataloader, val_dataloader


def binary_auc_score(targets: np.ndarray, scores: np.ndarray) -> float:
    targets = np.asarray(targets).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    finite_mask = np.isfinite(targets) & np.isfinite(scores)
    targets = targets[finite_mask]
    scores = scores[finite_mask]
    if targets.size == 0:
        return 0.5
    threshold = np.median(targets)
    labels = (targets > threshold).astype(np.int64)
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    pos_rank_sum = ranks[labels == 1].sum()
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


@torch.no_grad()
def evaluate_reward_model(
    reward_model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    reward_model.eval()
    losses = BatchMeanMeter()
    preds = []
    targets = []
    for batch_index, (states, actions, returns, masks) in enumerate(dataloader, start=1):
        states = states.to(device)
        actions = actions.to(device)
        returns = returns.to(device)
        masks = masks.to(device)
        try:
            pred_returns = reward_model(states, masks, actions=actions)
        except TypeError:
            pred_returns = reward_model(states, masks)
        loss = torch.nn.functional.mse_loss(pred_returns, returns)
        if torch.isfinite(loss):
            losses.update(loss.item(), states.shape[0])
            preds.append(pred_returns.detach().cpu().numpy())
            targets.append(returns.detach().cpu().numpy())
        if max_batches is not None and batch_index >= max_batches:
            break
    if losses.count == 0:
        return {"loss": float("nan"), "mse": float("nan"), "auc": 0.5}
    pred_array = np.concatenate(preds, axis=0).reshape(-1)
    target_array = np.concatenate(targets, axis=0).reshape(-1)
    reward_model.train()
    return {
        "loss": losses.mean,
        "mse": float(np.mean((pred_array - target_array) ** 2)),
        "auc": binary_auc_score(target_array, pred_array),
    }


def pretrain_reward_model(
    reward_model: TrajectoryRewardModel,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    max_batches: Optional[int],
    eval_max_batches: Optional[int],
    writer=None,
) -> Tuple[TrajectoryRewardModel, torch.optim.Optimizer, List[Dict[str, float]]]:
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)
    history: List[Dict[str, float]] = []

    reward_model.train()
    for epoch in range(1, epochs + 1):
        losses = BatchMeanMeter()
        preds = []
        targets = []
        skipped_batches = 0
        pbar = tqdm.tqdm(dataloader, desc=f"RM Pretrain {epoch}", leave=False)
        for batch_index, (states, actions, returns, masks) in enumerate(pbar, start=1):
            states = states.to(device)
            actions = actions.to(device)
            returns = returns.to(device)
            masks = masks.to(device)

            try:
                pred_returns = reward_model(states, masks, actions=actions)
            except TypeError:
                pred_returns = reward_model(states, masks)
            loss = torch.nn.functional.mse_loss(pred_returns, returns)
            if not torch.isfinite(loss):
                skipped_batches += 1
                logger.warning(
                    "Skipping non-finite RM batch %s: loss=%s states_finite=%s returns_finite=%s pred_finite=%s",
                    batch_index,
                    loss.item(),
                    torch.isfinite(states).all().item(),
                    torch.isfinite(returns).all().item(),
                    torch.isfinite(pred_returns).all().item(),
                )
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
            optimizer.step()

            losses.update(loss.item(), states.shape[0])
            preds.append(pred_returns.detach().cpu().numpy())
            targets.append(returns.detach().cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.5f}")

            if max_batches is not None and batch_index >= max_batches:
                break

        epoch_loss = losses.mean
        train_pred = np.concatenate(preds, axis=0).reshape(-1) if preds else np.asarray([])
        train_target = np.concatenate(targets, axis=0).reshape(-1) if targets else np.asarray([])
        train_auc = binary_auc_score(train_target, train_pred) if preds else 0.5
        val_metrics = evaluate_reward_model(reward_model, val_dataloader, device, max_batches=eval_max_batches)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_mse": float(np.mean((train_pred - train_target) ** 2)) if preds else float("nan"),
            "train_auc": train_auc,
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_auc": val_metrics["auc"],
            "skipped_batches": skipped_batches,
        }
        history.append(epoch_metrics)
        for key in ["train_loss", "train_mse", "train_auc", "val_loss", "val_mse", "val_auc", "skipped_batches"]:
            add_scalar(writer, f"rm_pretrain/{key}", epoch_metrics[key], epoch)
        logger.info(
            "RM epoch %s: train_loss=%.6f train_auc=%.6f val_loss=%.6f val_auc=%.6f skipped_batches=%s",
            epoch,
            epoch_metrics["train_loss"],
            epoch_metrics["train_auc"],
            epoch_metrics["val_loss"],
            epoch_metrics["val_auc"],
            skipped_batches,
        )

    return reward_model, optimizer, history


@torch.no_grad()
def evaluate_generated_reward(
    cbd_model: DFUSER,
    reward_model: TrajectoryRewardModel,
    states: torch.Tensor,
    returns: torch.Tensor,
    masks: torch.Tensor,
    condition_steps: int,
) -> Tuple[float, float]:
    cbd_model.eval()
    reward_model.eval()
    prefix_len = min(max(condition_steps, 1), states.shape[1])
    generated = cbd_model.diffuser.conditional_sample(cond=states[:, :prefix_len, :], returns=returns, horizon=cbd_model.step_len)
    generated = torch.nan_to_num(generated, nan=0.0, posinf=0.0, neginf=0.0)
    predicted = reward_model(generated, masks)
    predicted = torch.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)
    return predicted.mean().item(), predicted.std().item()


def train_cbd_ppo(
    cbd_path: str = "saved_model/CBDtest/diffuser.pt",
    save_path: str = "saved_model/DDPO-CBD",
    train_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    reward_model_lr: float = 1e-4,
    n_timesteps: int = 10,
    pretrain_rm_epochs: int = 1,
    save_freq: int = 5,
    device: str = "auto",
    num_workers: int = 0,
    max_rm_batches: Optional[int] = None,
    max_policy_batches: Optional[int] = None,
    train_data_path: str = "data/trajectory/trajectory_data.csv",
    return_transform: str = "log1p",
    return_clip_quantile: float = 0.99,
    top_return_quantile: float = 0.0,
    freeze_rm: bool = True,
    reward_eval_freq: int = 10,
    ppo_update_iters: int = 4,
    clip_param: float = 0.2,
    kl_coef: float = 0.1,
    max_grad_norm: float = 1.0,
    condition_steps: int = 24,
    cond_obs_training: bool = True,
    rm_model: str = "unet",
    rm_hidden_dim: int = 128,
    rm_val_ratio: float = 0.05,
    rm_eval_batches: Optional[int] = None,
    tb_log_dir: str = "logs/tensorboard/cbd_ppo",
    sample_value_clip: float = 1e6,
):
    device_obj = get_device(device)
    os.makedirs(save_path, exist_ok=True)
    writer = create_summary_writer(tb_log_dir)
    logger.info("Using device: %s", device_obj)

    cbd_model = load_cbd_model(
        cbd_path,
        n_timesteps=n_timesteps,
        device=device_obj,
        cond_obs_training=cond_obs_training,
    )
    dataloader, val_dataloader = build_dataloader(
        cbd_model.step_len,
        batch_size=batch_size,
        num_workers=num_workers,
        train_data_path=train_data_path,
        return_transform=return_transform,
        return_clip_quantile=return_clip_quantile,
        top_return_quantile=top_return_quantile,
        val_ratio=rm_val_ratio,
    )
    if rm_model == "unet":
        reward_model = UNetTrajectoryRewardModel(
            state_dim=16,
            horizon=cbd_model.step_len,
            hidden_dim=rm_hidden_dim,
            emb_dim=rm_hidden_dim,
            traj_add_a=False,
        ).to(device_obj)
    elif rm_model == "transformer":
        reward_model = TrajectoryRewardModel(state_dim=16, hidden_dim=rm_hidden_dim).to(device_obj)
    else:
        raise ValueError(f"Unsupported rm_model: {rm_model}")
    logger.info("Reward model: %s, parameters=%s", rm_model, sum(p.numel() for p in reward_model.parameters()))

    logger.info("Pretraining reward model...")
    reward_model, rm_optimizer, rm_pretrain_history = pretrain_reward_model(
        reward_model=reward_model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        device=device_obj,
        epochs=pretrain_rm_epochs,
        lr=reward_model_lr,
        max_batches=max_rm_batches,
        eval_max_batches=rm_eval_batches,
        writer=writer,
    )
    if not freeze_rm:
        logger.warning("DDPO mode freezes the reward model after pretraining to match the source project logic.")
        freeze_rm = True

    training_history = {
        "epoch": [],
        "policy_loss": [],
        "kl_penalty": [],
        "total_loss": [],
        "rm_loss": [],
        "reward_mean": [],
        "reward_std": [],
        "advantage_mean": [],
        "advantage_std": [],
        "ratio_mean": [],
        "ratio_std": [],
        "clip_fraction": [],
        "condition_steps_mean": [],
        "skipped_update": [],
        "diffuser_param_delta": [],
        "inv_param_delta": [],
        "rm_pretrain_loss": [item["train_loss"] for item in rm_pretrain_history],
        "rm_pretrain_metrics": rm_pretrain_history,
    }

    best_reward = float("-inf")
    trainer = CBDDDPOTrainer(
        cbd_model=cbd_model,
        reward_model=reward_model,
        device=device_obj,
        lr=lr,
        ppo_update_iters=ppo_update_iters,
        clip_param=clip_param,
        kl_coef=kl_coef,
        max_grad_norm=max_grad_norm,
        condition_steps=condition_steps,
        sample_value_clip=sample_value_clip,
    )
    initial_policy_params = {
        name: param.detach().cpu().clone() for name, param in cbd_model.diffuser.model.named_parameters()
    }
    initial_inv_params = {
        name: param.detach().cpu().clone() for name, param in cbd_model.diffuser.inv_model.named_parameters()
    }
    logger.info(
        "Starting DDPO/PPO fine-tuning: ppo_update_iters=%s clip=%.3f kl=%.3f condition_steps=%s",
        ppo_update_iters,
        clip_param,
        kl_coef,
        condition_steps,
    )

    for epoch in range(1, train_epochs + 1):
        cbd_model.train()
        if freeze_rm:
            reward_model.eval()
        else:
            reward_model.train()

        epoch_rm = BatchMeanMeter()
        epoch_reward_mean = BatchMeanMeter()
        epoch_reward_std = BatchMeanMeter()
        epoch_stats = {
            key: BatchMeanMeter()
            for key in [
                "policy_loss",
                "kl_penalty",
                "total_loss",
                "advantage_mean",
                "advantage_std",
                "ratio_mean",
                "ratio_std",
                "clip_fraction",
                "condition_steps_mean",
                "skipped_update",
            ]
        }
        latest_reward_mean = 0.0
        latest_reward_std = 0.0

        pbar = tqdm.tqdm(dataloader, desc=f"CBD-PPO Epoch {epoch}")
        for batch_index, (states, actions, returns, masks) in enumerate(pbar, start=1):
            states = states.to(device_obj)
            actions = actions.to(device_obj)
            returns = returns.to(device_obj)
            masks = masks.to(device_obj)
            current_batch_size = states.shape[0]
            if not (torch.isfinite(states).all() and torch.isfinite(returns).all()):
                logger.warning(
                    "Skipping CBD-PPO batch %s because input tensors are non-finite: states=%s returns=%s",
                    batch_index,
                    torch.isfinite(states).all().item(),
                    torch.isfinite(returns).all().item(),
                )
                continue

            stats = trainer.train_step(states=states, returns=returns, masks=masks)
            for key, meter in epoch_stats.items():
                meter.update(stats[key], current_batch_size)

            if freeze_rm:
                with torch.no_grad():
                    pred_returns = reward_model(states, masks)
                    rm_loss = torch.nn.functional.mse_loss(pred_returns, returns)
            else:
                pred_returns = reward_model(states, masks)
                rm_loss = torch.nn.functional.mse_loss(pred_returns, returns)
                rm_optimizer.zero_grad()
                rm_loss.backward()
                rm_optimizer.step()

            if batch_index == 1 or (reward_eval_freq > 0 and batch_index % reward_eval_freq == 0):
                reward_mean, reward_std = evaluate_generated_reward(
                    cbd_model=cbd_model,
                    reward_model=reward_model,
                    states=states,
                    returns=returns,
                    masks=masks,
                    condition_steps=condition_steps,
                )
                latest_reward_mean = reward_mean
                latest_reward_std = reward_std
            else:
                reward_mean = latest_reward_mean
                reward_std = latest_reward_std

            epoch_rm.update(rm_loss.item(), current_batch_size)
            epoch_reward_mean.update(reward_mean, current_batch_size)
            epoch_reward_std.update(reward_std, current_batch_size)

            pbar.set_postfix(
                ppo=f"{stats['policy_loss']:.4f}",
                kl=f"{stats['kl_penalty']:.4f}",
                reward=f"{reward_mean:.4f}",
                skip=f"{stats['skipped_update']:.0f}",
            )

            if max_policy_batches is not None and batch_index >= max_policy_batches:
                break

        mean_rm = epoch_rm.mean
        mean_reward = epoch_reward_mean.mean
        mean_reward_std = epoch_reward_std.mean
        mean_stats = {key: meter.mean for key, meter in epoch_stats.items()}
        policy_delta = _l2_delta(initial_policy_params, cbd_model.diffuser.model)
        inv_delta = _l2_delta(initial_inv_params, cbd_model.diffuser.inv_model)

        training_history["epoch"].append(epoch)
        for key, value in mean_stats.items():
            training_history[key].append(value)
        training_history["rm_loss"].append(mean_rm)
        training_history["reward_mean"].append(mean_reward)
        training_history["reward_std"].append(mean_reward_std)
        training_history["diffuser_param_delta"].append(policy_delta)
        training_history["inv_param_delta"].append(inv_delta)
        for key, value in mean_stats.items():
            add_scalar(writer, f"cbd_ppo/{key}", value, epoch)
        add_scalar(writer, "cbd_ppo/rm_loss", mean_rm, epoch)
        add_scalar(writer, "cbd_ppo/reward_mean", mean_reward, epoch)
        add_scalar(writer, "cbd_ppo/reward_std", mean_reward_std, epoch)
        add_scalar(writer, "cbd_ppo/diffuser_param_delta", policy_delta, epoch)
        add_scalar(writer, "cbd_ppo/inv_param_delta", inv_delta, epoch)

        logger.info(
            "Epoch %s: ppo=%.6f kl=%.6f rm=%.6f reward=%.6f±%.6f ratio=%.4f clip=%.4f policy_delta=%.6f inv_delta=%.6f",
            epoch,
            mean_stats["policy_loss"],
            mean_stats["kl_penalty"],
            mean_rm,
            mean_reward,
            mean_reward_std,
            mean_stats["ratio_mean"],
            mean_stats["clip_fraction"],
            policy_delta,
            inv_delta,
        )

        if mean_reward > best_reward:
            best_reward = mean_reward
            cbd_model.save_net(save_path, save_name="_best")
            torch.save(
                {
                    "reward_model": reward_model.state_dict(),
                    "epoch": epoch,
                    "reward_mean": mean_reward,
                },
                os.path.join(save_path, "reward_model_best.pt"),
            )
            logger.info("Saved best CBD-PPO checkpoint with reward %.6f", best_reward)

        if epoch % save_freq == 0:
            cbd_model.save_net(save_path, save_name=f"_epoch_{epoch}")
            torch.save(
                {
                    "reward_model": reward_model.state_dict(),
                    "epoch": epoch,
                    "reward_mean": mean_reward,
                },
                os.path.join(save_path, f"reward_model_epoch_{epoch}.pt"),
            )

    cbd_model.save_net(save_path, save_name="")
    torch.save(reward_model.state_dict(), os.path.join(save_path, "reward_model.pt"))

    history_path = os.path.join(save_path, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    logger.info("Training completed. Final model: %s", os.path.join(save_path, "diffuser.pt"))
    logger.info("Best generated reward: %.6f", best_reward)
    if writer is not None:
        writer.close()
    return cbd_model, reward_model, training_history


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CBD with a CBD-PPO/DDPO-style offline loop")
    parser.add_argument("--cbd_path", type=str, default="saved_model/CBDtest/diffuser.pt")
    parser.add_argument("--save_path", type=str, default="saved_model/DDPO-CBD")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--reward_model_lr", type=float, default=1e-4)
    parser.add_argument("--n_timesteps", type=int, default=10)
    parser.add_argument("--pretrain_rm_epochs", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_rm_batches", type=int, default=None)
    parser.add_argument("--max_policy_batches", type=int, default=None)
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="Backward-compatible alias for --max_policy_batches.",
    )
    parser.add_argument("--train_data_path", type=str, default="data/trajectory/trajectory_data.csv")
    parser.add_argument("--return_transform", type=str, default="log1p", choices=["log1p", "linear", "sigmoid"])
    parser.add_argument("--return_clip_quantile", type=float, default=0.99)
    parser.add_argument("--top_return_quantile", type=float, default=0.0)
    parser.add_argument("--freeze_rm", action="store_true", default=False)
    parser.add_argument("--update_rm", action="store_true", default=False, help="Deprecated in DDPO mode; RM is frozen after pretraining.")
    parser.add_argument("--reward_eval_freq", type=int, default=10)
    parser.add_argument("--ppo_update_iters", type=int, default=4)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--condition_steps", type=int, default=24)
    parser.add_argument("--rm_model", type=str, default="unet", choices=["unet", "transformer"])
    parser.add_argument("--rm_hidden_dim", type=int, default=128)
    parser.add_argument("--rm_val_ratio", type=float, default=0.05)
    parser.add_argument("--rm_eval_batches", type=int, default=None)
    parser.add_argument("--tb_log_dir", type=str, default="logs/tensorboard/cbd_ppo")
    parser.add_argument("--sample_value_clip", type=float, default=1e6)
    parser.add_argument(
        "--cond_obs_training",
        type=parse_bool_arg,
        nargs="?",
        const=True,
        default=True,
        help="true for CBD checkpoints, false only when intentionally fine-tuning a DD/basic diffusion checkpoint",
    )
    args = parser.parse_args()

    freeze_rm = True
    if args.update_rm:
        logger.warning("--update_rm is ignored: source-aligned DDPO freezes RM after pretraining.")
    if args.max_train_batches is not None:
        args.max_policy_batches = args.max_train_batches

    train_cbd_ppo(
        cbd_path=args.cbd_path,
        save_path=args.save_path,
        train_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        reward_model_lr=args.reward_model_lr,
        n_timesteps=args.n_timesteps,
        pretrain_rm_epochs=args.pretrain_rm_epochs,
        save_freq=args.save_freq,
        device=args.device,
        num_workers=args.num_workers,
        max_rm_batches=args.max_rm_batches,
        max_policy_batches=args.max_policy_batches,
        train_data_path=args.train_data_path,
        return_transform=args.return_transform,
        return_clip_quantile=args.return_clip_quantile,
        top_return_quantile=args.top_return_quantile,
        freeze_rm=freeze_rm,
        reward_eval_freq=args.reward_eval_freq,
        ppo_update_iters=args.ppo_update_iters,
        clip_param=args.clip_param,
        kl_coef=args.kl_coef,
        max_grad_norm=args.max_grad_norm,
        condition_steps=args.condition_steps,
        cond_obs_training=args.cond_obs_training,
        rm_model=args.rm_model,
        rm_hidden_dim=args.rm_hidden_dim,
        rm_val_ratio=args.rm_val_ratio,
        rm_eval_batches=args.rm_eval_batches,
        tb_log_dir=args.tb_log_dir,
        sample_value_clip=args.sample_value_clip,
    )


if __name__ == "__main__":
    main()
