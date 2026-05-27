import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidding_train_env.baseline.ddpo.reward_model import TrajectoryRewardModel
from bidding_train_env.baseline.dit.DFUSER import DFUSER
from bidding_train_env.baseline.dit.dataset import aigb_dataset


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        if device_arg.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str, device: torch.device) -> DFUSER:
    model = DFUSER(
        dim_obs=16,
        n_timesteps=10,
        model_choice="Unet",
        attn_block="vanilla",
        predict_epsilon=False,
        cond_obs_training=True,
        pred_one_step=False,
        traj_add_a=False,
    )
    model.load_net(model_path, device=str(device))
    model = model.to(device)
    model.use_cuda = device.type == "cuda"
    model.eval()
    return model


def build_splits(
    train_data_path: str,
    batch_size: int,
    seed: int,
    return_transform: str,
    return_clip_quantile: float,
) -> Tuple[DataLoader, DataLoader, int]:
    dataset = aigb_dataset(
        step_len=48,
        load_preprocessed_tain_data=True,
        sparse_data=False,
        simplify_state=False,
        rtg_preference="score",
        train_data_path=train_data_path,
        return_transform=return_transform,
        return_clip_quantile=return_clip_quantile,
    )
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(num_samples * 0.8)
    train_idx = indices[:split].tolist()
    eval_idx = indices[split:].tolist()
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(Subset(dataset, eval_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, eval_loader, num_samples


def train_reward_model(
    reward_model: TrajectoryRewardModel,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    max_train_batches: int,
) -> List[float]:
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)
    history: List[float] = []
    reward_model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for batch_index, (states, _, returns, masks) in enumerate(
            tqdm.tqdm(train_loader, desc=f"RM Train {epoch}", leave=False),
            start=1,
        ):
            states = states.to(device)
            returns = returns.to(device)
            masks = masks.to(device)
            pred_returns = reward_model(states, masks)
            loss = torch.nn.functional.mse_loss(pred_returns, returns)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if max_train_batches > 0 and batch_index >= max_train_batches:
                break
        history.append(float(np.mean(losses)) if losses else 0.0)
    return history


@torch.no_grad()
def evaluate_model_scores(
    model: DFUSER,
    reward_model: TrajectoryRewardModel,
    eval_loader: DataLoader,
    device: torch.device,
    max_eval_batches: int,
) -> Dict[str, float]:
    reward_model.eval()
    scores: List[float] = []
    oracle_scores: List[float] = []
    for batch_index, (states, _, returns, masks) in enumerate(
        tqdm.tqdm(eval_loader, desc="RM Eval", leave=False),
        start=1,
    ):
        states = states.to(device)
        returns = returns.to(device)
        masks = masks.to(device)
        generated = model.diffuser.conditional_sample(cond=states, returns=returns, horizon=model.step_len)
        pred_generated = reward_model(generated, masks)
        pred_oracle = reward_model(states, masks)
        scores.extend(pred_generated.squeeze(-1).detach().cpu().tolist())
        oracle_scores.extend(pred_oracle.squeeze(-1).detach().cpu().tolist())
        if max_eval_batches > 0 and batch_index >= max_eval_batches:
            break
    return {
        "rm_score_mean": float(np.mean(scores)) if scores else 0.0,
        "rm_score_std": float(np.std(scores)) if scores else 0.0,
        "oracle_rm_score_mean": float(np.mean(oracle_scores)) if oracle_scores else 0.0,
    }


def build_markdown(result_rows: List[Dict[str, float]]) -> str:
    lines = [
        "# Public Trajectory RM Alignment",
        "",
        "| Model | RM Score | RM Std | Relative Lift vs CBD |",
        "| --- | ---: | ---: | ---: |",
    ]
    cbd_score = next((row["rm_score_mean"] for row in result_rows if row["model"] == "CBD"), None)
    for row in result_rows:
        lift = 0.0 if cbd_score in (None, 0) else (row["rm_score_mean"] - cbd_score) / cbd_score * 100.0
        lift_str = "--" if row["model"] == "CBD" else f"{lift:+.2f}%"
        lines.append(
            f"| {row['model']} | {row['rm_score_mean']:.4f} | {row['rm_score_std']:.4f} | {lift_str} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate public trajectory RM alignment for CBD and CBD-PPO.")
    parser.add_argument("--train_data_path", type=str, default="data/trajectory/trajectory_data.csv")
    parser.add_argument("--cbd_path", type=str, default="saved_model/CBDtest/diffuser_best.pt")
    parser.add_argument("--cbd_ppo_path", type=str, default="saved_model/DDPO-CBD-test/diffuser_best.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--rm_epochs", type=int, default=2)
    parser.add_argument("--rm_lr", type=float, default=1e-4)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="./results/public_rm_alignment")
    parser.add_argument("--return_transform", type=str, default="log1p", choices=["log1p", "linear", "sigmoid"])
    parser.add_argument("--return_clip_quantile", type=float, default=0.99)
    args = parser.parse_args()

    device = get_device(args.device)
    train_loader, eval_loader, num_samples = build_splits(
        args.train_data_path,
        args.batch_size,
        args.seed,
        args.return_transform,
        args.return_clip_quantile,
    )

    reward_model = TrajectoryRewardModel(state_dim=16, hidden_dim=128).to(device)
    rm_history = train_reward_model(
        reward_model=reward_model,
        train_loader=train_loader,
        device=device,
        epochs=args.rm_epochs,
        lr=args.rm_lr,
        max_train_batches=args.max_train_batches,
    )

    models = {
        "CBD": load_model(args.cbd_path, device),
        "CBD-PPO": load_model(args.cbd_ppo_path, device),
    }

    rows: List[Dict[str, float]] = []
    for model_name, model in models.items():
        summary = evaluate_model_scores(
            model=model,
            reward_model=reward_model,
            eval_loader=eval_loader,
            device=device,
            max_eval_batches=args.max_eval_batches,
        )
        summary["model"] = model_name
        rows.append(summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": timestamp,
        "num_samples": num_samples,
        "rm_history": rm_history,
        "rows": rows,
    }
    json_path = output_dir / f"public_rm_alignment_{timestamp}.json"
    md_path = output_dir / f"public_rm_alignment_{timestamp}.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(build_markdown(rows))

    print(f"[INFO] Saved JSON results to {json_path}")
    print(f"[INFO] Saved Markdown report to {md_path}")
    print(build_markdown(rows))


if __name__ == "__main__":
    main()
