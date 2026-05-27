"""
CBD (Causal Bidding Diffusion) Training Script

基于 ICML_DiT_bid/CAD 思路：
- diffusion + inverse dynamics
- U-Net backbone (CBD)
- score/reward RTG 偏好
"""
import os
import sys
import torch
from torch.utils.data import DataLoader
import tqdm
import argparse
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidding_train_env.baseline.dit.DFUSER import DFUSER
from bidding_train_env.baseline.dit.dataset import aigb_dataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_cbd_training(
    save_path="saved_model/CBDtest",
    train_epoch=100,
    batch_size=1000,
    lr=1e-4,
    seed=200,
    n_timesteps=100,
    model_choice='Unet',
    attn_block='vanilla',
    predict_epsilon=False,
    cond_obs_training=True,
    pred_one_step=False,
    traj_add_a=False,
    rtg_preference='score',
    save_every=20,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_path, exist_ok=True)

    logger.info(f"Using device: {device}")
    logger.info(f"Train epochs: {train_epoch}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Diffusion steps: {n_timesteps}")
    logger.info(f"Model: {model_choice} (CBD)")

    algorithm = DFUSER(
        dim_obs=16,
        lr=lr,
        network_random_seed=seed,
        n_timesteps=n_timesteps,
        model_choice=model_choice,
        attn_block=attn_block,
        predict_epsilon=predict_epsilon,
        cond_obs_training=cond_obs_training,
        pred_one_step=pred_one_step,
        traj_add_a=traj_add_a,
    ).to(device)

    total_params = sum(p.numel() for p in algorithm.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    dataset = aigb_dataset(
        algorithm.step_len,
        load_preprocessed_tain_data=True,
        sparse_data=False,
        simplify_state=False,
        rtg_preference=rtg_preference,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Number of batches: {len(dataloader)}")

    best_loss = float('inf')

    for epoch in tqdm.tqdm(range(1, train_epoch + 1), desc='Training CBD'):
        epoch_loss = 0.0
        epoch_diff = 0.0
        epoch_inv = 0.0
        n = 0

        for states, actions, returns, masks in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            returns = returns.to(device)
            masks = masks.to(device)

            loss, (diff_loss, inv_loss), _ = algorithm.trainStep(states, actions, returns, masks)

            epoch_loss += loss.item()
            epoch_diff += diff_loss.item()
            epoch_inv += inv_loss.item() if inv_loss is not None else 0.0
            n += 1

        avg_loss = epoch_loss / max(n, 1)
        avg_diff = epoch_diff / max(n, 1)
        avg_inv = epoch_inv / max(n, 1)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}: loss={avg_loss:.6f}, diff={avg_diff:.6f}, inv={avg_inv:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            algorithm.save_net(save_path, save_name='_best')

        if epoch % save_every == 0:
            algorithm.save_net(save_path, save_name=f'_epoch_{epoch}')

    algorithm.save_net(save_path, save_name='')
    logger.info(f"Training completed. Final model: {save_path}/diffuser.pt")
    logger.info(f"Best loss: {best_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Train CBD model for bidding')
    parser.add_argument('--save_path', type=str, default='saved_model/CBDtest')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--n_timesteps', type=int, default=100)
    parser.add_argument('--model_choice', type=str, default='Unet', choices=['Unet'])
    parser.add_argument('--attn_block', type=str, default='vanilla', choices=['causal', 'vanilla'])
    parser.add_argument('--predict_epsilon', action='store_true', default=False)
    parser.add_argument('--cond_obs_training', action='store_true', default=True)
    parser.add_argument('--pred_one_step', action='store_true', default=False)
    parser.add_argument('--traj_add_a', action='store_true', default=False)
    parser.add_argument('--rtg_preference', type=str, default='score', choices=['reward', 'score'])
    parser.add_argument('--save_every', type=int, default=20)
    args = parser.parse_args()

    run_cbd_training(
        save_path=args.save_path,
        train_epoch=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        n_timesteps=args.n_timesteps,
        model_choice=args.model_choice,
        attn_block=args.attn_block,
        predict_epsilon=args.predict_epsilon,
        cond_obs_training=args.cond_obs_training,
        pred_one_step=args.pred_one_step,
        traj_add_a=args.traj_add_a,
        rtg_preference=args.rtg_preference,
        save_every=args.save_every,
    )


if __name__ == '__main__':
    main()
