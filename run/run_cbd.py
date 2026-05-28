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
from run.training_log_utils import BatchMeanMeter, add_scalar, create_summary_writer, format_loss_for_log

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
    num_workers=2,
    tb_log_dir="logs/tensorboard/cbd",
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_path, exist_ok=True)

    logger.info(f"Using device: {device}")
    logger.info(f"Train epochs: {train_epoch}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Diffusion steps: {n_timesteps}")
    logger.info(f"Model: {model_choice} (CBD)")
    writer = create_summary_writer(tb_log_dir)

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
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Number of batches: {len(dataloader)}")

    best_loss = float('inf')

    for epoch in tqdm.tqdm(range(1, train_epoch + 1), desc='Training CBD'):
        epoch_loss = BatchMeanMeter()
        epoch_diff = BatchMeanMeter()
        epoch_inv = BatchMeanMeter()

        for states, actions, returns, masks in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            returns = returns.to(device)
            masks = masks.to(device)
            current_batch_size = states.shape[0]

            loss, (diff_loss, inv_loss), _ = algorithm.trainStep(states, actions, returns, masks)

            epoch_loss.update(loss.item(), current_batch_size)
            epoch_diff.update(diff_loss.item(), current_batch_size)
            epoch_inv.update(inv_loss.item() if inv_loss is not None else 0.0, current_batch_size)

        avg_loss = epoch_loss.mean
        avg_diff = epoch_diff.mean
        avg_inv = epoch_inv.mean
        add_scalar(writer, "cbd/loss_mean", avg_loss, epoch)
        add_scalar(writer, "cbd/diff_mean", avg_diff, epoch)
        add_scalar(writer, "cbd/inv_mean", avg_inv, epoch)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %s: loss_mean=%s, diff_mean=%s, inv_mean=%s",
                epoch,
                format_loss_for_log(avg_loss),
                format_loss_for_log(avg_diff),
                format_loss_for_log(avg_inv),
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            algorithm.save_net(save_path, save_name='_best')

        if epoch % save_every == 0:
            algorithm.save_net(save_path, save_name=f'_epoch_{epoch}')

    algorithm.save_net(save_path, save_name='')
    logger.info(f"Training completed. Final model: {save_path}/diffuser.pt")
    logger.info("Best loss_mean: %s", format_loss_for_log(best_loss))
    if writer is not None:
        writer.close()


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
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--tb_log_dir', type=str, default='logs/tensorboard/cbd')
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
        num_workers=args.num_workers,
        tb_log_dir=args.tb_log_dir,
    )


if __name__ == '__main__':
    main()
