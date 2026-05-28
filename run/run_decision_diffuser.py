import torch
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidding_train_env.baseline.dd.DFUSER import (DFUSER)
import time
import numpy as np
from bidding_train_env.baseline.dd.dataset import aigb_dataset
from torch.utils.data import DataLoader
from run.training_log_utils import BatchMeanMeter, add_scalar, create_summary_writer, format_loss_for_log


def run_decision_diffuser(
        save_path="saved_model/DDtest",
        train_epoch=1,
        batch_size=1000,
        num_workers=2,
        tb_log_dir="logs/tensorboard/dd"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("train_epoch", train_epoch)
    print("batch-size", batch_size)

    algorithm = DFUSER()
    algorithm = algorithm.to(device)

    args_dict = {'data_version': 'monk_data_small'}
    dataset = aigb_dataset(algorithm.step_len, **args_dict)
    dataloader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    writer = create_summary_writer(tb_log_dir)

    # 参数数量
    total_params = sum(p.numel() for p in algorithm.parameters())
    print(f"参数数量：{total_params}")

    # 3. 迭代训练

    for epoch in range(1, train_epoch + 1):
        epoch_loss = BatchMeanMeter()
        epoch_diff = BatchMeanMeter()
        epoch_inv = BatchMeanMeter()
        epoch_time = []
        for batch_index, (states, actions, returns, masks) in enumerate(dataloader):
            states = states.to(device)
            actions = actions.to(device)
            returns = returns.to(device)
            masks = masks.to(device)
            current_batch_size = states.shape[0]

            start_time = time.time()

            # 训练
            train_result = algorithm.trainStep(states, actions, returns, masks)
            all_loss, (diffuse_loss, inv_loss) = train_result[:2]
            all_loss = all_loss.detach().clone()
            diffuse_loss = diffuse_loss.detach().clone()
            inv_loss = inv_loss.detach().clone()
            end_time = time.time()
            epoch_loss.update(all_loss.item(), current_batch_size)
            epoch_diff.update(diffuse_loss.item(), current_batch_size)
            epoch_inv.update(inv_loss.item(), current_batch_size)
            epoch_time.append(end_time - start_time)

        print(
            f"Epoch {epoch}: "
            f"loss_mean={format_loss_for_log(epoch_loss.mean)}, "
            f"diff_mean={format_loss_for_log(epoch_diff.mean)}, "
            f"inv_mean={format_loss_for_log(epoch_inv.mean)}, "
            f"batch_time_mean={np.mean(epoch_time):.4f}s"
        )
        add_scalar(writer, "dd/loss_mean", epoch_loss.mean, epoch)
        add_scalar(writer, "dd/diff_mean", epoch_diff.mean, epoch)
        add_scalar(writer, "dd/inv_mean", epoch_inv.mean, epoch)
        add_scalar(writer, "dd/batch_time_mean", np.mean(epoch_time), epoch)

    algorithm.save_net(save_path, save_name="")
    if writer is not None:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Decision Diffuser for bidding")
    parser.add_argument("--save_path", type=str, default="saved_model/DDtest")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--tb_log_dir", type=str, default="logs/tensorboard/dd")
    args = parser.parse_args()
    run_decision_diffuser(
        save_path=args.save_path,
        train_epoch=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tb_log_dir=args.tb_log_dir,
    )


if __name__ == '__main__':
    main()
