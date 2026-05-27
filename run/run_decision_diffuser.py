import torch
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidding_train_env.baseline.dd.DFUSER import (DFUSER)
import time
from bidding_train_env.baseline.dd.dataset import aigb_dataset
from torch.utils.data import DataLoader


def run_decision_diffuser(
        save_path="saved_model/DDtest",
        train_epoch=1,
        batch_size=1000):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("train_epoch", train_epoch)
    print("batch-size", batch_size)

    algorithm = DFUSER()
    algorithm = algorithm.to(device)

    args_dict = {'data_version': 'monk_data_small'}
    dataset = aigb_dataset(algorithm.step_len, **args_dict)
    dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, num_workers=2, pin_memory=True)

    # 参数数量
    total_params = sum(p.numel() for p in algorithm.parameters())
    print(f"参数数量：{total_params}")

    # 3. 迭代训练

    epi = 1
    for epoch in range(0, train_epoch):
        for batch_index, (states, actions, returns, masks) in enumerate(dataloader):
            states.to(device)
            actions.to(device)
            returns.to(device)
            masks.to(device)

            start_time = time.time()

            # 训练
            train_result = algorithm.trainStep(states, actions, returns, masks)
            all_loss, (diffuse_loss, inv_loss) = train_result[:2]
            all_loss = all_loss.detach().clone()
            diffuse_loss = diffuse_loss.detach().clone()
            inv_loss = inv_loss.detach().clone()
            end_time = time.time()
            print(
                f"第{epi}个batch训练时间为: {end_time - start_time} s, all_loss: {all_loss}, diffuse_loss: {diffuse_loss}, inv_loss: {inv_loss}")
            epi += 1

    # algorithm.save_model(save_path, epi)
    algorithm.save_net(save_path, epi)


def main():
    parser = argparse.ArgumentParser(description="Train Decision Diffuser for bidding")
    parser.add_argument("--save_path", type=str, default="saved_model/DDtest")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()
    run_decision_diffuser(
        save_path=args.save_path,
        train_epoch=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
