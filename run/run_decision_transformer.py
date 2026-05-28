import numpy as np
import argparse
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.dt.utils import EpisodeReplayBuffer
from bidding_train_env.baseline.dt.dt import DecisionTransformer
from torch.utils.data import DataLoader
import logging
import pickle
from run.training_log_utils import BatchMeanMeter, add_scalar, create_summary_writer, format_loss_for_log

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_dt(epochs=100, batch_size=32, save_path="saved_model/DTtest", num_workers=0, tb_log_dir="logs/tensorboard/dt"):
    train_model(
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
        num_workers=num_workers,
        tb_log_dir=tb_log_dir,
    )


def train_model(epochs=100, batch_size=32, save_path="saved_model/DTtest", num_workers=0, tb_log_dir="logs/tensorboard/dt"):
    state_dim = 16

    replay_buffer = EpisodeReplayBuffer(16, 1, "./data/trajectory/trajectory_data.csv")
    save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
                        save_path)
    logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")
    logger.info(f"Train samples per epoch: {len(replay_buffer)}")

    model = DecisionTransformer(state_dim=state_dim, act_dim=1, state_mean=replay_buffer.state_mean,
                                state_std=replay_buffer.state_std)
    dataloader = DataLoader(
        replay_buffer,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    writer = create_summary_writer(tb_log_dir)

    model.train()
    global_step = 0
    for epoch in range(1, epochs + 1):
        epoch_loss = BatchMeanMeter()
        for states, actions, rewards, dones, rtg, timesteps, attention_mask in dataloader:
            train_loss = model.step(states, actions, rewards, dones, rtg, timesteps, attention_mask)
            current_batch_size = states.shape[0]
            epoch_loss.update(train_loss, current_batch_size)
            global_step += 1
            add_scalar(writer, "dt/action_loss_step", train_loss, global_step)
            model.scheduler.step()

        logger.info("Epoch %s: action_loss_mean=%s", epoch, format_loss_for_log(epoch_loss.mean))
        add_scalar(writer, "dt/action_loss_epoch", epoch_loss.mean, epoch)

    model.save_net(save_path)
    test_state = np.ones(state_dim, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")
    if writer is not None:
        writer.close()


def load_model():
    """
    加载模型。
    """
    with open('./Model/DT/saved_model/normalize_dict.pkl', 'rb') as f:
        normalize_dict = pickle.load(f)
    model = DecisionTransformer(state_dim=16, act_dim=1, state_mean=normalize_dict["state_mean"],
                                state_std=normalize_dict["state_std"])
    model.load_net("Model/DT/saved_model")
    test_state = np.ones(16, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")


def main():
    parser = argparse.ArgumentParser(description="Train Decision Transformer for bidding")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="saved_model/DTtest")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tb_log_dir", type=str, default="logs/tensorboard/dt")
    args = parser.parse_args()
    run_dt(
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_path,
        num_workers=args.num_workers,
        tb_log_dir=args.tb_log_dir,
    )


if __name__ == "__main__":
    main()
