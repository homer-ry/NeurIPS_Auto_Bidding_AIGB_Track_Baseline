"""
DiT (Diffusion Transformer) Training Script

基于 CAD (Causal Auto-bidding via Diffusion Models) 的训练流程
"""
import os
import sys
import torch
from torch.utils.data import DataLoader
import tqdm
import argparse
from datetime import datetime
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidding_train_env.baseline.dit.DFUSER import DFUSER
from bidding_train_env.baseline.dit.dataset import aigb_dataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def run_dit_training(
        save_path="saved_model/DiTtest",
        train_epoch=1000,
        batch_size=1000,
        gamma=1, 
        tau=0.01, 
        lr=1e-4,
        network_random_seed=200,
        n_timesteps=100,  # DiT 推荐更多步数
        pretrained_model=None,
        test_generation=False,
        model_choice='DiT1d',  # DiT 模型
        attn_block='causal',  # 因果注意力
        predict_epsilon=True,
        cond_obs_training=False,
        pred_one_step=False,
        traj_add_a=False,
        rtg_preference='score',
        save_every=100
    ):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Train epochs: {train_epoch}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Diffusion steps: {n_timesteps}")
    logger.info(f"Model: {model_choice} with {attn_block} attention")
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 初始化 DiT 模型
    dim_obs = 16
    algorithm = DFUSER(
        dim_obs=dim_obs, 
        gamma=gamma, 
        tau=tau, 
        lr=lr,
        network_random_seed=network_random_seed, 
        n_timesteps=n_timesteps,
        model_choice=model_choice,
        attn_block=attn_block,
        predict_epsilon=predict_epsilon,
        cond_obs_training=cond_obs_training,
        pred_one_step=pred_one_step,
        traj_add_a=traj_add_a
    )
    
    # 加载预训练模型（如果有）
    if test_generation and pretrained_model is not None:
        logger.info(f'Loading model from {pretrained_model}')
        algorithm.load_net(pretrained_model)
        
    algorithm = algorithm.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in algorithm.parameters())
    logger.info(f"Total parameters: {total_params:,} of model {model_choice}")
    
    # 加载数据集
    logger.info("Loading dataset...")
    dataset = aigb_dataset(
        algorithm.step_len, 
        load_preprocessed_tain_data=True, 
        sparse_data=False,  # 使用完整数据集
        simplify_state=False, 
        rtg_preference=rtg_preference
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=int(batch_size), 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Number of batches: {len(dataloader)}")
    
    # 训练循环
    best_loss = float('inf')
    
    for epoch in tqdm.tqdm(range(1, train_epoch + 1), desc='Training DiT'):
        record_epoch_loss = 0.
        record_epoch_diff_loss = 0.
        record_epoch_inv_loss = 0.
        record_epoch_a_t_loss = 0.
        num_batches = 0
        
        for batch_index, (states, actions, returns, masks) in enumerate(dataloader):
            states = states.to(device)
            actions = actions.to(device)
            returns = returns.to(device)
            masks = masks.to(device)
            
            # 训练步骤
            result = algorithm.trainStep(states, actions, returns, masks)
            
            # trainStep 返回: loss, (diffuse_loss, inv_loss), infos
            total_loss = result[0]
            diffuse_loss, inv_loss = result[1]
            
            record_epoch_loss += total_loss.item()
            record_epoch_diff_loss += diffuse_loss.item()
            
            if inv_loss is not None:
                record_epoch_inv_loss += inv_loss.item()
            
            num_batches += 1
        
        # 计算平均损失
        avg_loss = record_epoch_loss / num_batches
        avg_diff_loss = record_epoch_diff_loss / num_batches
        
        # 日志记录
        if epoch % 10 == 0:
            log_msg = f"Epoch {epoch}: Total Loss={avg_loss:.4f}, Diff Loss={avg_diff_loss:.4f}"
            if record_epoch_inv_loss > 0:
                log_msg += f", Inv Loss={record_epoch_inv_loss/num_batches:.4f}"
            logger.info(log_msg)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            algorithm.save_net(save_path, save_name='diffuser_best.pt')
            logger.info(f"Saved best model at epoch {epoch} with loss {best_loss:.4f}")
        
        # 定期保存检查点
        if epoch % save_every == 0:
            algorithm.save_net(save_path, save_name=f'diffuser_epoch_{epoch}.pt')
            logger.info(f"Saved checkpoint at epoch {epoch}")
    
    # 保存最终模型
    algorithm.save_net(save_path, save_name='diffuser.pt')
    logger.info(f"Training completed! Final model saved to {save_path}/diffuser.pt")
    logger.info(f"Best loss: {best_loss:.4f}")
    
    return algorithm


def main():
    parser = argparse.ArgumentParser(description='Train DiT model for bidding')
    
    # 基础参数
    parser.add_argument('--save_path', type=str, default='saved_model/DiTtest',
                       help='Path to save models')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=200,
                       help='Random seed')
    
    # DiT 特定参数
    parser.add_argument('--n_timesteps', type=int, default=100,
                       help='Number of diffusion timesteps (10/100 recommended)')
    parser.add_argument('--model_choice', type=str, default='DiT1d',
                       choices=['DiT1d', 'Unet'],
                       help='Model architecture')
    parser.add_argument('--attn_block', type=str, default='causal',
                       choices=['causal', 'vanilla'],
                       help='Attention block type')
    parser.add_argument('--predict_epsilon', action='store_true', default=True,
                       help='Predict noise instead of state')
    parser.add_argument('--traj_add_a', action='store_true', default=False,
                       help='Include actions in trajectory')
    parser.add_argument('--rtg_preference', type=str, default='score',
                       choices=['reward', 'score'],
                       help='Return-to-go preference (score considers CPA)')
    
    # 其他参数
    parser.add_argument('--save_every', type=int, default=100,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained model')
    parser.add_argument('--test_gen', action='store_true',
                       help='Test generation capability')
    
    args = parser.parse_args()
    
    # 运行训练
    run_dit_training(
        save_path=args.save_path,
        train_epoch=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        network_random_seed=args.seed,
        n_timesteps=args.n_timesteps,
        pretrained_model=args.pretrained,
        test_generation=args.test_gen,
        model_choice=args.model_choice,
        attn_block=args.attn_block,
        predict_epsilon=args.predict_epsilon,
        traj_add_a=args.traj_add_a,
        rtg_preference=args.rtg_preference,
        save_every=args.save_every
    )


if __name__ == '__main__':
    main()
