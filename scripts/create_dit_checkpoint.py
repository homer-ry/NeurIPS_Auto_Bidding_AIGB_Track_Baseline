"""
创建预初始化的 DiT checkpoint 用于快速演示评估流程
"""
import torch
import sys
import os

sys.path.insert(0, '.')

from bidding_train_env.baseline.dit.DFUSER import DFUSER

print("=" * 70)
print("创建 DiT 预初始化 Checkpoint")
print("=" * 70)

# 创建 DiT 模型实例
print("\n[1/3] 创建 DiT 模型...")
model = DFUSER(
    dim_obs=16,
    n_timesteps=10,
    model_choice='DiT1d',
    attn_block='causal',
    predict_epsilon=True
)

print(f"✅ 模型创建成功")
print(f"✅ 参数量: {sum(p.numel() for p in model.parameters()):,}")

# 创建保存目录
save_path = "saved_model/DiTtest"
os.makedirs(save_path, exist_ok=True)

# 保存模型
print(f"\n[2/3] 保存模型到 {save_path}/diffuser.pt ...")
model.save_net(save_path, save_name='diffuser.pt')

print(f"✅ 模型已保存")

# 验证加载
print(f"\n[3/3] 验证模型加载...")
model_test = DFUSER(
    dim_obs=16,
    n_timesteps=10,
    model_choice='DiT1d',
    attn_block='causal'
)
model_test.load_net(f"{save_path}/diffuser.pt", device='cpu')

print("✅ 模型加载验证成功")

print("\n" + "=" * 70)
print("✅ DiT Checkpoint 创建完成！")
print("=" * 70)
print(f"\n模型路径: {save_path}/diffuser.pt")
print(f"模型大小: {os.path.getsize(f'{save_path}/diffuser.pt') / 1024 / 1024:.2f} MB")
print("\n现在可以运行评估流程了！")
