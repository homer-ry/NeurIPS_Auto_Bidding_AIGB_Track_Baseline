"""
CBD vs DDPO-CBD 对比评估脚本

对比CBD baseline和DDPO fine-tuned模型的效果，分析收敛性
"""

import subprocess
import re
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def modify_strategy_import(strategy_name):
    """切换策略"""
    init_file = 'bidding_train_env/strategy/__init__.py'
    
    if strategy_name == 'CBD':
        content = '''# from .player_bidding_strategy import PlayerBiddingStrategy as PlayerBiddingStrategy
# from .dt_bidding_strategy import DtBiddingStrategy as PlayerBiddingStrategy
# from .dd_bidding_strategy import DdBiddingStrategy as PlayerBiddingStrategy
# from .dit_bidding_strategy import DiTBiddingStrategy as PlayerBiddingStrategy
from .cbd_bidding_strategy import CbdBiddingStrategy as PlayerBiddingStrategy
# from .ddpo_cbd_bidding_strategy import DdpoCbdBiddingStrategy as PlayerBiddingStrategy
'''
    elif strategy_name == 'DDPO-CBD':
        content = '''# from .player_bidding_strategy import PlayerBiddingStrategy as PlayerBiddingStrategy
# from .dt_bidding_strategy import DtBiddingStrategy as PlayerBiddingStrategy
# from .dd_bidding_strategy import DdBiddingStrategy as PlayerBiddingStrategy
# from .dit_bidding_strategy import DiTBiddingStrategy as PlayerBiddingStrategy
# from .cbd_bidding_strategy import CbdBiddingStrategy as PlayerBiddingStrategy
from .ddpo_cbd_bidding_strategy import DdpoCbdBiddingStrategy as PlayerBiddingStrategy
'''
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    with open(init_file, 'w') as f:
        f.write(content)
    print(f"[INFO] 已切换策略为: {strategy_name}")


def run_evaluate():
    """运行评估"""
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    result = subprocess.run(
        ['python', '-u', '-m', 'run.run_evaluate'],
        capture_output=True,
        text=True,
        env=env,
        timeout=600
    )
    return result.stdout + result.stderr


def parse_results(output):
    """解析评估结果"""
    results = {}
    patterns = {
        'total_reward': r'Total Reward:\s*([\d.]+)',
        'total_cost': r'Total Cost:\s*([\d.]+)',
        'cpa_real': r'CPA-real:\s*([\d.]+)',
        'cpa_constraint': r'CPA-constraint:\s*([\d.]+)',
        'score': r'Score:\s*([\d.]+)'
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            results[key] = float(match.group(1))
    return results


def generate_comparison_report(cbd_results, ddpo_results, timestamp):
    """生成对比报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CBD vs DDPO-CBD 对比评估报告")
    report_lines.append("=" * 80)
    report_lines.append(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append(f"{'指标':<20} {'CBD':>15} {'DDPO-CBD':>15} {'提升':>15} {'更好':>8}")
    report_lines.append("-" * 80)
    
    metrics = [
        ('Total Reward', 'total_reward', 'higher'),
        ('Total Cost', 'total_cost', 'lower'),
        ('CPA Real', 'cpa_real', 'lower'),
        ('Score', 'score', 'higher')
    ]
    
    improvements = {}
    for label, key, direction in metrics:
        cbd_val = cbd_results.get(key, 0)
        ddpo_val = ddpo_results.get(key, 0)
        
        if direction == 'higher':
            improvement = ((ddpo_val - cbd_val) / max(abs(cbd_val), 1e-6)) * 100
            better = 'DDPO-CBD' if ddpo_val > cbd_val else 'CBD'
        else:
            improvement = ((cbd_val - ddpo_val) / max(abs(cbd_val), 1e-6)) * 100
            better = 'DDPO-CBD' if ddpo_val < cbd_val else 'CBD'
        
        improvements[key] = improvement
        report_lines.append(
            f"{label:<20} {cbd_val:>15.4f} {ddpo_val:>15.4f} {improvement:>14.2f}% {better:>8}"
        )
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("收敛性分析")
    report_lines.append("=" * 80)
    
    # Load training history if available
    history_path = 'saved_model/DDPO-CBD/training_history.json'
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        report_lines.append(f"训练轮次: {len(history.get('epoch', []))}")
        report_lines.append(f"最终策略损失: {history['policy_loss'][-1]:.6f}")
        report_lines.append(f"最终奖励损失: {history['reward_loss'][-1]:.6f}")
        report_lines.append(f"最终预测奖励: {history['mean_reward'][-1]:.4f} ± {history['std_reward'][-1]:.4f}")
        
        # Check convergence
        if len(history['policy_loss']) > 10:
            recent_losses = history['policy_loss'][-10:]
            loss_std = np.std(recent_losses)
            report_lines.append(f"最后10轮策略损失标准差: {loss_std:.6f}")
            if loss_std < 0.01:
                report_lines.append("✓ 训练已收敛 (策略损失标准差 < 0.01)")
            else:
                report_lines.append("○ 训练可能未完全收敛")
    
    return '\n'.join(report_lines), improvements


def plot_comparison(history_path, output_path):
    """绘制训练曲线"""
    if not os.path.exists(history_path):
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Policy loss
    axes[0, 0].plot(history['epoch'], history['policy_loss'], color='blue')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Policy Loss')
    axes[0, 0].set_title('DDPO Policy Loss (Convergence)')
    axes[0, 0].grid(True)
    
    # Reward loss
    axes[0, 1].plot(history['epoch'], history['reward_loss'], color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reward Loss')
    axes[0, 1].set_title('Reward Model Loss')
    axes[0, 1].grid(True)
    
    # Predicted reward
    axes[1, 0].plot(history['epoch'], history['mean_reward'], color='green')
    axes[1, 0].fill_between(
        history['epoch'],
        np.array(history['mean_reward']) - np.array(history['std_reward']),
        np.array(history['mean_reward']) + np.array(history['std_reward']),
        alpha=0.3, color='green'
    )
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Predicted Reward')
    axes[1, 0].set_title('RM Predicted Reward')
    axes[1, 0].grid(True)
    
    # Eval score
    if history.get('eval_score'):
        eval_epochs = np.linspace(1, len(history['epoch']), len(history['eval_score']))
        axes[1, 1].plot(eval_epochs, history['eval_score'], marker='o', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Evaluation Score')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'ddpo_cbd_training_curves.png'), dpi=150)
    plt.close()


def main():
    print("=" * 80)
    print("CBD vs DDPO-CBD 对比评估")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    
    all_results = {}
    
    # Evaluate CBD baseline
    print("\n" + "=" * 80)
    print("[1/2] 评估 CBD 基线模型")
    print("=" * 80)
    modify_strategy_import('CBD')
    output = run_evaluate()
    cbd_results = parse_results(output)
    all_results['CBD'] = cbd_results
    
    print(f"CBD 结果:")
    print(f"  Reward: {cbd_results.get('total_reward', 0):.2f}")
    print(f"  Cost: {cbd_results.get('total_cost', 0):.2f}")
    print(f"  Score: {cbd_results.get('score', 0):.4f}")
    
    # Evaluate DDPO-CBD
    print("\n" + "=" * 80)
    print("[2/2] 评估 DDPO-CBD 模型")
    print("=" * 80)
    modify_strategy_import('DDPO-CBD')
    output = run_evaluate()
    ddpo_results = parse_results(output)
    all_results['DDPO-CBD'] = ddpo_results
    
    print(f"DDPO-CBD 结果:")
    print(f"  Reward: {ddpo_results.get('total_reward', 0):.2f}")
    print(f"  Cost: {ddpo_results.get('total_cost', 0):.2f}")
    print(f"  Score: {ddpo_results.get('score', 0):.4f}")
    
    # Generate report
    report, improvements = generate_comparison_report(cbd_results, ddpo_results, timestamp)
    
    print("\n" + report)
    
    # Save results
    txt_file = f'results/cbd_vs_ddpo_compare_{timestamp}.txt'
    json_file = f'results/cbd_vs_ddpo_compare_{timestamp}.json'
    
    with open(txt_file, 'w') as f:
        f.write(report)
    
    with open(json_file, 'w') as f:
        json.dump({
            'CBD': cbd_results,
            'DDPO-CBD': ddpo_results,
            'improvements': improvements,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\n[INFO] 结果已保存到: {txt_file}")
    print(f"[INFO] JSON 结果已保存到: {json_file}")
    
    # Plot training curves if available
    plot_comparison('saved_model/DDPO-CBD/training_history.json', 'results')
    
    return all_results


if __name__ == '__main__':
    main()
