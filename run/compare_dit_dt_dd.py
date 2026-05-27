"""
三模型对比评估脚本: DiT vs DT vs DD

自动评估三个模型并生成对比报告
"""
import subprocess
import sys
import re
from datetime import datetime
import json
import os


def modify_strategy_import(strategy_name):
    """修改策略导入文件"""
    init_file = 'bidding_train_env/strategy/__init__.py'
    
    if strategy_name == 'DT':
        content = '''# from .player_bidding_strategy import PlayerBiddingStrategy as PlayerBiddingStrategy
from .dt_bidding_strategy import DtBiddingStrategy as PlayerBiddingStrategy
# from .dd_bidding_strategy import DdBiddingStrategy as PlayerBiddingStrategy
# from .dit_bidding_strategy import DiTBiddingStrategy as PlayerBiddingStrategy
'''
    elif strategy_name == 'DD':
        content = '''# from .player_bidding_strategy import PlayerBiddingStrategy as PlayerBiddingStrategy
# from .dt_bidding_strategy import DtBiddingStrategy as PlayerBiddingStrategy
from .dd_bidding_strategy import DdBiddingStrategy as PlayerBiddingStrategy
# from .dit_bidding_strategy import DiTBiddingStrategy as PlayerBiddingStrategy
'''
    elif strategy_name == 'DiT':
        content = '''# from .player_bidding_strategy import PlayerBiddingStrategy as PlayerBiddingStrategy
# from .dt_bidding_strategy import DtBiddingStrategy as PlayerBiddingStrategy
# from .dd_bidding_strategy import DdBiddingStrategy as PlayerBiddingStrategy
from .dit_bidding_strategy import DiTBiddingStrategy as PlayerBiddingStrategy
'''
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    with open(init_file, 'w') as f:
        f.write(content)
    print(f"[INFO] 已切换策略为: {strategy_name}")


def run_evaluate():
    """运行评估并返回结果"""
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    result = subprocess.run(
        ['python', '-u', '-m', 'run.run_evaluate'],
        capture_output=True,
        text=True,
        env=env
    )
    return result.stdout + result.stderr


def parse_results(output):
    """从输出中解析评估指标"""
    results = {}
    
    # 使用正则表达式提取关键指标
    patterns = {
        'strategy_name': r'Strategy Name:\s*(\S+)',
        'total_reward': r'Total Reward:\s*([\d.]+)',
        'total_cost': r'Total Cost:\s*([\d.]+)',
        'cpa_real': r'CPA-real:\s*([\d.]+)',
        'cpa_constraint': r'CPA-constraint:\s*([\d.]+)',
        'score': r'Score:\s*([\d.]+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            try:
                results[key] = float(match.group(1))
            except ValueError:
                results[key] = match.group(1)
    
    return results


def main():
    print("=" * 70)
    print("DiT vs DT vs DD 三模型对比评估")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}
    
    models = ['DT', 'DD', 'DiT']
    
    for i, model_name in enumerate(models):
        print("\n" + "=" * 70)
        print(f"[{i+1}/3] 评估 {model_name} 模型")
        print("=" * 70)
        
        modify_strategy_import(model_name)
        output = run_evaluate()
        results = parse_results(output)
        results['raw_output'] = output
        all_results[model_name] = results
        
        # 打印关键指标
        if results:
            print(f"\n{model_name} 结果:")
            print(f"  Reward: {results.get('total_reward', 0):.2f}")
            print(f"  Cost: {results.get('total_cost', 0):.2f}")
            print(f"  CPA: {results.get('cpa_real', 0):.4f}")
            print(f"  Score: {results.get('score', 0):.4f}")
    
    # 生成对比报告
    report = generate_report(all_results)
    
    # 保存结果
    save_results(all_results, report, timestamp)
    
    # 打印对比结果
    print("\n" + "=" * 70)
    print("三模型对比结果")
    print("=" * 70)
    print(report)


def generate_report(results):
    """生成对比报告"""
    report_lines = []
    report_lines.append(f"{'Metric':<20} {'DT':>15} {'DD':>15} {'DiT':>15} {'Best':>10}")
    report_lines.append("-" * 80)
    
    metrics = [
        ('Total Reward', 'total_reward', 'higher'),
        ('Total Cost', 'total_cost', 'lower'),
        ('CPA Real', 'cpa_real', 'lower'),
        ('CPA Constraint', 'cpa_constraint', 'same'),
        ('Score', 'score', 'higher')
    ]
    
    for label, key, direction in metrics:
        dt_val = results['DT'].get(key, 0)
        dd_val = results['DD'].get(key, 0)
        dit_val = results['DiT'].get(key, 0)
        
        # 确定最佳值
        if direction == 'higher':
            best_val = max(dt_val, dd_val, dit_val)
            best_model = 'DT' if dt_val == best_val else ('DD' if dd_val == best_val else 'DiT')
        elif direction == 'lower':
            best_val = min(dt_val, dd_val, dit_val)
            best_model = 'DT' if dt_val == best_val else ('DD' if dd_val == best_val else 'DiT')
        else:
            best_model = '-'
        
        report_lines.append(
            f"{label:<20} {dt_val:>15.4f} {dd_val:>15.4f} {dit_val:>15.4f} {best_model:>10}"
        )
    
    return '\n'.join(report_lines)


def save_results(results, report, timestamp):
    """保存评估结果到文件"""
    os.makedirs('results', exist_ok=True)
    
    # 保存为文本报告
    report_file = f'results/three_model_compare_{timestamp}.txt'
    
    with open(report_file, 'w') as f:
        f.write("DiT vs DT vs DD 三模型评估对比报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(report)
        f.write("\n\n")
        
        for model_name in ['DT', 'DD', 'DiT']:
            f.write("=" * 80 + "\n")
            f.write(f"详细输出 - {model_name}\n")
            f.write("=" * 80 + "\n")
            f.write(results[model_name].get('raw_output', ''))
            f.write("\n\n")
    
    print(f"\n[INFO] 结果已保存到: {report_file}")
    
    # 保存为 JSON
    json_file = f'results/three_model_compare_{timestamp}.json'
    
    # 移除原始输出，避免 JSON 过大
    json_results = {
        k: {key: val for key, val in v.items() if key != 'raw_output'}
        for k, v in results.items()
    }
    json_results['timestamp'] = timestamp
    json_results['report'] = report
    
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"[INFO] JSON 结果已保存到: {json_file}")


if __name__ == '__main__':
    main()
