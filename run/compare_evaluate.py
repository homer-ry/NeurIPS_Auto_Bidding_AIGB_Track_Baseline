"""
对比评估 DT 和 DD 两个 checkpoint 的效果
运行: python run/compare_evaluate.py
"""
import subprocess
import sys
import re
from datetime import datetime
import json


def modify_strategy_import(strategy_name):
    """修改策略导入文件"""
    init_file = 'bidding_train_env/strategy/__init__.py'
    
    if strategy_name == 'DT':
        content = '''# from .player_bidding_strategy import PlayerBiddingStrategy as PlayerBiddingStrategy
from .dt_bidding_strategy import DtBiddingStrategy as PlayerBiddingStrategy
# from .dd_bidding_strategy import DdBiddingStrategy as PlayerBiddingStrategy
'''
    elif strategy_name == 'DD':
        content = '''# from .player_bidding_strategy import PlayerBiddingStrategy as PlayerBiddingStrategy
# from .dt_bidding_strategy import DtBiddingStrategy as PlayerBiddingStrategy
from .dd_bidding_strategy import DdBiddingStrategy as PlayerBiddingStrategy
'''
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    with open(init_file, 'w') as f:
        f.write(content)
    print(f"[INFO] 已切换策略为: {strategy_name}")


def run_evaluate():
    """运行评估并返回结果"""
    import os
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    result = subprocess.run(
        ['python', '-u', 'run/run_evaluate.py'],
        capture_output=True,
        text=True,
        cwd='/Users/rongyu/Documents/develop/bidding/tianchi_bidding/NeurIPS_Auto_Bidding_AIGB_Track_Baseline'
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
    print("=" * 60)
    print("DT vs DD Checkpoint 对比评估")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}
    
    # 评估 DT
    print("\n" + "=" * 60)
    print("[1/2] 评估 Decision Transformer (DT)")
    print("=" * 60)
    modify_strategy_import('DT')
    dt_output = run_evaluate()
    dt_results = parse_results(dt_output)
    dt_results['raw_output'] = dt_output
    all_results['DT'] = dt_results
    
    print(dt_output)
    
    # 评估 DD
    print("\n" + "=" * 60)
    print("[2/2] 评估 Decision Diffuser (DD)")
    print("=" * 60)
    modify_strategy_import('DD')
    dd_output = run_evaluate()
    dd_results = parse_results(dd_output)
    dd_results['raw_output'] = dd_output
    all_results['DD'] = dd_results
    
    print(dd_output)
    
    # 生成对比报告
    report = generate_report(all_results)
    
    # 保存结果
    save_results(all_results, report, timestamp)
    
    # 打印对比结果
    print("\n" + "=" * 60)
    print("对比结果摘要")
    print("=" * 60)
    print(report)


def generate_report(results):
    """生成对比报告"""
    report_lines = []
    report_lines.append(f"{'Metric':<20} {'DT':>15} {'DD':>15} {'Diff':>15}")
    report_lines.append("-" * 70)
    
    metrics = [
        ('Total Reward', 'total_reward'),
        ('Total Cost', 'total_cost'),
        ('CPA Real', 'cpa_real'),
        ('CPA Constraint', 'cpa_constraint'),
        ('Score', 'score')
    ]
    
    for label, key in metrics:
        dt_val = results['DT'].get(key, 0)
        dd_val = results['DD'].get(key, 0)
        diff = dt_val - dd_val if isinstance(dt_val, (int, float)) and isinstance(dd_val, (int, float)) else 'N/A'
        
        if isinstance(diff, float):
            diff_str = f"{diff:+.4f}"
        else:
            diff_str = str(diff)
        
        report_lines.append(f"{label:<20} {dt_val:>15.4f} {dd_val:>15.4f} {diff_str:>15}")
    
    return '\n'.join(report_lines)


def save_results(results, report, timestamp):
    """保存评估结果到文件"""
    # 保存为文本报告
    report_file = f'results/eval_compare_{timestamp}.txt'
    
    # 确保 results 目录存在
    import os
    os.makedirs('results', exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write("DT vs DD 评估对比报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(report)
        f.write("\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("详细输出 - DT\n")
        f.write("=" * 70 + "\n")
        f.write(results['DT'].get('raw_output', ''))
        
        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write("详细输出 - DD\n")
        f.write("=" * 70 + "\n")
        f.write(results['DD'].get('raw_output', ''))
    
    print(f"\n[INFO] 结果已保存到: {report_file}")
    
    # 保存为 JSON
    json_file = f'results/eval_compare_{timestamp}.json'
    
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
