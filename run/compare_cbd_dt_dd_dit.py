"""
四模型对比评估脚本: CBD vs DT vs DD vs DiT
"""
import subprocess
import re
from datetime import datetime
import json
import os


def modify_strategy_import(strategy_name):
    init_file = 'bidding_train_env/strategy/__init__.py'

    if strategy_name == 'DT':
        content = '''# from .player_bidding_strategy import PlayerBiddingStrategy as PlayerBiddingStrategy
from .dt_bidding_strategy import DtBiddingStrategy as PlayerBiddingStrategy
# from .dd_bidding_strategy import DdBiddingStrategy as PlayerBiddingStrategy
# from .dit_bidding_strategy import DiTBiddingStrategy as PlayerBiddingStrategy
# from .cbd_bidding_strategy import CbdBiddingStrategy as PlayerBiddingStrategy
'''
    elif strategy_name == 'DD':
        content = '''# from .player_bidding_strategy import PlayerBiddingStrategy as PlayerBiddingStrategy
# from .dt_bidding_strategy import DtBiddingStrategy as PlayerBiddingStrategy
from .dd_bidding_strategy import DdBiddingStrategy as PlayerBiddingStrategy
# from .dit_bidding_strategy import DiTBiddingStrategy as PlayerBiddingStrategy
# from .cbd_bidding_strategy import CbdBiddingStrategy as PlayerBiddingStrategy
'''
    elif strategy_name == 'DiT':
        content = '''# from .player_bidding_strategy import PlayerBiddingStrategy as PlayerBiddingStrategy
# from .dt_bidding_strategy import DtBiddingStrategy as PlayerBiddingStrategy
# from .dd_bidding_strategy import DdBiddingStrategy as PlayerBiddingStrategy
from .dit_bidding_strategy import DiTBiddingStrategy as PlayerBiddingStrategy
# from .cbd_bidding_strategy import CbdBiddingStrategy as PlayerBiddingStrategy
'''
    elif strategy_name == 'CBD':
        content = '''# from .player_bidding_strategy import PlayerBiddingStrategy as PlayerBiddingStrategy
# from .dt_bidding_strategy import DtBiddingStrategy as PlayerBiddingStrategy
# from .dd_bidding_strategy import DdBiddingStrategy as PlayerBiddingStrategy
# from .dit_bidding_strategy import DiTBiddingStrategy as PlayerBiddingStrategy
from .cbd_bidding_strategy import CbdBiddingStrategy as PlayerBiddingStrategy
'''
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    with open(init_file, 'w') as f:
        f.write(content)
    print(f"[INFO] 已切换策略为: {strategy_name}")


def run_evaluate():
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


def generate_report(results):
    report_lines = []
    report_lines.append(f"{'Metric':<20} {'DT':>12} {'DD':>12} {'DiT':>12} {'CBD':>12} {'Best':>8}")
    report_lines.append("-" * 82)

    metrics = [
        ('Total Reward', 'total_reward', 'higher'),
        ('Total Cost', 'total_cost', 'lower'),
        ('CPA Real', 'cpa_real', 'lower'),
        ('CPA Constraint', 'cpa_constraint', 'same'),
        ('Score', 'score', 'higher')
    ]

    for label, key, direction in metrics:
        dt_val = results.get('DT', {}).get(key, 0)
        dd_val = results.get('DD', {}).get(key, 0)
        dit_val = results.get('DiT', {}).get(key, 0)
        cbd_val = results.get('CBD', {}).get(key, 0)

        vals = {'DT': dt_val, 'DD': dd_val, 'DiT': dit_val, 'CBD': cbd_val}
        if direction == 'higher':
            best_model = max(vals, key=vals.get)
        elif direction == 'lower':
            best_model = min(vals, key=vals.get)
        else:
            best_model = '-'

        report_lines.append(
            f"{label:<20} {dt_val:>12.4f} {dd_val:>12.4f} {dit_val:>12.4f} {cbd_val:>12.4f} {best_model:>8}"
        )

    return '\n'.join(report_lines)


def save_results(results, report, timestamp):
    os.makedirs('results', exist_ok=True)
    txt_file = f'results/four_model_compare_{timestamp}.txt'
    json_file = f'results/four_model_compare_{timestamp}.json'

    with open(txt_file, 'w') as f:
        f.write("CBD vs DT vs DD vs DiT 四模型评估对比报告\n")
        f.write("=" * 90 + "\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(report)
        f.write("\n\n")
        for m in ['DT', 'DD', 'DiT', 'CBD']:
            f.write("=" * 90 + "\n")
            f.write(f"详细输出 - {m}\n")
            f.write("=" * 90 + "\n")
            f.write(results.get(m, {}).get('raw_output', ''))
            f.write("\n\n")

    json_results = {
        k: {kk: vv for kk, vv in v.items() if kk != 'raw_output'}
        for k, v in results.items()
    }
    json_results['timestamp'] = timestamp
    json_results['report'] = report

    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"[INFO] 结果已保存到: {txt_file}")
    print(f"[INFO] JSON 结果已保存到: {json_file}")


def main():
    print("=" * 90)
    print("CBD vs DT vs DD vs DiT 四模型对比评估")
    print("=" * 90)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}
    models = ['DT', 'DD', 'DiT', 'CBD']

    for i, model_name in enumerate(models):
        print("\n" + "=" * 90)
        print(f"[{i+1}/{len(models)}] 评估 {model_name} 模型")
        print("=" * 90)
        modify_strategy_import(model_name)
        output = run_evaluate()
        result = parse_results(output)
        result['raw_output'] = output
        all_results[model_name] = result

        print(f"{model_name} 结果:")
        print(f"  Reward: {result.get('total_reward', 0):.2f}")
        print(f"  Cost: {result.get('total_cost', 0):.2f}")
        print(f"  CPA: {result.get('cpa_real', 0):.4f}")
        print(f"  Score: {result.get('score', 0):.4f}")

    report = generate_report(all_results)
    save_results(all_results, report, timestamp)

    print("\n" + "=" * 90)
    print("四模型对比结果")
    print("=" * 90)
    print(report)


if __name__ == '__main__':
    main()
