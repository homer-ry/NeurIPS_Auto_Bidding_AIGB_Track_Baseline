import argparse
import json
import math
import os
import pickle
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidding_train_env.environment.offline_env import OfflineEnv
from bidding_train_env.strategy.cbd_bidding_strategy import CbdBiddingStrategy
from bidding_train_env.strategy.dd_bidding_strategy import DdBiddingStrategy
from bidding_train_env.strategy.ddpo_cbd_bidding_strategy import DDPOCbdBiddingStrategy
from bidding_train_env.strategy.dit_bidding_strategy import DiTBiddingStrategy
from bidding_train_env.strategy.dt_bidding_strategy import DtBiddingStrategy
from bidding_train_env.strategy.player_bidding_strategy import PlayerBiddingStrategy


def get_score_nips(reward: float, cpa: float, cpa_constraint: float) -> float:
    beta = 2
    penalty = 1.0
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


@dataclass
class EpisodeData:
    key: Tuple[int, int]
    budget: float
    cpa_constraint: float
    category: int
    pvalues: List[np.ndarray]
    pvalue_sigmas: List[np.ndarray]
    least_winning_costs: List[np.ndarray]


def load_period_episodes(file_path: str) -> List[EpisodeData]:
    cache_path = f"{file_path}.episodes.pkl"
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(file_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    df = pd.read_csv(file_path)
    df = df.sort_values(["deliveryPeriodIndex", "advertiserNumber", "timeStepIndex"])
    episodes: List[EpisodeData] = []

    grouped = df.groupby(["deliveryPeriodIndex", "advertiserNumber"], sort=True)
    for key, group in grouped:
        first_row = group.iloc[0]
        timestep_group = group.groupby("timeStepIndex", sort=True)
        pvalues = timestep_group["pValue"].apply(lambda x: x.to_numpy(dtype=np.float32)).tolist()
        pvalue_sigmas = timestep_group["pValueSigma"].apply(lambda x: x.to_numpy(dtype=np.float32)).tolist()
        least_winning_costs = timestep_group["leastWinningCost"].apply(lambda x: x.to_numpy(dtype=np.float32)).tolist()
        episodes.append(
            EpisodeData(
                key=(int(key[0]), int(key[1])),
                budget=float(first_row["budget"]),
                cpa_constraint=float(first_row["CPAConstraint"]),
                category=int(first_row["advertiserCategoryIndex"]),
                pvalues=pvalues,
                pvalue_sigmas=pvalue_sigmas,
                least_winning_costs=least_winning_costs,
            )
        )
    with open(cache_path, "wb") as f:
        pickle.dump(episodes, f)
    return episodes


def configure_agent(agent, budget: float, cpa: float, category: int) -> None:
    agent.budget = float(budget)
    agent.remaining_budget = float(budget)
    agent.cpa = float(cpa)
    agent.category = int(category)
    if hasattr(agent, "cpa_condition"):
        agent.cpa_condition = torch.clamp(
            (torch.tensor(float(cpa), dtype=torch.float32) - 6) / (12 - 6),
            min=0.0,
            max=1.0,
        )
    agent.reset()


def build_agent(
    model_name: str,
    model_overrides: Optional[Dict[str, Optional[str]]] = None,
    model_options: Optional[Dict[str, float]] = None,
):
    model_overrides = model_overrides or {}
    model_options = model_options or {}
    if model_name == "Heuristic":
        return PlayerBiddingStrategy()
    if model_name == "DT":
        return DtBiddingStrategy()
    if model_name == "DD":
        return DdBiddingStrategy()
    if model_name == "DiT":
        return DiTBiddingStrategy(model_name=model_overrides.get("DiT"))
    if model_name == "CBD":
        return CbdBiddingStrategy(model_name=model_overrides.get("CBD"))
    if model_name == "CBD-PPO":
        return DDPOCbdBiddingStrategy(
            model_name=model_overrides.get("CBD-PPO"),
            bid_scale=model_options.get("CBD-PPO-bid-scale", 1.0),
            bid_bias=model_options.get("CBD-PPO-bid-bias", 0.0),
        )
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_episode(agent, episode: EpisodeData, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    configure_agent(agent, episode.budget, episode.cpa_constraint, episode.category)
    env = OfflineEnv()

    history = {
        "historyBids": [],
        "historyAuctionResult": [],
        "historyImpressionResult": [],
        "historyLeastWinningCost": [],
        "historyPValueInfo": [],
    }

    rewards = np.zeros(len(episode.pvalues), dtype=np.float32)
    total_value = 0.0

    for timestep_index, (pvalue, pvalue_sigma, least_winning_cost) in enumerate(
        zip(episode.pvalues, episode.pvalue_sigmas, episode.least_winning_costs)
    ):
        if agent.remaining_budget < env.min_remaining_budget:
            bid = np.zeros(pvalue.shape[0], dtype=np.float32)
        else:
            bid = agent.bidding(
                timestep_index,
                pvalue,
                pvalue_sigma,
                history["historyPValueInfo"],
                history["historyBids"],
                history["historyAuctionResult"],
                history["historyImpressionResult"],
                history["historyLeastWinningCost"],
            )

        tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
            pvalue,
            pvalue_sigma,
            bid,
            least_winning_cost,
            rng=rng,
        )

        over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
        while over_cost_ratio > 0:
            pv_index = np.where(tick_status == 1)[0]
            if pv_index.size == 0:
                break
            dropped_pv_index = rng.choice(
                pv_index,
                int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                replace=False,
            )
            bid[dropped_pv_index] = 0
            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
                pvalue,
                pvalue_sigma,
                bid,
                least_winning_cost,
                rng=rng,
            )
            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

        agent.remaining_budget -= float(np.sum(tick_cost))
        rewards[timestep_index] = float(np.sum(tick_conversion))
        total_value += float(np.sum(tick_value))

        history["historyPValueInfo"].append(
            np.array([(pvalue[i], pvalue_sigma[i]) for i in range(pvalue.shape[0])], dtype=np.float32)
        )
        history["historyBids"].append(np.asarray(bid, dtype=np.float32))
        history["historyLeastWinningCost"].append(np.asarray(least_winning_cost, dtype=np.float32))
        history["historyAuctionResult"].append(
            np.array([(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])], dtype=np.float32)
        )
        history["historyImpressionResult"].append(
            np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pvalue.shape[0])], dtype=np.float32)
        )

    total_reward = float(np.sum(rewards))
    total_cost = float(episode.budget - agent.remaining_budget)
    cpa_real = total_cost / (total_reward + 1e-10)
    score = get_score_nips(total_reward, cpa_real, episode.cpa_constraint)
    budget_utilization = total_cost / (episode.budget + 1e-10)
    return {
        "reward": total_reward,
        "cost": total_cost,
        "value": total_value,
        "cpa_real": cpa_real,
        "cpa_constraint": float(episode.cpa_constraint),
        "score": score,
        "budget_utilization": budget_utilization,
        "is_zero_conversion": 1.0 if total_reward == 0 else 0.0,
    }


def summarize(metrics_list: Iterable[Dict[str, float]]) -> Dict[str, float]:
    metrics_list = list(metrics_list)
    if not metrics_list:
        return {}

    summary: Dict[str, float] = {}
    keys = metrics_list[0].keys()
    for key in keys:
        values = np.array([item[key] for item in metrics_list], dtype=np.float64)
        if key in {"reward", "cost", "value", "score"}:
            summary[f"{key}_sum"] = float(np.sum(values))
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def evaluate_model_on_periods(
    agent,
    periods: List[int],
    traffic_dir: str,
    repeats: int,
    max_episodes_per_period: int,
    base_seed: int,
) -> Tuple[Dict[str, dict], List[Dict[str, float]]]:
    per_period_summary: Dict[str, dict] = {}
    all_episode_metrics: List[Dict[str, float]] = []

    for period in periods:
        file_path = os.path.join(traffic_dir, f"period-{period}.csv")
        episodes = load_period_episodes(file_path)
        if max_episodes_per_period > 0:
            episodes = episodes[:max_episodes_per_period]

        period_metrics: List[Dict[str, float]] = []
        for repeat in range(repeats):
            for episode_idx, episode in enumerate(episodes):
                seed = base_seed + period * 100000 + repeat * 1000 + episode_idx
                metrics = evaluate_episode(agent, episode, seed=seed)
                metrics["period"] = period
                metrics["repeat"] = repeat
                metrics["advertiser_number"] = episode.key[1]
                period_metrics.append(metrics)
                all_episode_metrics.append(metrics)

        per_period_summary[str(period)] = summarize(period_metrics)
        per_period_summary[str(period)]["num_episodes"] = len(period_metrics)

    return per_period_summary, all_episode_metrics


def build_markdown_report(results: Dict[str, dict], model_order: List[str], periods: List[int]) -> str:
    lines: List[str] = []
    lines.append("# Open-Source Benchmark Results")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Model | Mean Score | Mean Reward | Mean CPA | Mean Budget Util. | Zero-Conv Ratio |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for model_name in model_order:
        overall = results[model_name]["overall"]
        lines.append(
            f"| {model_name} | {overall['score_mean']:.4f} | {overall['reward_mean']:.4f} | "
            f"{overall['cpa_real_mean']:.4f} | {overall['budget_utilization_mean']:.4f} | "
            f"{overall['is_zero_conversion_mean']:.4f} |"
        )

    lines.append("")
    lines.append("## Per Period")
    lines.append("")
    for period in periods:
        lines.append(f"### Period-{period}")
        lines.append("")
        lines.append("| Model | Mean Score | Mean Reward | Mean CPA | Mean Budget Util. |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for model_name in model_order:
            period_summary = results[model_name]["per_period"][str(period)]
            lines.append(
                f"| {model_name} | {period_summary['score_mean']:.4f} | {period_summary['reward_mean']:.4f} | "
                f"{period_summary['cpa_real_mean']:.4f} | {period_summary['budget_utilization_mean']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run open-source benchmark on public traffic periods.")
    parser.add_argument("--traffic_dir", type=str, default="./data/traffic")
    parser.add_argument("--periods", type=int, nargs="+", default=[7, 8, 9, 10, 11, 12, 13])
    parser.add_argument("--models", type=str, nargs="+", default=["Heuristic", "DT", "DD", "DiT", "CBD", "CBD-PPO"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max_episodes_per_period", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--output_dir", type=str, default="./results/open_source_benchmark")
    parser.add_argument("--cbd_model_path", type=str, default="")
    parser.add_argument("--cbd_ppo_model_path", type=str, default="")
    parser.add_argument("--dit_model_path", type=str, default="")
    parser.add_argument("--cbd_ppo_bid_scale", type=float, default=1.0)
    parser.add_argument("--cbd_ppo_bid_bias", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_models = {"Heuristic", "DT", "DD", "DiT", "CBD", "CBD-PPO"}
    model_order = [model for model in args.models if model in valid_models]
    results: Dict[str, dict] = {}
    model_overrides = {
        "CBD": args.cbd_model_path or None,
        "CBD-PPO": args.cbd_ppo_model_path or None,
        "DiT": args.dit_model_path or None,
    }
    model_options = {
        "CBD-PPO-bid-scale": args.cbd_ppo_bid_scale,
        "CBD-PPO-bid-bias": args.cbd_ppo_bid_bias,
    }

    for model_name in model_order:
        agent = build_agent(model_name, model_overrides=model_overrides, model_options=model_options)
        print(f"[INFO] Evaluating model: {model_name}")
        per_period_summary, all_episode_metrics = evaluate_model_on_periods(
            agent=agent,
            periods=args.periods,
            traffic_dir=args.traffic_dir,
            repeats=args.repeats,
            max_episodes_per_period=args.max_episodes_per_period,
            base_seed=args.seed,
        )
        overall_summary = summarize(all_episode_metrics)
        overall_summary["num_episodes"] = len(all_episode_metrics)
        results[model_name] = {
            "overall": overall_summary,
            "per_period": per_period_summary,
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"benchmark_{timestamp}.json"
    md_path = output_dir / f"benchmark_{timestamp}.md"

    json_payload = {
        "timestamp": timestamp,
        "periods": args.periods,
        "models": model_order,
        "repeats": args.repeats,
        "max_episodes_per_period": args.max_episodes_per_period,
        "results": results,
    }
    json_path.write_text(json.dumps(json_payload, indent=2))
    md_path.write_text(build_markdown_report(results, model_order, args.periods))

    print(f"[INFO] Saved JSON results to {json_path}")
    print(f"[INFO] Saved Markdown report to {md_path}")
    print(build_markdown_report(results, model_order, args.periods))


if __name__ == "__main__":
    main()
