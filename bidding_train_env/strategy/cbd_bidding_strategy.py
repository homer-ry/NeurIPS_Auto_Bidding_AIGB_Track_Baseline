import numpy as np
import torch
import os

from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from bidding_train_env.baseline.dit.DFUSER import DFUSER


class CbdBiddingStrategy(BaseBiddingStrategy):
    """
    CBD (Causal Bidding Diffusion) Strategy
    Use U-Net backbone diffusion + inverse dynamics.
    """

    def __init__(
        self,
        budget=100,
        name="CBD-Bidding-Strategy",
        cpa=2,
        category=1,
        model_name=None,
        model_param=None,
    ):
        super().__init__(budget, name, cpa, category)

        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)

        if model_param is None:
            model_param = {
                "n_timesteps": 10,
                "model_choice": "Unet",
                "state_dim": 16,
                "attn_block": "vanilla",
                "predict_epsilon": False,
            }

        if model_name is not None:
            model_candidates = [model_name]
        else:
            model_candidates = [
                os.path.join(dir_name, "saved_model", "CBDtest", "diffuser_best.pt"),
                os.path.join(dir_name, "saved_model", "CBDtest", "diffuser.pt"),
            ]
        model_path = next((path for path in model_candidates if os.path.exists(path)), None)
        if model_path is None:
            raise FileNotFoundError("No CBD checkpoint found. Checked: " + ", ".join(model_candidates))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DFUSER(
            dim_obs=model_param["state_dim"],
            n_timesteps=model_param["n_timesteps"],
            model_choice=model_param["model_choice"],
            attn_block=model_param.get("attn_block", "vanilla"),
            predict_epsilon=model_param.get("predict_epsilon", False),
            cond_obs_training=True,
            pred_one_step=False,
            traj_add_a=False,
        )
        self.model.load_net(model_path, device=self.device)

        self.state_dim = model_param["state_dim"]
        self.input = np.zeros((48, self.state_dim + 1), dtype=np.float32)

    def reset(self):
        self.remaining_budget = self.budget
        self.input = np.zeros((48, self.state_dim + 1), dtype=np.float32)

    def bidding(
        self,
        timeStepIndex,
        pValues,
        pValueSigmas,
        historyPValueInfo,
        historyBid,
        historyAuctionResult,
        historyImpressionResult,
        historyLeastWinningCost,
    ):
        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pValue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0
        historical_conversion_mean = np.mean([np.mean(reward) for reward in history_conversion]) if history_conversion else 0
        historical_lwc_mean = np.mean([np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0
        historical_pvalues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0
        historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0

        def mean_of_last_n_elements(history, n):
            last_n_data = history[max(0, len(history) - n): len(history)]
            if len(last_n_data) == 0:
                return 0
            return np.mean([np.mean(data) for data in last_n_data])

        last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
        last_three_lwc_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
        last_three_pvalues_mean = mean_of_last_n_elements(history_pValue, 3)
        last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

        current_pvalues_mean = np.mean(pValues)
        current_pv_num = len(pValues)
        historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
        last_three_pv_num_total = sum([len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if historyBid else 0

        test_state = np.array([
            time_left,
            budget_left,
            historical_bid_mean,
            last_three_bid_mean,
            historical_lwc_mean,
            historical_pvalues_mean,
            historical_conversion_mean,
            historical_xi_mean,
            last_three_lwc_mean,
            last_three_pvalues_mean,
            last_three_conversion_mean,
            last_three_xi_mean,
            current_pvalues_mean,
            current_pv_num,
            last_three_pv_num_total,
            historical_pv_num_total,
        ])

        self.input[timeStepIndex, : self.state_dim] = test_state
        self.input[:, -1] = timeStepIndex

        x = torch.tensor(self.input.reshape(-1), device=self.device, dtype=torch.float32)
        rtg = torch.tensor([1.0], device=self.device, dtype=torch.float32)

        actions, _ = self.model(x, rtg)
        alpha = max(actions[0].item(), 0)
        bids = alpha * pValues
        return bids
