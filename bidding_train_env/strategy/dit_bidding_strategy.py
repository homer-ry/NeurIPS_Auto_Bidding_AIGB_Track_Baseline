"""
DiT (Diffusion Transformer) Bidding Strategy

基于 CAD (Causal Auto-bidding via Diffusion Models) 的评估策略
支持可选的 Return Model 梯度引导和 Inverse Dynamic Model
"""
import numpy as np
import torch
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from bidding_train_env.baseline.dit.DFUSER import DFUSER
import os


class DiTBiddingStrategy(BaseBiddingStrategy):
    """
    DiT-based Bidding Strategy with Causal Attention
    """

    def __init__(self, 
                 budget=100, 
                 name="DiT-Causal-Bidding-Strategy", 
                 cpa=2, 
                 category=1,
                 model_name=None, 
                 model_param=None,
                 selective_forward=False,
                 advertiser_id=999,
                 traj_add_a=False,
                 use_RM=False,
                 Return_Model_path=None,
                 use_IDM=False,
                 IDM_path=None):
        
        super().__init__(budget, name, cpa, category)
        
        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        
        # 默认模型参数
        if model_param is None:
            model_param = {
                "n_timesteps": 10,  # 匹配训练时的配置
                "model_choice": 'DiT1d',  # 使用 DiT
                "state_dim": 16,
                "attn_block": 'causal',  # 因果注意力
                "predict_epsilon": True
            }
        
        # 模型路径
        if model_name is not None:
            model_candidates = [model_name]
        else:
            model_candidates = [
                os.path.join(dir_name, "saved_model", "DiTtest", 'diffuser_best.pt'),
                os.path.join(dir_name, "saved_model", "DiTtest", 'diffuser.pt'),
            ]
        model_path = next((path for path in model_candidates if os.path.exists(path)), None)
        if model_path is None:
            raise FileNotFoundError("No DiT checkpoint found. Checked: " + ", ".join(model_candidates))
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 初始化 DiT 模型
        self.model = DFUSER(
            dim_obs=model_param["state_dim"], 
            n_timesteps=model_param["n_timesteps"],
            model_choice=model_param["model_choice"],
            attn_block=model_param.get("attn_block", "causal"),
            predict_epsilon=model_param.get("predict_epsilon", True),
            traj_add_a=traj_add_a,
        )
        
        # 加载模型权重
        self.model.load_net(model_path, device=self.device)
        
        transition_dim = model_param["state_dim"] + 1 if traj_add_a else model_param["state_dim"]
        
        # Return Model 和 IDM (可选)
        self.use_RM = use_RM
        self.use_IDM = use_IDM
        
        if use_RM and Return_Model_path is not None:
            # 可以在这里添加 Return Model 的初始化
            pass
        
        if use_IDM and IDM_path is not None:
            # 可以在这里添加 IDM 的初始化
            pass
        
        # 状态维度
        self.state_dim = model_param["state_dim"]
        
        if traj_add_a:
            self.input = np.random.randn(48, self.state_dim + 2)
        else:
            self.input = np.random.randn(48, self.state_dim + 1)
        
        self.input = np.clip(self.input, -1., 1.)
        
        # CPA 条件归一化
        self.cpa_condition = torch.clamp(
            (torch.tensor(cpa, dtype=torch.float32) - 6) / (12 - 6), 
            min=0., max=1.
        )
        
        self.remaining_budget_last = self.budget
        
        # 归一化常数（基于完整数据集）
        self.states_min = np.array([
            2.08333333e-02, 2.31625624e-11, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            7.89913522e-04, 8.34000000e+02, 0.00000000e+00, 0.00000000e+00
        ])
        
        self.states_max = np.array([
            1.00000000e+00, 1.00000000e+00, 5.07260731e-01, 6.42348431e-01,
            3.22196518e-01, 1.30024173e-02, 1.32797358e-02, 9.86182965e-01,
            3.68157790e-01, 1.30024173e-02, 1.52702529e-02, 9.98711765e-01,
            1.34468593e-02, 3.17400000e+04, 8.96240000e+04, 4.94623000e+05
        ])
        
        self.R_min, self.R_max = 0., 1000.
        self.A_min, self.A_max = 0, 30
        
        self.selective_forward = selective_forward
        self.advertiser_id = advertiser_id
        self.traj_add_a = traj_add_a

    def reset(self):
        """重置预算和输入"""
        self.remaining_budget = self.budget
        self.input = np.zeros((48, self.state_dim + 1))

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        """
        DiT 出价策略
        
        使用因果 Diffusion Transformer 生成轨迹并预测出价
        """
        # 构建当前状态特征（与 DD 相同）
        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pValue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0
        historical_conversion_mean = np.mean(
            [np.mean(reward) for reward in history_conversion]
        ) if history_conversion else 0
        historical_LeastWinningCost_mean = np.mean(
            [np.mean(price) for price in historyLeastWinningCost]
        ) if historyLeastWinningCost else 0
        historical_pValues_mean = np.mean(
            [np.mean(value) for value in history_pValue]
        ) if history_pValue else 0
        historical_bid_mean = np.mean(
            [np.mean(bid) for bid in historyBid]
        ) if historyBid else 0

        def mean_of_last_n_elements(history, n):
            last_three_data = history[max(0, n - 3):n]
            if len(last_three_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_three_data])

        last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
        last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
        last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
        last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)

        historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
        last_three_pv_num_total = sum(
            [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]
        ) if historyBid else 0

        # 构建状态向量
        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_three_bid_mean,
            historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
            historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
            last_three_conversion_mean, last_three_xi_mean,
            current_pValues_mean, current_pv_num, last_three_pv_num_total,
            historical_pv_num_total
        ])

        # 归一化状态
        test_state_normalized = (test_state - self.states_min) / (self.states_max - self.states_min + 1e-8)
        test_state_normalized = np.clip(test_state_normalized, 0, 1)

        # 填充输入
        for i in range(self.state_dim):
            self.input[timeStepIndex, i] = test_state_normalized[i]
        
        # Return-to-go (使用 Score)
        self.input[:, -1] = self.cpa_condition.item()
        
        # 转换为 tensor
        x = torch.tensor(self.input.reshape(-1), device=self.device, dtype=torch.float32)
        
        # 准备 rtg (return-to-go) - 使用 CPA 作为目标
        rtg = torch.tensor([self.cpa_condition.item()], device=self.device, dtype=torch.float32)
        
        # DiT 前向推理 (需要传入 x 和 rtg 两个参数)
        actions, _ = self.model(x, rtg)
        
        # 获取第一个动作维度作为 alpha
        alpha = actions[0].item()
        alpha = max(0, alpha)  # 确保非负
        
        # 计算出价
        bids = alpha * pValues
        
        return bids
