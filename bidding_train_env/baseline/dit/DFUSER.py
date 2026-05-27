from torch.optim import Adam
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from .unet import TemporalUnet
import numpy as np


def extract(a, t, x_shape: list):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, 1, 1)


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


class WeightedStateLoss(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, targ, masks: torch.Tensor=None, weights=None):
        loss = self._loss(pred, targ)
        if masks is not None:
            loss = loss * masks[:, :, None].float()
        if weights is None:
            weighted_loss = loss.mean()
        else:
            weighted_loss = (loss * weights).mean()
        return weighted_loss, {'a_loss': None}


class WeightedStateL2(WeightedStateLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


Losses = {
    'state_l2': WeightedStateL2,
}


class GaussianInvDynDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=10,
                 clip_denoised=False, predict_epsilon=True, hidden_dim=256,
                 loss_discount=1.0, returns_condition=False,
                 condition_guidance_w=0.1,
                 use_noisy_condition=False,
                 action_max=30,
                 cond_obs_training=False,
                 pred_one_step=False,
                 traj_add_a=False,
                 ):
        super().__init__()

        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.cond_obs_training = cond_obs_training
        self.model = model
        self.pred_one_step = pred_one_step
        self.traj_add_a = traj_add_a

        self.inv_model = nn.Sequential(
            nn.Linear(4 * self.observation_dim, hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, self.action_dim),
        )
        self.action_max = action_max

        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses['state_l2'](loss_weights)

        # custom hyperparameters
        self.use_noisy_condition = use_noisy_condition

    def get_loss_weights(self, discount):

        self.action_weight = 1
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.matmul(discounts[:, None], dim_weights[None, :])

        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):

        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns: torch.Tensor = torch.ones(1, 1), aigb_base = False):
        if aigb_base: # whether use aigb baseline
            if self.returns_condition:
                # epsilon could be epsilon or x0 itself

                # epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
                epsilon_cond = self.model(x, t, returns, use_dropout=False)  # x already contains the cond (observed states)
                # epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
                epsilon_uncond = self.model(x, t, returns, force_dropout=True)  # x already contains the cond (observed states)

                epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
            else:
                epsilon = self.model(x, t)

        else:
            epsilon = self.model(x, t, returns, use_dropout=False, force_dropout=False)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        # if self.clip_denoised:
        #     x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, cond, t, returns: torch.Tensor = torch.ones(1, 1)):
        with torch.no_grad():
            b, _, _ = x.shape
            model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
            noise = 0.5 * torch.randn_like(x, device=x.device)
            nonzero_mask = (1 - (t == 0).float()).reshape(b, 1, 1)
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, shape, cond, returns: torch.Tensor = torch.ones(1, 1)):
        with torch.no_grad():
            torch.random.manual_seed(1019)
            batch_size = shape[0]
            x = 0.5 * torch.randn(shape[0], shape[1], shape[2], device=cond.device)
            # x = torch.randn(shape[0], shape[1], shape[2], device=cond.device)

            x = self.apply_conditioning(x, cond, action_dim=0, time_step=self.n_timesteps)

            for i in range(self.n_timesteps - 1, -1, -1):
                timesteps = torch.ones(batch_size,
                                       device=cond.device) * i
                x = self.p_sample(x, cond, timesteps, returns)

                x = self.apply_conditioning(x, cond, action_dim=0, time_step=i)

            return x

    def apply_conditioning(self, x, conditions, action_dim: int, time_step:int):
        if self.use_noisy_condition:
            # apply noise to the condition --> no use
            if time_step > 0:
                conditions = conditions.unsqueeze(0)
                noise = torch.randn_like(conditions, device=conditions.device)
                time_step = torch.randint(time_step-1, time_step, (conditions.shape[0],), device=conditions.device).long()
                conditions_noisy = self.q_sample(x_start=conditions, t=time_step, noise=noise)
                conditions_noisy = conditions_noisy.squeeze(0)
            else:
                conditions_noisy = conditions
        else:
            conditions_noisy = conditions
        
        if self.traj_add_a:
            action_dim = 1
            # x[:, :conditions_noisy.shape[-2], action_dim:] = conditions_noisy[:, :, action_dim:]  # 把 state_{0~t} 替换为真实观测值
            # x[:, :conditions_noisy.shape[-2]-1, 0] = conditions_noisy[:, :-1, 0]  # 把 a_{0~t-1} 替换为真实观测值
            
            # Step Back idea
            x[:, :conditions_noisy.shape[-2]-1, action_dim:] = conditions_noisy[:, :-1, action_dim:]  # 把 state_{0~t-1} 替换为真实观测值
            x[:, :conditions_noisy.shape[-2]-1, 0] = conditions_noisy[:, :-1, 0]  # 把 a_{0~t-1} 替换为真实观测值
        else:
            x[:, :conditions_noisy.shape[-2], action_dim:] = conditions_noisy  # 把 state_{0~t} 替换为真实观测值

        return x

    #  @torch.no_grad()
    def conditional_sample(self, cond, returns: torch.Tensor = torch.ones(1, 1), horizon: int = 48):
        with torch.no_grad():
            if len(cond.shape) == 2:
                batch_size = 1
                cond = cond.unsqueeze(0)
            else:
                batch_size = cond.shape[0]
            horizon = self.horizon
            if self.traj_add_a:
                shape = torch.tensor([batch_size, horizon, self.observation_dim+self.action_dim])
            else:
                shape = torch.tensor([batch_size, horizon, self.observation_dim])
            return self.p_sample_loop(shape, cond, returns)

    def forward(self, cond, returns):
        x_0 = self.conditional_sample(cond=cond, returns=returns)
        return x_0
    
    def selective_forward(self, cond, returns, advertiser_id=999, budget_left_dim=1, draw_fig=True, Return_Model=None):
        MAX_SAMPLING_TIMES = 10
        # repeat多个cond以便并行计算
        cond = cond.unsqueeze(0)
        cond_repeat = cond.repeat(MAX_SAMPLING_TIMES, 1, 1)
        returns_repeat = returns.repeat(MAX_SAMPLING_TIMES, 1)
        x_0_generations = self.conditional_sample(cond=cond_repeat, returns=returns_repeat)

        cur_time = cond.shape[-2] -1  # t start from 0
        next_time = cond.shape[-2]

        if next_time == 48:  # bidding 已到最后一步
            selected_x_0 = x_0_generations[0]
        else:
            if Return_Model is not None:

                # # do gradient update
                # x = x_0_generations[0].unsqueeze(0)
                # x.requires_grad_()
                # pred_return = Return_Model(x)
                # grad = torch.autograd.grad([pred_return.sum()], [x])[0]
                # x.detach()
                # return x + 0.01 * grad



                pred_returns = Return_Model(x_0_generations).view(-1).cpu()
                # 过滤出介于 0 和 1 之间的值的索引
                valid_returns_indices = (pred_returns >= 0) & (pred_returns <= 1)

                # 找到这些值中的最大值及其索引
                try: 
                    max_return, max_index = pred_returns[valid_returns_indices].max(dim=0)

                    # 获取在原始展平张量中的索引
                    selected_i = torch.arange(pred_returns.size(0))[valid_returns_indices][max_index]
                except:
                    selected_i = 0
                pass

            else: 
                # Step Back的思想：利用s_t和生成的s_t的gap来筛选轨迹
                s_t_gt = cond[:, cur_time, 1:]
                s_t_generated = x_0_generations[:, cur_time, 1:]
                gap = ((s_t_gt - s_t_generated)**2).sum(dim=-1)
                selected_i = torch.argmin(gap)

                # # 设置一些业务的逻辑来做筛选
                # # pick out the most causal trajectory based on budget_left
                # budget_left_t = x_0_generations[:, cur_time, budget_left_dim]
                # budget_left_tp1 = x_0_generations[:, next_time, budget_left_dim]
                # budget_change = budget_left_t - budget_left_tp1

                # # 过滤出大于 0 的元素及其对应的下标
                # positive_indices = torch.nonzero(budget_change > 0).squeeze()
                # positive_values = budget_change[positive_indices]
                
                # try:
                #     # 对大于 0 的元素进行排序
                #     sorted_indices = torch.argsort(positive_values)
                #     sorted_positive_values = positive_values[sorted_indices]

                #     # 找出中间数
                #     middle_index = len(sorted_positive_values) // 2
                #     middle_value = sorted_positive_values[middle_index]

                #     # 找出中间数在原始张量中的下标
                #     selected_i = positive_indices[sorted_indices[middle_index]]
                # except:  # 如果没有任何一条符合causal
                #     selected_i = torch.argmax(budget_change)

            selected_x_0 = x_0_generations[selected_i]

        selected_x_0 = selected_x_0.unsqueeze(0)
        assert len(selected_x_0.shape) == 3

        # if draw_fig:
        #     plt.figure(figsize=(12, 8))
        #     if self.traj_add_a: 
        #         budget_left_dim = 2
        #     else:
        #         budget_left_dim = 1
        #     plt.plot(range(cond.shape[-2]), cond[0, :, budget_left_dim].cpu(), color='blue', alpha=0.5)
        #     plt.plot(range(cond.shape[-2]-1, 48), selected_x_0[0, cond.shape[-2]-1:, budget_left_dim].cpu(), color='red', alpha=0.5)
        #     # 添加标题和标签
        #     plt.title('Causality of generated budget_left')
        #     plt.xlabel(f'Time Steps (steps 0-{cond.shape[-2]-1} are observed gt and steps {cond.shape[-2]}-47 are generated)')
        #     plt.ylabel('Budget_left')
        #     plt.savefig(os.path.join('logs/inference', f'budget_left_adID_{advertiser_id}_step_{cond.shape[-2]}_sDim_{self.horizon}_oneStep_{self.pred_one_step}_condObs_{self.cond_obs_training}_pred_noise_{self.predict_epsilon}.png'))
        #     plt.clf()

        return selected_x_0


    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)

        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(t.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(t.device)
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None, masks=None):
        noise = torch.randn_like(x_start, device=x_start.device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        t = t.to(x_noisy.device)

        """ One Core Trick: conditional_training"""
        # >>>>>>>>>
        if self.cond_obs_training:
            observed_period = torch.randint(1, self.horizon, (1,), device=x_start.device).long() 
            if self.traj_add_a:
                # mask the a_t
                a_t_target = x_start[:, observed_period-1, 0]
                # x_start[:, observed_period-1, 0] = x_noisy[:, observed_period-1, 0]
                # 尝试新的做法：只condition在[s_{0:t-1}, a_{0:t-1}]，让diffuser生成[s_{t:T}, a_{t:T}]以克服a_t预测不准的奇怪现象，
                # 同时预测的s_t和其ground_truth做gap可以筛选出最causal的轨迹，同时，哪怕再差的diffuser，其生成的轨迹内部都具有不错的连续性，尽管其与观测的部分存在不causal的情形
                x_noisy = torch.cat((x_start[:, :observed_period-1, :], x_noisy[:, observed_period-1:, :]), dim=1)
            else:
                x_noisy = torch.cat((x_start[:, :observed_period, :], x_noisy[:, observed_period:, :]), dim=1)

            x_recon = self.model(x_noisy, t, returns) 
        else:
            observed_period = 0
            x_noisy = x_noisy
            x_recon = self.model(x_noisy, t, returns) 
        # <<<<<<<<

        if self.pred_one_step:
            """极端情形，只预测下一步state"""
            loss, info = self.loss_fn(x_recon[:, observed_period:observed_period+1, :], x_start[:, observed_period:observed_period+1, :], masks[:, observed_period:observed_period+1])
        else:
            if self.predict_epsilon:
                loss, info = self.loss_fn(x_recon[:, observed_period:, :], noise[:, observed_period:, :], masks[:, observed_period:])
            else:
                # loss, info = self.loss_fn(x_recon[:, observed_period-1:, :], x_start[:, observed_period-1:, :])  # 含 s_t, a_t
                loss, info = self.loss_fn(x_recon[:, observed_period:, :], x_start[:, observed_period:, :], masks[:, observed_period:])  # 不含 s_t, a_t
                if self.traj_add_a:
                    a_t_pred = x_recon[:, observed_period-1, 0]
                    pred_a_loss = ((a_t_pred - a_t_target)**2).mean()
                    # loss += pred_a_loss * (self.observation_dim/self.action_dim)
                    info["a_loss"] = pred_a_loss
                    
        return loss, info

    def loss(self, x, cond, returns, masks, traj_add_a=False):

        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()

        if self.traj_add_a:
            diffuse_loss, info = self.p_losses(x[:, :, :], cond, t, returns, masks)
        else:
            diffuse_loss, info = self.p_losses(x[:, :, self.action_dim:], cond, t, returns, masks)
        # Calculating inv loss
        x_t = x[:, :-1, self.action_dim:]
        a_t = x[:, :-1, :self.action_dim]
        x_t_1 = x[:, 1:, self.action_dim:]
        x_t_2 = torch.cat(
            [torch.zeros(x.shape[0], 1, x.shape[-1] - self.action_dim, device=x.device), x[:, :-2, self.action_dim:]],
            dim=1)
        x_t_3 = torch.cat(
            [torch.zeros(x.shape[0], 2, x.shape[-1] - self.action_dim, device=x.device), x[:, :-3, self.action_dim:]],
            dim=1)
        # x_comb_t = torch.cat([x_t_2, x_t_3, x_t, x_t_1], dim=-1)
        # revised: every obeservation x_comb_t (dim=64) := [s_{t-2}, s_{t-1}, s_t, s_{t+1}], s_{t-2} --> x_t_3, s_{t-1} --> x_t_2, s_{t+1} --> x_t_1
        x_comb_t = torch.cat([x_t_3, x_t_2, x_t, x_t_1], dim=-1)  
        x_comb_t = x_comb_t.reshape(-1, 4 * self.observation_dim)
        masks_flat = masks[:, :-1].reshape(-1)
        x_comb_t = x_comb_t[masks_flat]
        a_t = a_t.reshape(-1, self.action_dim)
        a_t = a_t[masks_flat]
        pred_a_t = self.inv_model(x_comb_t)
        inv_loss = F.mse_loss(pred_a_t, a_t)
        loss = (1 / 2) * (diffuse_loss + inv_loss)

        return loss, info, (diffuse_loss, inv_loss)


class DFUSER(nn.Module):
    def __init__(self, dim_obs=9, dim_actions=1, gamma=1, tau=0.01, lr=1e-4,
                 network_random_seed=200,
                 ACTION_MAX=30, ACTION_MIN=0,
                 step_len=48, n_timesteps=10,
                 use_noisy_condition=False,
                 model_choice='Unet',
                 attn_block='vanilla',
                 predict_epsilon=False,
                 cond_obs_training=True,
                 pred_one_step=False,
                 traj_add_a=False):

        super().__init__()

        self.n_timestamps = n_timesteps
        self.dim_obs = dim_obs

        if traj_add_a:
            self.transition_dim = dim_obs + dim_actions
        else:
            self.transition_dim = dim_obs

        self.num_of_actions = dim_actions
        self.ACTION_MAX = ACTION_MAX
        self.ACTION_MIN = ACTION_MIN
        self.network_random_seed = network_random_seed
        self.step_len = step_len
        self.model_choice = model_choice
        self.attn_block = attn_block
        self.predict_epsilon = predict_epsilon
        self.cond_obs_training = cond_obs_training
        self.pred_one_step = pred_one_step
        self.traj_add_a = traj_add_a

        if model_choice == 'Unet':
            model = TemporalUnet(
                horizon=step_len,
                transition_dim=self.transition_dim,
                cond_dim=dim_actions,
                returns_condition=True,
                dim=128,
                condition_dropout=0.25,
                calc_energy=False
            ).to('cpu')
        if model_choice == 'DiT1d':
            from .dit import DiT1d
            model = DiT1d(in_dim=self.transition_dim, emb_dim=256, d_model=256, n_heads=8, depth=12, dropout=0.1, attn_block=attn_block).to('cpu')

        self.diffuser = GaussianInvDynDiffusion(
            model=model,
            horizon=step_len,
            observation_dim=dim_obs,
            action_dim=dim_actions,
            clip_denoised=True,
            predict_epsilon=predict_epsilon,
            hidden_dim=256,
            n_timesteps=n_timesteps,
            loss_discount=1,
            returns_condition=True,
            condition_guidance_w=1.2,
            use_noisy_condition=use_noisy_condition,
            action_max=self.ACTION_MAX,
            cond_obs_training=cond_obs_training,
            pred_one_step=pred_one_step,
            traj_add_a=traj_add_a
        )

        self.step = 0

        torch.random.manual_seed(network_random_seed)

        self.num_of_episodes = 0

        self.GAMMA = gamma
        self.tau = tau
        self.num_of_steps = 0
        # cuda usage
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.diffuser.cuda()

        self.diffuser_lr = lr

        self.diffuserModel_optimizer = torch.optim.Adam(self.diffuser.model.parameters(), lr=lr)
        self.invModel_optimizer = torch.optim.Adam(self.diffuser.inv_model.parameters(), lr=lr)

    def toCuda(self):
        self.diffuser.cuda()

    def trainStep(self, states, actions, returns, masks):
        self.diffuser.train()
        if self.use_cuda:
            self.diffuser.cuda()
            states = states.cuda()
            actions = actions.cuda()
            returns = returns.cuda()
            masks = masks.cuda()

        x = torch.cat([actions, states], dim=-1)
        # x = torch.cat([actions, states], dim=-1)

        # 随机conditional 在loss中实现了 此处cond无效
        cond = torch.ones_like(states[:, 0], device=states.device)[:, None, :]

        loss, infos, (diffuse_loss, inv_loss) = self.diffuser.loss(x, cond, returns=returns, masks=masks, traj_add_a=self.traj_add_a)
        inv_loss.backward()
        self.invModel_optimizer.step()
        self.invModel_optimizer.zero_grad()

        diffuse_loss.backward()
        self.diffuserModel_optimizer.step()
        self.diffuserModel_optimizer.zero_grad()

        return loss, (diffuse_loss, inv_loss), infos


    def forward(self, x: torch.Tensor, rtg:torch.Tensor, cpa: torch.Tensor=None, selective_forward: bool=False, advertiser_id=999, Return_Model=None):
        if len(list(x.shape)) < 2:
            x = torch.reshape(x, [48, self.transition_dim + 1])  # 最后一个维度是time_index, 暂时没有用上，也没有必要用上
        else:
            x = x[0][0]
        cur_time = int(x[0][-1].item()) # time_index从0开始计数
        cur_time = cur_time + 1
        states = x[:cur_time]
        states = states[:, :-1]
        conditions = states
        if rtg is None:
            returns = torch.tensor([[1.0]], device=x.device)
        else:
            returns = torch.tensor([[rtg.item()]], device=x.device)
        # returns = torch.tensor([[1.0, cpa]], device=x.device)
        if selective_forward:
            if Return_Model is not None:
                x_0 = self.diffuser.selective_forward(cond=conditions, returns=returns, advertiser_id=advertiser_id, Return_Model=Return_Model)
            else:
                x_0 = self.diffuser.selective_forward(cond=conditions, returns=returns, advertiser_id=advertiser_id)
        else:
            x_0 = self.diffuser(cond=conditions, returns=returns)
        
        if self.traj_add_a:
        #     a_t = x_0[0, cur_time-1, 0]
        #     return a_t
            action_dim = 1
        else:
            action_dim = 0
        states = x_0[0, :cur_time + 1, action_dim:]
        states_next = states[None, -1]
        if cur_time > 1:
            states_curt1 = conditions[-2].float()[None, action_dim:]
        else:
            states_curt1 = torch.zeros_like(states_next, device=states_next.device)
        if cur_time > 2:
            states_curt2 = conditions[-3].float()[None, action_dim:]
        else:
            states_curt2 = torch.zeros_like(states_next, device=states_next.device)
        states_comb = torch.hstack([states_curt2, states_curt1, conditions[-1].float()[None, action_dim:], states_next])
        actions = self.diffuser.inv_model(states_comb)
        # actions = torch.clamp(actions, min=self.ACTION_MIN, max=self.ACTION_MAX)
        actions = actions.detach().cpu()[0]  # .cpu().data.numpy()
        return actions, states_next
        
    def test_generalization(self, states: torch.Tensor, returns: torch.Tensor, save_path='logs/'):
        import matplotlib.pyplot as plt

        conditions_half = states[:,:states.shape[1]//2].reshape(states.shape[0], states.shape[1]//2, -1) 
        x_0 = self.diffuser(cond=conditions_half, returns=returns)
        # 1. reconstruction error test
        recon_error = (abs(x_0 - states)).mean()

        # 2.visualizing the causality via time_left and budget_left
        if self.traj_add_a:
            generated_action = x_0 [:10, :, 0]
            generated_time_left = x_0[:10, :, 1]
            generated_budget_left = x_0[:10, :, 2]
            gt_action = states[:10, :, 0]
            gt_time_left = states[:10, :, 1]
            gt_budget_left = states[:10,:, 2]
        else:
            generated_time_left = x_0[:10, :, 0]
            generated_budget_left = x_0[:10, :, 1]
            gt_time_left = states[:10, :, 0]
            gt_budget_left = states[:10,:, 1]

        generated_time_left = generated_time_left.cpu().numpy()
        generated_budget_left = generated_budget_left.cpu().numpy()
        gt_time_left = gt_time_left.cpu().numpy()
        gt_budget_left = gt_budget_left.cpu().numpy()
        

        plt.figure(figsize=(12, 8))
        for i in range(generated_time_left.shape[0]):
            plt.plot(range(48), gt_time_left[i, :], color='blue', alpha=0.5)
            plt.plot(range(23, 48),generated_time_left[i, 23:], color='red', alpha=0.5)

        # 添加标题和标签
        plt.title('Causality of generated time_left')
        plt.xlabel('Time Steps (steps 0-23 are observed gt and steps 24-47 are generated)')
        plt.ylabel('Time Left')
        plt.savefig(os.path.join(save_path, f'sDim_{self.dim_obs}_TransDim{self.transition_dim}_oneStep_{self.pred_one_step}_condObs_{self.cond_obs_training}_causality_of_generated_time_left_{self.model_choice}_{self.attn_block}_pred_noise_{self.predict_epsilon}.png'))
        plt.clf()

        plt.figure(figsize=(12, 8))
        for i in range(generated_time_left.shape[0]):
            if self.pred_one_step:
                plt.plot(range(24), gt_budget_left[i, :24], color='blue', alpha=0.5)
                plt.plot(range(23, 25),generated_budget_left[i, 23:25], color='red', alpha=0.5)
            else:
                plt.plot(range(48), gt_budget_left[i, :], color='blue', alpha=0.5)
                plt.plot(range(23, 48),generated_budget_left[i, 23:], color='red', alpha=0.5)


        # 添加标题和标签
        plt.title('Causality of generated budget_left')
        plt.xlabel('Time Steps (steps 0-23 are observed gt and steps 24-47 are generated)')
        plt.ylabel('Budget_left')
        plt.savefig(os.path.join(save_path, f'sDim_{self.dim_obs}_TransDim{self.transition_dim}_oneStep_{self.pred_one_step}_condObs_{self.cond_obs_training}_causality_of_generated_budget_left_{self.model_choice}_{self.attn_block}_pred_noise_{self.predict_epsilon}.png'))
        plt.clf()

        if self.traj_add_a:
            generated_action = generated_action.cpu().numpy()
            gt_action = gt_action.cpu().numpy()
            
            plt.figure(figsize=(12, 8))
            for i in range(generated_action.shape[0]):
                if self.pred_one_step:
                    plt.plot(range(24), gt_action[i, :24], color='blue', alpha=0.5)
                    plt.plot(range(23, 25),generated_action[i, 23:25], color='red', alpha=0.5)
                else:
                    plt.plot(range(48), gt_action[i, :], color='blue', alpha=0.5)
                    plt.plot(range(23, 48),generated_action[i, 23:], color='red', alpha=0.5)


            # 添加标题和标签
            plt.title('Causality of generated action')
            plt.xlabel('Time Steps (steps 0-23 are observed gt and steps 24-47 are generated)')
            plt.ylabel('action')
            plt.savefig(os.path.join(save_path, f'sDim_{self.dim_obs}_TransDim{self.transition_dim}_oneStep_{self.pred_one_step}_condObs_{self.cond_obs_training}_causality_of_generated_ACTION_{self.model_choice}_{self.attn_block}_pred_noise_{self.predict_epsilon}.png'))
            plt.clf()

        return recon_error

    def save_net(self, save_path, save_name=''):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(self.diffuser.state_dict(), f'{save_path}/diffuser{save_name}.pt')

    def save_model(self, save_path, epi):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        model_temp = self.cpu()
        jit_model = torch.jit.script(model_temp)
        torch.jit.save(jit_model, f'{save_path}/diffuser_{epi}.pth')

    def load_net(self, load_path="saved_model/fixed_initial_budget", device='cuda:0'):
        self.diffuser.load_state_dict(torch.load(load_path, map_location='cpu'))
        self.optimizer = Adam(self.diffuser.parameters(), lr=self.diffuser_lr)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.diffuser.cuda()
