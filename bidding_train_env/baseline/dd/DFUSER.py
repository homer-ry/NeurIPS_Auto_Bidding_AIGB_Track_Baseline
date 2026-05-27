"""
Yewen ICML_DiT_bid DD/CAD diffuser adapted for this project.

The source implementation under ``yewen/ICML_DiT_bid/bidding_train_env/baseline/dd``
shares the same backbone files as this repository's ``baseline/dit`` package.
Reuse that adapted implementation here so DD, CBD and DiT do not drift.
"""

from bidding_train_env.baseline.dit.DFUSER import (
    DFUSER as _YewenDFUSER,
    GaussianInvDynDiffusion,
    Losses,
    WeightedStateL2,
    WeightedStateLoss,
    cosine_beta_schedule,
    extract,
)
import torch
from torch.optim import Adam


class DFUSER(_YewenDFUSER):
    """
    DD-compatible entrypoint for the yewen/CAD diffuser.

    Defaults are chosen for this AIGB project: 16-dimensional states and U-Net
    backbone.  The parent implementation provides yewen's conditional training,
    optional DiT backbone, optional action-in-trajectory mode, and selective
    forward support.
    """

    def __init__(
        self,
        dim_obs=16,
        dim_actions=1,
        gamma=1,
        tau=0.01,
        lr=1e-4,
        network_random_seed=200,
        ACTION_MAX=30,
        ACTION_MIN=0,
        step_len=48,
        n_timesteps=10,
        use_noisy_condition=False,
        model_choice="Unet",
        attn_block="vanilla",
        predict_epsilon=False,
        cond_obs_training=False,
        pred_one_step=False,
        traj_add_a=False,
    ):
        super().__init__(
            dim_obs=dim_obs,
            dim_actions=dim_actions,
            gamma=gamma,
            tau=tau,
            lr=lr,
            network_random_seed=network_random_seed,
            ACTION_MAX=ACTION_MAX,
            ACTION_MIN=ACTION_MIN,
            step_len=step_len,
            n_timesteps=n_timesteps,
            use_noisy_condition=use_noisy_condition,
            model_choice=model_choice,
            attn_block=attn_block,
            predict_epsilon=predict_epsilon,
            cond_obs_training=cond_obs_training,
            pred_one_step=pred_one_step,
            traj_add_a=traj_add_a,
        )
        self.num_of_states = self.dim_obs

    def forward(
        self,
        x,
        rtg=None,
        cpa=None,
        selective_forward=False,
        advertiser_id=999,
        Return_Model=None,
    ):
        if hasattr(x, "dim") and x.dim() == 2:
            x = x.reshape(-1)
        result = super().forward(
            x,
            rtg,
            cpa=cpa,
            selective_forward=selective_forward,
            advertiser_id=advertiser_id,
            Return_Model=Return_Model,
        )
        if rtg is None and isinstance(result, tuple):
            return result[0]
        return result

    def load_net(self, load_path="saved_model/fixed_initial_budget", device="cuda:0"):
        checkpoint = torch.load(load_path, map_location="cpu")
        current_state = self.diffuser.state_dict()
        compatible_state = {}
        skipped_keys = []
        for key, value in checkpoint.items():
            if key in current_state and current_state[key].shape == value.shape:
                compatible_state[key] = value
            else:
                skipped_keys.append(key)

        current_state.update(compatible_state)
        self.diffuser.load_state_dict(current_state)
        if skipped_keys:
            print(
                "Partially loaded DD checkpoint: "
                f"{len(compatible_state)} tensors loaded, {len(skipped_keys)} skipped due to shape/name mismatch."
            )
        self.optimizer = Adam(self.diffuser.parameters(), lr=self.diffuser_lr)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.diffuser.cuda()
