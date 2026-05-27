from typing import Optional

import torch
import torch.nn as nn

from .dit_utils import SinusoidalEmbedding
from .dit_base_diffusion import BaseNNDiffusion
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# class Attention(nn.Module):
#     fused_attn: Final[bool]
#     def __init__(
#             self,
#             dim: int,
#             num_heads: int = 8,
#             qkv_bias: bool = False,
#             qk_norm: bool = False,
#             attn_drop: float = 0.,
#             proj_drop: float = 0.,
#             norm_layer: nn.Module = nn.LayerNorm,
#     ) -> None:
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.fused_attn = use_fused_attn()

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)
#         q, k = self.q_norm(q), self.k_norm(k)

#         if self.fused_attn:
#             x = F.scaled_dot_product_attention(
#                 q, k, v,
#                 dropout_p=self.attn_drop.p if self.training else 0.,
#             )
#         else:
#             q = q * self.scale
#             attn = q @ k.transpose(-2, -1)
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x = attn @ v

#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


class CausalMultiheadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, attn_drop=0.1, resid_drop=0.1, n_ctx=1024):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)

        # 1*1*n_ctx*n_ctx
        # 下三角矩阵，下面全1
        self.register_buffer("bias",
                             torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
                                                                                          
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.proj = nn.Linear(hidden_size, hidden_size)
        self.n_head = n_heads

    def forward(self, x, mask): # batch*(seq*3)*dim, batch*(seq*3)
        B, T, C = x.size() # T=seq*3, C=dim

        # batch*n_head*T*C // self.n_head
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        mask = mask.view(B, -1)
        # batch*1*1*(seq*3)
        mask = mask[:, None, None, :]
        # 1->0, 0->-10000
        mask = (1.0 - mask) * -10000.0
        # batch*n_head*T*T
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.where(self.bias[:, :, :T, :T].bool(), att, self.masked_bias.to(att.dtype))
        att = att + mask
        att = F.softmax(att, dim=-1)
        self._attn_map = att.clone()
        att = self.attn_drop(att)
        # batch*n_head*T*C // self.n_head
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class DiTBlock(nn.Module):
    """ A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning. """

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)
        # self.attn = CausalMultiheadAttention(hidden_size, n_heads, attn_drop=dropout, resid_drop=dropout)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        def approx_gelu(): return nn.GELU(approximate="tanh")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), approx_gelu(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6, bias=True))

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = self.query(x), self.key(x), self.value(x)
        # x = x + gate_msa.unsqueeze(1) * self.attn(x, x, x)[0]  # <-- this implementation from the cleandiffuser is wrong!
        x = x + gate_msa.unsqueeze(1) * self.attn(q, k, v)[0] 
        # x = x + gate_msa.unsqueeze(1) * self.attn(x)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class CausalDiTBlock(nn.Module):
    """ A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning. """

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # self.query = nn.Linear(hidden_size, hidden_size)
        # self.key = nn.Linear(hidden_size, hidden_size)
        # self.value = nn.Linear(hidden_size, hidden_size)
        # self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)
        self.attn = CausalMultiheadAttention(hidden_size, n_heads, attn_drop=dropout, resid_drop=dropout)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        def approx_gelu(): return nn.GELU(approximate="tanh")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), approx_gelu(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6, bias=True))

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        # q, k, v = self.query(x), self.key(x), self.value(x)
        # x = x + gate_msa.unsqueeze(1) * self.attn(x, x, x)[0]  # <-- this implementation from the cleandiffuser is wrong!
        mask = torch.ones(x.shape[0], x.shape[1]).to(x.device)  # 全 1 mask，因为diffusion中所有位置都有数值
        x = x + gate_msa.unsqueeze(1) * self.attn(x, mask)[0] 
        # x = x + gate_msa.unsqueeze(1) * self.attn(x)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer1d(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)



class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiT1d(BaseNNDiffusion):
    def __init__(
        self,
        in_dim: int,
        emb_dim: int,
        d_model: int = 256,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
        attn_block: str = 'causal',
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.in_dim, self.emb_dim = in_dim, emb_dim
        self.d_model = d_model

        self.x_proj = nn.Linear(in_dim, d_model)
        self.pos_emb = SinusoidalEmbedding(d_model)
        self.pos_emb_cache = None

        """ 更正 cleandiffuser 的 code，使其和 DiT original code 一致 """
        self.t_embedder = TimestepEmbedder(emb_dim)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)  # y在这里是return
        self.map_return = nn.Sequential(torch.nn.Linear(1, emb_dim),
                                        nn.SiLU(),
                                        nn.Linear(emb_dim, emb_dim))

        self.map_emb = nn.Sequential(
            nn.Linear(emb_dim*2, d_model), nn.Mish(), nn.Linear(d_model, d_model), nn.Mish())

        if attn_block == 'causal':
            self.blocks = nn.ModuleList([
                CausalDiTBlock(d_model, n_heads, dropout) for _ in range(depth)])
        if attn_block == 'vanilla':
            self.blocks = nn.ModuleList([
                DiTBlock(d_model, n_heads, dropout) for _ in range(depth)])
            
        self.final_layer = FinalLayer1d(d_model, in_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize time step embedding MLP:
        nn.init.normal_(self.map_emb[0].weight, std=0.02)
        nn.init.normal_(self.map_emb[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self,
                x: torch.Tensor, t_noise: torch.Tensor,
                c_return: Optional[torch.Tensor] = None,
                use_dropout: bool = True,
                force_dropout: bool = False):
        """
        Input:
            x:          (b, horizon, in_dim)  --> noisy_traj
            t_noise:    (b, )   --> the noise for time step t's diffusion 
            c_return:   (b, )   --> the return to go as a condition

        Output:
            y:          (b, horizon, in_dim)  --> pred_noise or pred_recon
        """

        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
            self.pos_emb_cache = self.pos_emb(torch.arange(x.shape[1], device=x.device))

        x = self.x_proj(x) + self.pos_emb_cache[None,]
        # t embedding method 1
        # t_emb = self.map_noise(t_noise)  
        # t embedding method 2 from official DiT code
        t_emb = self.t_embedder(t_noise)

        # TODO: implement the classfier-free guidance here by randomly dropout
        c_emb = self.map_return(c_return)

        emb = self.map_emb(torch.cat((t_emb, c_emb), dim=-1))

        # if c_return is not None:
        #     emb = emb + condition
        # else:
        #     emb = emb + torch.zeros_like(emb)
        # emb = self.map_emb(emb)


        for block in self.blocks:
            x = block(x, emb)
        x = self.final_layer(x, emb)
        return x
