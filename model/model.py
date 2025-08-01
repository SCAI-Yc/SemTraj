import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Attention, Mlp
import math
import torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

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
    
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    Here we don't have explicit labels, thus head info is used.
    """
    def __init__(self, embedding_dim=128, hidden_dim=256, dropout_prob=0.1):
        super().__init__()
        # use_cfg_emedding = dropout_prob > 0
        # self.dropout_prob = dropout_prob

        # # Wide part (linear model for continuous attributes)
        # self.wide_fc = nn.Linear(5, embedding_dim)
        # self.wide_fc = nn.Linear(3, embedding_dim)

        # Deep part (neural network for categorical attributes)
        self.depature_embedding = nn.Embedding(288, hidden_dim)
        self.sid_embedding = nn.Embedding(257, hidden_dim)
        self.eid_embedding = nn.Embedding(257, hidden_dim)
        self.deep_fc1 = nn.Linear(hidden_dim*3, embedding_dim)
        self.deep_fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, attr):
        # Categorical attributes
        depature, sid, eid = attr[:, 0].long(
        ), attr[:, 1].long(), attr[:, 2].long()

        # Deep part
        depature_embed = self.depature_embedding(depature)
        sid_embed = self.sid_embedding(sid)
        eid_embed = self.eid_embedding(eid)
        categorical_embed = torch.cat(
            (depature_embed, sid_embed, eid_embed), dim=1)
        deep_out = F.relu(self.deep_fc1(categorical_embed))
        deep_out = self.deep_fc2(deep_out)

        return deep_out

class TDFormerBlock(nn.Module):
    """
    A TDFormer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, head_nums, mlp_ratio=4.0, **block_kwargs) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(hidden_size, num_heads=head_nums, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self_made_gelu = GELU()
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        approx_gelu = lambda: self_made_gelu
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attention(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of SemTraj.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
class SemTraj(nn.Module):
    """
    Diffusion model with a TDFormer backbone for trajectories.
    """
    def __init__(
            self,
            in_channels=2,
            hidden_size=128,
            depth=18,
            head_nums=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            traj_length=200,
            learn_sigma=True,
            guidance_scale=3
        ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.head_nums = head_nums
        self.guidance_scale = guidance_scale

        self.num_patches = traj_length
        
        self.conv_in = torch.nn.Conv1d(in_channels,
                                       out_channels=hidden_size,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder()
        
        num_patches = traj_length
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            TDFormerBlock(hidden_size=hidden_size, head_nums=head_nums, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size=1, out_channels=self.out_channels)

        self.init_weights()

    def init_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding
        init_pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.array([i for i in range(self.num_patches)]))
        self.pos_embed.data.copy_(torch.from_numpy(init_pos_embed).float().unsqueeze(0))

        # Initialize label(head) embedding table:
        nn.init.normal_(self.y_embedder.depature_embedding.weight, std=0.02)
        nn.init.normal_(self.y_embedder.sid_embedding.weight, std=0.02)
        nn.init.normal_(self.y_embedder.eid_embedding.weight, std=0.02)
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y):
        """
        Forward pass of SemTraj.
        x: (N, T, C) tensor of spatial inputs (trajectory or latent representations of trajectory)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of heads
        """
        x_backup, t_backup = x, t
        # cond_noise = self.forward(x, t, y)
        x = self.conv_in(x).flatten(2).transpose(1, 2) + self.pos_embed     # (N, T, D) => (batch_size, timesteps, hidden_dim)
        t = self.t_embedder(t)                                              # (N, D)
        y = self.y_embedder(y)                                              # (N, D)
        c = t + y
        for block in self.blocks:
            x = block(x, c)                                                 # (N, T, D)
        x = self.final_layer(x, c)                                          # (N, T, C)
        x = x.transpose(1, 2)                                               # (N, C, T)
        cond_noise = x

        place_vector = torch.zeros(y.shape, device=y.device)
        x, t, y = x_backup, t_backup, place_vector
        x = self.conv_in(x).flatten(2).transpose(1, 2) + self.pos_embed     # (N, T, D) => (batch_size, timesteps, hidden_dim)
        t = self.t_embedder(t)                                              # (N, D)
        y = self.y_embedder(y)                                              # (N, D)
        c = t + y
        for block in self.blocks:
            x = block(x, c)                                                 # (N, T, D)
        x = self.final_layer(x, c)                                          # (N, T, C)
        x = x.transpose(1, 2)                                               # (N, C, T)
        uncond_noise = x
        
        pred_noise = cond_noise + self.guidance_scale * (cond_noise -
                                                         uncond_noise)
        return pred_noise

    
    def forward_with_cfg(self, x, t, y):
        """
        Forward pass of SemTraj, but also batches the unconditional forward pass for classifier-free guidance.
        """

        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


if __name__ == '__main__':
    from diffusion import create_diffusion

    model = SemTraj(
        in_channels=2,
        hidden_size=128,
        depth=18,
        head_nums=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        traj_length=400,
        learn_sigma=True
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')