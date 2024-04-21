# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Mlp

try:
    import xformers.ops as xops
except ImportError:
    pass

try:
    from flash_attn.flash_attention import FlashMHA

    flash_attn_available = True
except ImportError:
    flash_attn_available = False


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DynamicPositionBias(nn.Module):
    """From https://github.com/lucidrains/x-transformers/"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            depth: int = 2,
            log_distance: bool = False,
            norm: bool = False,
    ):
        super().__init__()
        assert depth >= 1, "depth for dynamic position bias MLP must be >= 1"
        self.log_distance = log_distance

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, dim),
                    nn.LayerNorm(dim) if norm else nn.Identity(),
                    nn.ReLU(),
                )
            ]
        )

        for _ in range(depth - 1):
            self.mlp.append(
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim) if norm else nn.Identity(),
                    nn.ReLU(),
                )
            )

        self.mlp.append(nn.Linear(dim, num_heads))

    def forward(self, qk_dots):
        n, device, dtype = qk_dots.shape[-1], qk_dots.device, qk_dots.dtype

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device=device)
        ctx_arange = torch.arange(n, device=device)
        indices = rearrange(seq_arange, "i -> i 1") - rearrange(ctx_arange, "j -> 1 j")
        indices += n - 1

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device=device, dtype=dtype)
        pos = rearrange(pos, "... -> ... 1")

        if self.log_distance:
            # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases
        bias = pos[indices]
        bias = rearrange(bias, "i j h -> h i j")
        return qk_dots + bias


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos=None, alpha=0.5, distance_matrix=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if rel_pos is not None:
            attn = rel_pos(attn)
        attn = attn.softmax(dim=-1)
        if distance_matrix is not None:
            if isinstance(alpha, float):
                attn = alpha * distance_matrix + (1 - alpha) * attn
            else:
                attn = alpha[:, None, None, None] * distance_matrix + (1 - alpha[:, None, None, None]) * attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_xformers_flash_attention_op(q, k, v, attn_bias=None):
    try:
        flash_attention_op = xops.MemoryEfficientAttentionFlashAttentionOp
        fw, bw = flash_attention_op
        if fw.supports(xops.fmha.Inputs(query=q, key=k, value=v, attn_bias=attn_bias)):
            return flash_attention_op
    except Exception as e:
        print(e)

    return None


class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 1, 3, 4)
        )
        # rel_pos = RelPosBiasFlash(rel_pos) if rel_pos is not None else None
        # qkv has shape [3, B, num_heads, N, C // num_heads]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        x = xops.memory_efficient_attention(
            q,
            k,
            v,
            p=self.attn_drop,
            attn_bias=rel_pos,
            # op=get_xformers_flash_attention_op(q, k, v, rel_pos),
        )
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with gated adaptive layer norm (adaLN) conditioning.
    """

    def __init__(
            self, hidden_size, num_heads, mlp_ratio=4.0, attention_class=Attention, flash_attention=False,
            **block_kwargs
    ):
        super().__init__()
        self.flash_attention = flash_attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attention_class(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        ) if not flash_attention else FlashMHA(
            embed_dim=hidden_size,  # total channels (= num_heads * head_dim)
            num_heads=num_heads,  # number of heads
            device="cuda"
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, rel_pos=None, alpha=0.5, distance_attention=None):
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp,) = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        attn = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))[0] if self.flash_attention else self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rel_pos, alpha, distance_attention)
        x = x + gate_msa.unsqueeze(1) * attn
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, learn_sigma=False, dont_learn_pos=False):
        super().__init__()
        out_features = 6 if dont_learn_pos else 8
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_features * (2 if learn_sigma else 1), bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ConfidenceHead(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def fourier_transform(strokes, size):
    # strokes: (N, T, 8)
    # size: int
    # returns: (N, T, size)
    N, T, _ = strokes.shape

    # create the fourier basis
    basis = torch.arange(0, size // 2, dtype=torch.float32, device=strokes.device) * 2 * math.pi
    basis = basis.view(1, 1, -1).expand(N, T, -1)
    # basis = basis.unsqueeze(-1).expand(-1, -1, -1, 8)

    # create the strokes
    strokes = strokes.view(N, T, 1, 24).expand(N, T, size // 2, 24)

    # create the fourier representation
    fourier_cos = torch.cat([torch.cos(basis * strokes[..., i]) for i in range(24)], dim=-1)
    fourier_sin = torch.cat([torch.sin(basis * strokes[..., i]) for i in range(24)], dim=-1)
    fourier = torch.cat([fourier_cos, fourier_sin], dim=-1)

    return fourier


def sinusodial_pos_embedding(
        max_len, features, min_scale: float = 1.0, max_scale: float = 20000.0
):
    position = torch.arange(0, max_len)[:, None]
    scale_factor = -np.log(max_scale / min_scale) / (features // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, features // 2) * scale_factor)
    rads = position * div_term

    pe = torch.zeros((max_len, features))
    pe[:, : features // 2] = torch.sin(rads)
    pe[:, features // 2: 2 * (features // 2)] = torch.cos(rads)

    return pe


class XProjecter(nn.Module):
    def __init__(self, hidden_size, use_fourier=False, is_cld=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_fourier = use_fourier
        self.in_features = 24 if is_cld else 16
        # self.linear = nn.Linear(hidden_size if use_fourier else 16, hidden_size, bias=True)
        self.linear = nn.Linear(
            hidden_size if use_fourier else self.in_features, hidden_size, bias=True
        )

    def forward(self, x):
        if self.use_fourier:
            x = fourier_transform(x, self.hidden_size // self.in_features)
        x = self.linear(x)
        return x


def compute_distance_matrix(x, ctx):
    """
    Computes the distance matrix between the spatial positions of the pixels in the image.
    """
    joint_sequence = x + ctx
    # position is defined as the first two elements of last dimension, compute distance matrix
    # between these positions
    position = joint_sequence[..., :2]
    # compute batched pairwise distance matrix
    distance_matrix = torch.cdist(position, position)
    distance_matrix = -torch.cdist(position, position)

    # perform a softmax over the distance matrix, row-wise
    distance_matrix = torch.nn.functional.softmax(distance_matrix, dim=1)

    return distance_matrix.unsqueeze(1)


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            max_seq_len=1000,
            learn_sigma=False,
            use_fourier=False,
            with_skips=False,
            with_confidence=False,
            with_rel_pos=False,
            dont_learn_pos=False,
            normalize_input=False,
            efficient_attention=False,
            is_cld=False,
            use_flash_attention=False,
            with_distance_attention=False,
            fixed_alpha=False,
            triple_cfg=False,
            cfg_scale=1.5,
            cfg_scale2=1.5,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.num_heads = num_heads
        self.with_skips = with_skips
        self.with_confidence = with_confidence
        self.with_rel_pos = with_rel_pos
        self.n_skips = (depth // 2) - 1
        self.num_classes = num_classes
        self.normalize_input = normalize_input
        self.use_flash_attention = use_flash_attention and flash_attn_available
        print(f"Using flash attention: {self.use_flash_attention}")
        self.with_distance_attention = with_distance_attention
        self.fixed_alpha = fixed_alpha
        self.triple_cfg = triple_cfg
        self.cfg_scale = cfg_scale
        self.cfg_scale2 = cfg_scale2
        print(f"Using triple cfg: {self.triple_cfg}")

        self.x_embedder = XProjecter(hidden_size, use_fourier=use_fourier, is_cld=is_cld)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # Will use fixed sin-cos embedding:
        self.pos_embed = sinusodial_pos_embedding(max_seq_len, hidden_size)
        attention_class = EfficientAttention if efficient_attention else Attention

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_class=attention_class,
                    flash_attention=self.use_flash_attention
                )
                for _ in range(depth)
            ]
        )
        if with_skips:
            self.project_skip = nn.ModuleList(
                [nn.Linear(2 * hidden_size, hidden_size) for _ in range(self.n_skips)]
            )
        if with_confidence:
            self.confidence_head = ConfidenceHead(hidden_size)
        if with_rel_pos:
            self.rel_pos = DynamicPositionBias(dim=hidden_size // 4, num_heads=num_heads)
        else:
            self.rel_pos = None

        self.final_layer = FinalLayer(
            hidden_size, learn_sigma=learn_sigma, dont_learn_pos=dont_learn_pos
        )

        self.initialize_weights()

    @property
    def device(self):
        return next(self.parameters()).device

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2])
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.x_embedder.linear.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

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

    def forward(self, x, ctx, t, y, **kwargs):
        y = y.squeeze(-1)
        if t.ndim > 1:
            t = t.squeeze()

        # t is in range [-15, 15], get an alpha in range [0.1, 0.5]
        if self.fixed_alpha:
            alpha = 0.5
        else:
            alpha = (t - (-15)) * (0.5 - 0.1) / (15 - (-15)) + 0.1
        # print("alpha_1", alpha.shape, alpha)
        distance_matrix = compute_distance_matrix(x, ctx) if self.with_distance_attention else None

        x = torch.cat([x, ctx], dim=-1)
        x = x / x.std(dim=(1, 2), keepdim=True) if self.normalize_input else x
        x = self.x_embedder(x) + self.pos_embed.to(x.device)
        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)
        intermediates = []
        for idx, block in enumerate(self.blocks):
            x = block(x, c, self.rel_pos, alpha, distance_matrix)  # (N, T, D)
            if self.with_skips:
                if idx < self.n_skips:
                    intermediates.append(x)
                elif idx > self.n_skips and idx != len(self.blocks) - 1:
                    x = torch.cat([x, intermediates.pop()], dim=-1)
                    x = self.project_skip[idx - self.n_skips - 1](x)

        confidence = self.confidence_head(x, c) if self.with_confidence else None

        x = self.final_layer(x, c)
        return x, confidence

    def forward_with_cfg(self, x, ctx, t, y, **kwargs):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        if not self.triple_cfg:
            combined_x = torch.cat([x, x], dim=0)
            combined_ctx = torch.cat([ctx, ctx], dim=0)
            combined_t = torch.cat([t, t], dim=0)
            combined_y = torch.cat([y, torch.ones_like(y) * self.num_classes], dim=0)

            model_out, confidence = self.forward(combined_x, combined_ctx, combined_t, combined_y)
            cond_eps, uncond_eps = torch.chunk(model_out, 2, dim=0)

            if confidence is not None:
                cond_confidence, uncond_confidence = torch.chunk(confidence, 2, dim=0)
                confidence = uncond_confidence + self.cfg_scale * (cond_confidence - uncond_confidence)

            eps = uncond_eps + self.cfg_scale * (cond_eps - uncond_eps)

            return eps, confidence
        else:
            zero_ctx = torch.zeros_like(ctx)
            zero_class = torch.ones_like(y) * self.num_classes

            combined_x = torch.cat([x, x, x], dim=0)
            combined_ctx = torch.cat([zero_ctx, ctx, ctx], dim=0)
            combined_t = torch.cat([t, t, t], dim=0)
            combined_y = torch.cat([zero_class, zero_class, y], dim=0)

            model_out, confidence = self.forward(combined_x, combined_ctx, combined_t, combined_y)
            uncond_eps, no_class_eps, cond = torch.chunk(model_out, 3, dim=0)

            if confidence is not None:
                cond_confidence, uncond_confidence, uncond_confidence_2 = torch.chunk(confidence, 3, dim=0)
                confidence = uncond_confidence + self.cfg_scale * (cond_confidence - uncond_confidence)

            eps = uncond_eps + self.cfg_scale * (no_class_eps - uncond_eps) + self.cfg_scale2 * (cond - no_class_eps)

            return eps, confidence


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################


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

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)


def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)


def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_S(**kwargs):
    return DiT(depth=12, hidden_size=384, num_heads=6, **kwargs)


if __name__ == "__main__":
    from prettytable import PrettyTable
    from fvcore.nn import FlopCountAnalysis


    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel() / 1e6
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params:.2f}M")
        return total_params


    b, T, e = 2, 180, 8
    x = torch.randn((b, T, e))
    ctx = torch.randn((b, T, e))
    t = torch.randint(0, 10, (b,))
    c = torch.randint(0, 10, (b,))

    # model = DiT_S(num_classes=10, max_seq_len=T, use_fourier=True)
    model = DiT(
        hidden_size=768,
        depth=12,
        num_heads=12,
        num_classes=10,
        max_seq_len=T,
        use_fourier=False,
        with_rel_pos=False,
        with_distance_attention=True,
        is_cld=False,
    )
    count_parameters(model)

    macs_analysis = FlopCountAnalysis(model, (x, ctx, t, c))
    macs = macs_analysis.total() / b
    flops = macs * 2
    print(f"FLOPs: {flops / 1e9:.3f}G")
