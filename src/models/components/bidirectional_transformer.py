import math
import torch
import torch.nn as nn
from einops import rearrange

from timm.models.vision_transformer import Mlp


def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        # print(f"Initializing Module {classname}.")
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
    # elif "Parameter" in classname:
    #     return nn.init.trunc_normal_(m, 0.0, 0.02)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim should be divisible by num_heads, got {dim} and {num_heads}"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos=None):
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
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class DiTBlock(nn.Module):
    """
    A DiT block with gated adaptive layer norm (adaLN) conditioning.
    """

    def __init__(
        self, hidden_size, num_heads, mlp_ratio=4.0, attention_class=Attention, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attention_class(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
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

    def forward(self, x, c, rel_pos=None):
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp,) = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rel_pos
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class BidirectionalTransformer(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        n_layers=12,
        num_heads=8,
        max_seq_len=1024,
        max_tokens=100,
        num_classes=10,
        class_dropout_prob=0.1,
    ):
        super(BidirectionalTransformer, self).__init__()
        # hidden_dim = 4 * dim
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(max_tokens + 1, hidden_size)  # // 8)
        self.pos_emb = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(self.max_seq_len, hidden_size * 8)), 0.0, 0.02
        )
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # self.register_buffer("pos_emb", nn.init.trunc_normal_(nn.Parameter(torch.zeros(1024, args.dim)), 0., 0.02))
        self.x_projecter = nn.Linear(8 * hidden_size, hidden_size)
        self.blocks = nn.Sequential(*[DiTBlock(hidden_size, num_heads) for _ in range(n_layers)])
        # self.proj = nn.Linear(hidden_size, hidden_size * 8)
        self.proj = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size),
            ]
            * 8
        )
        self.Token_Prediction = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Linear(in_features=hidden_size, out_features=hidden_size),
                        nn.GELU(),
                        nn.LayerNorm(hidden_size, eps=1e-12),
                    ]
                )
            ]
            * 8
        )
        self.bias = nn.Parameter(torch.zeros(self.max_seq_len, max_tokens + 1))
        self.ln = nn.LayerNorm(hidden_size * 8, eps=1e-12)
        self.drop = nn.Dropout(p=0.1)
        self.apply(weights_init)

    def forward(self, x, y):
        # print(f"max tokens: {x.max()}")
        token_embeddings = self.tok_emb(x)
        token_embeddings = rearrange(token_embeddings, "b (l f) d -> b l (f d)", f=8)
        # print(token_embeddings.shape)
        # exit(1)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:t, :]
        y = y.squeeze(-1)
        y = self.y_embedder(y, self.training)  # (N, D)
        # position_embeddings = self.pos_emb(x)
        embed = self.drop(self.ln(token_embeddings + position_embeddings))
        embed = self.x_projecter(embed)
        for block in self.blocks:
            embed = block(embed, y)
        embeds = []
        for proj in self.proj:
            embeds.append(proj(embed))
        for i in range(len(embeds)):
            embeds[i] = self.Token_Prediction[i](embeds[i])
        embed = torch.cat(embeds, dim=1)
        # embed = torch.cat(embeds, dim=-1)
        # embed = self.Token_Prediction(embed)
        # embed = rearrange(embed, "b l (h p) -> b (l p) h", p=8)
        logits = torch.matmul(embed, self.tok_emb.weight.T) + self.bias
        # print(logits.shape)
        return logits


if __name__ == "__main__":
    model = BidirectionalTransformer(
        hidden_size=(768 // 2), n_layers=6, num_heads=8, max_seq_len=1440, max_tokens=256
    )
    # print(
    #     f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M"
    # )
    from prettytable import PrettyTable
    from fvcore.nn import FlopCountAnalysis

    b = 2

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

    count_parameters(model)
    x = torch.randint(0, 257, (b, 1440))
    y = torch.randint(0, 10, (b, 1))
    # ctx = torch.randn(1, 512, 8)

    macs_analysis = FlopCountAnalysis(model, (x, y))
    macs = macs_analysis.total() / b
    flops = macs * 2
    print(f"FLOPs: {flops / 1e9:.3f}G")

    # logits = model(x, y)
    # print(logits.shape)
