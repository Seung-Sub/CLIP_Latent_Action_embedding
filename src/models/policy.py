"""잠재 정책 f: (z_{t−n}, z_t, g(A_past)) 3토큰 → ζ̂ (768) 1토큰.

모듈 3종 (공통 인터페이스: forward(tokens (B,3,768)) -> (B,768)):
  - MLPConcat      : 통짜 결합 MLP
  - CLSTransformer : 학습형 CLS 토큰 + self-attention
  - PMAReadout     : 학습형 query 1개가 3토큰에 cross-attention (Set Transformer PMA)

손실 3항 (plan.md §Phase2):
  L = λ_lat·[MSE+0.5(1−cos)](ζ̂, g(A_fut, z_t))   # 주 잠재 GT (VITA L_FM 자리)
    + λ_act·L1(h(ζ̂, z_t), A_fut)          # action 손실 (FLD 대응)
    + λ_wm ·0.5(1−cos)(ζ̂, z_next − z_t)          # 보조: 미래 시각델타 (FLARE식)
"""
import torch
import torch.nn as nn

LATENT = 768


class MLPConcat(nn.Module):
    def __init__(self, d_model=512, layers=4, heads=None, n_tokens=3):
        super().__init__()
        dims = [n_tokens * LATENT] + [d_model] * (layers - 1)
        net = []
        for i in range(len(dims) - 1):
            net += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
        net.append(nn.Linear(dims[-1], LATENT))
        self.net = nn.Sequential(nn.LayerNorm(n_tokens * LATENT), *net)

    def forward(self, tokens):                    # (B, 3, 768)
        return self.net(tokens.flatten(1))


class CLSTransformer(nn.Module):
    def __init__(self, d_model=512, layers=4, heads=8, n_tokens=3):
        super().__init__()
        self.proj = nn.Linear(LATENT, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=4 * d_model, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, layers)
        self.out = nn.Linear(d_model, LATENT)

    def forward(self, tokens):
        x = self.proj(tokens)                     # (B, 3, d)
        cls = self.cls.expand(len(x), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        return self.out(self.enc(x)[:, 0])        # CLS 출력


class PMAReadout(nn.Module):
    """학습형 query 1개 × key 3개 cross-attention 블록 (layers번 반복 정제)."""

    def __init__(self, d_model=512, layers=2, heads=8, n_tokens=3):
        super().__init__()
        self.proj = nn.Linear(LATENT, d_model)
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "ln_q": nn.LayerNorm(d_model),
                "ln_kv": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, heads, batch_first=True),
                "ln_f": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                     nn.Linear(4 * d_model, d_model)),
            }))
        self.out = nn.Linear(d_model, LATENT)

    def forward(self, tokens):
        kv = self.proj(tokens)                    # (B, 3, d)
        q = self.query.expand(len(kv), -1, -1)    # (B, 1, d)
        for b in self.blocks:
            attn_out, _ = b["attn"](b["ln_q"](q), b["ln_kv"](kv), b["ln_kv"](kv))
            q = q + attn_out
            q = q + b["ffn"](b["ln_f"](q))
        return self.out(q[:, 0])


MODULES = {"mlp": MLPConcat, "cls": CLSTransformer, "pma": PMAReadout}


def build_policy(name, d_model=512, layers=4, heads=8, n_tokens=3):
    return MODULES[name](d_model=d_model, layers=layers, heads=heads,
                         n_tokens=n_tokens)


def policy_losses(zeta, chunk_fut, z_cur, z_next, ae, w):
    """3항 손실. ae = 동결된 DeltaAE (g, h). chunk_fut (B,T,D) 정규화됨."""
    with torch.no_grad():
        lat_target = ae.g(chunk_fut, z_cur)               # 주 GT (동결 g)
    wm_target = z_next - z_cur
    cos = nn.functional.cosine_similarity
    l_lat = (nn.functional.mse_loss(zeta, lat_target)
             + 0.5 * (1 - cos(zeta, lat_target, dim=1)).mean())
    ahat = ae.h(zeta, z_cur)                              # 동결 디코딩 경로
    l_act = nn.functional.l1_loss(ahat, chunk_fut)
    l_wm = 0.5 * (1 - cos(zeta, wm_target, dim=1)).mean()
    total = w["lat"] * l_lat + w["act"] * l_act + w["wm"] * l_wm
    return total, {"lat": l_lat.item(), "act": l_act.item(), "wm": l_wm.item()}
