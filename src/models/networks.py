"""Phase 1 — 액션청크 <-> Δz(pooled 768) 결합 AE.

  인코더 g: 액션청크(T×D) + z_t → ζ (768)     [1D-CNN, 상태조건]
  디코더 h: Δz(768) + z_t → 액션청크           [MLP, 상태조건]

손실 (VITA 동형):
  align: g(a, z_t) ≈ Δz  (MSE + 0.5·cos)      — FM 자리
  recon: h(Δz, z_t) ≈ a  (L1)                  — FLD 대응
  cycle: h(g(a,z_t), z_t) ≈ a  (L1)            — L_AE 대응, phase2 디코딩 경로
"""
import torch
import torch.nn as nn


class ChunkEncoder(nn.Module):
    """(B, T, D) [+ z_t] → (B, latent). encoder_kind (절제 #6):
    cnn(기본 1D-CNN+mean) / strided(QueST식 causal strided conv) /
    transformer(3층 d256 [CLS]) / mlp(flatten)."""

    def __init__(self, action_dim, latent_dim=768, hidden=512, layers=4,
                 dropout=0.0, state_cond=True, encoder_kind="cnn", n_chunk=16):
        super().__init__()
        self.state_cond = state_cond
        self.kind = encoder_kind
        if encoder_kind == "cnn":
            convs, c_in = [], action_dim
            for _ in range(layers):
                convs += [nn.Conv1d(c_in, hidden, kernel_size=3, padding=1),
                          nn.GELU()]
                if dropout > 0:
                    convs.append(nn.Dropout(dropout))
                c_in = hidden
            self.conv = nn.Sequential(*convs)   # 이름 유지 = 구 ckpt state_dict 호환
            feat = hidden
        elif encoder_kind == "strided":   # QueST 2407.15840식: causal, stride 2 다운샘플
            # causal pad = k - s (검증 리뷰: (2,0)+stride2는 마지막 3스텝 그래디언트 0)
            convs, c_in = [], action_dim
            for i in range(layers):
                stride = 2 if i < 2 else 1
                convs += [nn.ConstantPad1d((3 - stride, 0), 0.0),
                          nn.Conv1d(c_in, hidden, kernel_size=3, stride=stride),
                          nn.GELU()]
                c_in = hidden
            self.body = nn.Sequential(*convs)
            feat = hidden
        elif encoder_kind == "transformer":
            d = 256
            self.inp = nn.Linear(action_dim, d)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.pos = nn.Parameter(torch.zeros(1, n_chunk + 1, d))
            enc = nn.TransformerEncoderLayer(d, 8, 4 * d, activation="gelu",
                                             batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(enc, 3)
            feat = d
        elif encoder_kind == "mlp":
            self.body = nn.Sequential(
                nn.LayerNorm(n_chunk * action_dim),
                nn.Linear(n_chunk * action_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU())
            feat = hidden
        else:
            raise ValueError(encoder_kind)
        head_in = feat + (latent_dim if state_cond else 0)
        self.head = nn.Sequential(nn.Linear(head_in, hidden), nn.GELU(),
                                  nn.Linear(hidden, latent_dim))

    def forward(self, chunk, z_t=None):
        if self.kind == "cnn":
            x = self.conv(chunk.transpose(1, 2)).mean(dim=2)
        elif self.kind == "strided":
            x = self.body(chunk.transpose(1, 2)).mean(dim=2)
        elif self.kind == "transformer":
            h = self.inp(chunk)
            h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1) + self.pos
            x = self.body(h)[:, 0]
        else:
            x = self.body(chunk.flatten(1))
        if self.state_cond:
            x = torch.cat([x, z_t], dim=1)
        return self.head(x)


class ChunkDecoder(nn.Module):
    """(B, latent) [+ z_t] → (B, T, D)"""

    def __init__(self, action_dim, n_chunk, latent_dim=768, hidden=512,
                 layers=4, dropout=0.0, state_cond=True):
        super().__init__()
        self.state_cond = state_cond
        in_dim = latent_dim * (2 if state_cond else 1)
        dims = [in_dim] + [hidden] * (layers - 1)
        mlp = [nn.LayerNorm(in_dim)]
        for i in range(len(dims) - 1):
            mlp += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
            if dropout > 0:
                mlp.append(nn.Dropout(dropout))
        mlp.append(nn.Linear(dims[-1], n_chunk * action_dim))
        self.mlp = nn.Sequential(*mlp)
        self.n_chunk, self.action_dim = n_chunk, action_dim

    def forward(self, z, z_t=None):
        if self.state_cond:
            z = torch.cat([z, z_t], dim=1)
        return self.mlp(z).view(-1, self.n_chunk, self.action_dim)


class DeltaAE(nn.Module):
    """align_mode (C8 절제): dz(기준) / direct(InfoNCE→모션문장) / hybrid(dz+λc·InfoNCE).

    direct/hybrid에서도 recon·cycle 손실은 유지 (contrastive 단독 붕괴 방지 — C8 규약).
    InfoNCE: in-batch negatives, 동일 문장 샘플은 다중 양성으로 마스킹(SupCon식 —
    고유 문장 ~330종이라 배치 내 중복이 흔해 false negative 보정 필수), 온도 학습형.
    """

    def __init__(self, action_dim, n_chunk, latent_dim=768, hidden=512,
                 layers=4, dropout=0.0, state_cond=True,
                 align_mode="dz", contrast_w=0.0, contrast_head=False,
                 g_state_cond=None, h_state_cond=None, encoder_kind="cnn",
                 contrast_loss="infonce"):
        super().__init__()
        # QueST 절제 #4: g/h 상태조건 개별 제어 (기본 = 기존 state_cond 동일)
        g_sc = state_cond if g_state_cond is None else g_state_cond
        h_sc = state_cond if h_state_cond is None else h_state_cond
        self.g = ChunkEncoder(action_dim, latent_dim, hidden, layers,
                              dropout, g_sc, encoder_kind, n_chunk)
        self.h = ChunkDecoder(action_dim, n_chunk, latent_dim, hidden,
                              layers, dropout, h_sc)
        assert align_mode in ("dz", "direct", "hybrid"), align_mode
        self.align_mode = align_mode
        self.contrast_w = contrast_w
        self.contrast_loss = contrast_loss   # "infonce"(기본) | "sigmoid"(SigLIP식)
        if align_mode != "dz":
            import numpy as np
            if contrast_loss == "sigmoid":
                # SigLIP 관례 초기화 (2303.15343): t'=log10, b=-10 (전역 1쌍)
                self.logit_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))
                self.logit_bias = nn.Parameter(torch.tensor(-10.0))
            else:
                # InfoNCE 학습형 온도 (CLIP 관례: log(1/0.07))
                self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))
            # 노름 분리 (레시피 변형): contrastive 전용 투영 (SimCLR 투영헤드 원리)
            if contrast_head:
                self.contrast_proj = nn.Linear(latent_dim, latent_dim)

    def info_nce(self, ghat, text_emb, sent_ids):
        """대조 정렬 손실. contrast_loss="infonce"(SupCon 다중양성) | "sigmoid"(SigLIP식
        쌍별 이진, 전역 t·b). contrast_proj 존재 시 g를 투영 후 정규화 (노름 분리)."""
        if hasattr(self, "contrast_proj"):
            ghat = self.contrast_proj(ghat)
        gn = nn.functional.normalize(ghat, dim=1)
        tn = nn.functional.normalize(text_emb, dim=1)
        pos = (sent_ids[:, None] == sent_ids[None, :])       # (B, B) 동일 문장 = 양성
        if self.contrast_loss == "sigmoid":
            # SigLIP 쌍별: label ±1, loss = −Σ log σ(label·(t·sim + b)) / B
            logits = (gn @ tn.T) * self.logit_scale.exp().clamp(max=200.0) \
                + self.logit_bias
            labels = torch.where(pos, 1.0, -1.0)
            # 전 쌍 평균 (버그 수정: .sum(1)은 ~B배 커져 hybrid에서 align 지배 →
            # align/recon과 스케일 정합 위해 mean over all pairs)
            return -nn.functional.logsigmoid(labels * logits).mean()
        logits = gn @ tn.T * self.logit_scale.exp().clamp(max=100.0)
        all_lse = torch.logsumexp(logits, dim=1)
        pos_lse = torch.logsumexp(
            logits.masked_fill(~pos, float("-inf")), dim=1)
        return (all_lse - pos_lse).mean()

    def losses(self, chunk, delta_z, w, z_t=None, text_emb=None, sent_ids=None):
        ghat = self.g(chunk, z_t)                # 액션(+상태) → 잠재
        ahat = self.h(delta_z, z_t)              # 실제 Δz(+상태) → 액션 (FLD 대응)
        acyc = self.h(ghat, z_t)                 # 왕복 (phase2 디코딩 경로)
        l_recon = nn.functional.l1_loss(ahat, chunk)
        l_cycle = nn.functional.l1_loss(acyc, chunk)
        parts = {}
        total = w["recon"] * l_recon + w["cycle"] * l_cycle
        if self.align_mode in ("dz", "hybrid"):
            cos = nn.functional.cosine_similarity(ghat, delta_z, dim=1)
            l_align = (nn.functional.mse_loss(ghat, delta_z)
                       + 0.5 * (1 - cos).mean())
            total = total + w["align"] * l_align
            parts["align"] = l_align.item()
        if self.align_mode in ("direct", "hybrid"):
            assert text_emb is not None and sent_ids is not None, \
                "direct/hybrid 정렬엔 모션 문장 임베딩 필요"
            l_con = self.info_nce(ghat, text_emb, sent_ids)
            cw = self.contrast_w if self.align_mode == "hybrid" else w["align"]
            total = total + cw * l_con
            parts["contrast"] = l_con.item()
        w_comp = float(w.get("comp", 0.0))
        if w_comp > 0 and self.g.kind not in ("cnn", "strided"):
            raise ValueError(f"loss.comp는 가변길이 인코더 전용 (현재 {self.g.kind} — "
                             "transformer/mlp는 고정 길이라 half-chunk 불가)")
        if w_comp > 0:                       # 절제 #2 (CLASP 1806.09655 조합성)
            T = chunk.shape[1]
            half = T // 2
            # z_mid ≈ z_t + g(전반) 로 근사한 상태에서 후반 인코딩 (텔레스코핑 정합)
            g_a = self.g(chunk[:, :half], z_t)
            z_mid = (z_t + g_a) if z_t is not None else None
            g_b = self.g(chunk[:, half:], z_mid)
            g_full = self.g(chunk, z_t)
            l_comp = nn.functional.mse_loss(g_a + g_b, g_full)
            total = total + w_comp * l_comp
            parts["comp"] = l_comp.item()
        w_vel = float(w.get("vel", 0.0))
        if w_vel > 0:                        # 절제 #5: 속도(1차 차분) L2
            l_vel = nn.functional.mse_loss(
                ahat[:, 1:] - ahat[:, :-1], chunk[:, 1:] - chunk[:, :-1])
            total = total + w_vel * l_vel
            parts["vel"] = l_vel.item()
        parts.update({"recon": l_recon.item(), "cycle": l_cycle.item()})
        return total, parts
