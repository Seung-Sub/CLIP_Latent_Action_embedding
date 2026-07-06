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
    """(B, T, D) [+ z_t] → (B, latent)"""

    def __init__(self, action_dim, latent_dim=768, hidden=512, layers=4,
                 dropout=0.0, state_cond=True):
        super().__init__()
        self.state_cond = state_cond
        convs, c_in = [], action_dim
        for _ in range(layers):
            convs += [nn.Conv1d(c_in, hidden, kernel_size=3, padding=1),
                      nn.GELU()]
            if dropout > 0:
                convs.append(nn.Dropout(dropout))
            c_in = hidden
        self.conv = nn.Sequential(*convs)
        head_in = hidden + (latent_dim if state_cond else 0)
        self.head = nn.Sequential(nn.Linear(head_in, hidden), nn.GELU(),
                                  nn.Linear(hidden, latent_dim))

    def forward(self, chunk, z_t=None):
        x = self.conv(chunk.transpose(1, 2)).mean(dim=2)   # 시간축 평균 풀링
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
                 align_mode="dz", contrast_w=0.0, contrast_head=False):
        super().__init__()
        self.g = ChunkEncoder(action_dim, latent_dim, hidden, layers,
                              dropout, state_cond)
        self.h = ChunkDecoder(action_dim, n_chunk, latent_dim, hidden,
                              layers, dropout, state_cond)
        assert align_mode in ("dz", "direct", "hybrid"), align_mode
        self.align_mode = align_mode
        self.contrast_w = contrast_w
        if align_mode != "dz":
            # 학습형 온도 (CLIP 관례: logit_scale = log(1/T), T init 0.07)
            import numpy as np
            self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))
            # 노름 분리 (레시피 변형): contrastive를 전용 투영 위에서 계산해
            # 회귀(비정규화 Δz)·디코드 기하와 분리 (SimCLR 투영헤드 원리)
            if contrast_head:
                self.contrast_proj = nn.Linear(latent_dim, latent_dim)

    def info_nce(self, ghat, text_emb, sent_ids):
        """SupCon식 다중 양성 InfoNCE. text_emb (B, d), sent_ids (B,).
        contrast_proj 존재 시 g를 투영 후 정규화 (노름 분리 레시피)."""
        if hasattr(self, "contrast_proj"):
            ghat = self.contrast_proj(ghat)
        gn = nn.functional.normalize(ghat, dim=1)
        tn = nn.functional.normalize(text_emb, dim=1)
        logits = gn @ tn.T * self.logit_scale.exp().clamp(max=100.0)
        pos = (sent_ids[:, None] == sent_ids[None, :])       # (B, B) 동일 문장 = 양성
        # loss_i = −log( Σ_pos exp / Σ_all exp )
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
        parts.update({"recon": l_recon.item(), "cycle": l_cycle.item()})
        return total, parts
