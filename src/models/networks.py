"""액션청크 <-> Δz AE 네트워크 (VITA Table 9 'AE network' 동형).

  ChunkEncoder g: 1D-CNN(hidden 512, 4층) -> latent 768   (VITA: cnn/512)
  ChunkDecoder h: MLP(hidden 512, 4층)   -> chunk         (VITA: simple/512)
손실 구성은 train_stackcup_ae.py 참조 (align 1.0 / recon 0.5 / cycle 0.5).
"""
import torch
import torch.nn as nn


class ChunkEncoder(nn.Module):
    """(B, n_chunk, action_dim) [+ z_t] -> (B, latent_dim)

    state_cond=True면 현재 이미지 임베딩 z_t를 조건으로 받아
    '같은 액션도 문맥 따라 다른 Δz' 문제를 해소 (상한 상승).
    """

    def __init__(self, action_dim, latent_dim=768, hidden=512, layers=4,
                 dropout=0.0, state_cond=False, cond_dim=None):
        super().__init__()
        self.state_cond = state_cond
        cond_dim = cond_dim or latent_dim
        convs = []
        c_in = action_dim
        for _ in range(layers):
            convs += [nn.Conv1d(c_in, hidden, kernel_size=3, padding=1),
                      nn.GELU()]
            if dropout > 0:
                convs.append(nn.Dropout(dropout))
            c_in = hidden
        self.conv = nn.Sequential(*convs)
        head_in = hidden + (cond_dim if state_cond else 0)
        self.head = nn.Sequential(nn.Linear(head_in, hidden), nn.GELU(),
                                  nn.Linear(hidden, latent_dim))

    def forward(self, chunk, z_t=None):          # (B, T, D), (B, latent)
        x = self.conv(chunk.transpose(1, 2)).mean(dim=2)   # 시간축 평균 풀링
        if self.state_cond:
            x = torch.cat([x, z_t], dim=1)
        return self.head(x)


class ChunkDecoder(nn.Module):
    """(B, delta_dim) [+ z_t] -> (B, n_chunk * action_dim)

    delta_dim: 디코더 입력 표현 차원 (pooled Δz=768, 패치그리드 Δ=16*1024 등)
    """

    def __init__(self, action_dim, n_chunk, latent_dim=768, hidden=512,
                 layers=4, dropout=0.0, state_cond=False, delta_dim=None,
                 cond_dim=None):
        super().__init__()
        self.state_cond = state_cond
        delta_dim = delta_dim or latent_dim
        cond_dim = cond_dim or latent_dim
        in_dim = delta_dim + (cond_dim if state_cond else 0)
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


def info_nce(q, k, temp):
    """배치 대칭 InfoNCE — 짝(q_i, k_i)만 양성, 배치 내 나머지는 음성."""
    qn = nn.functional.normalize(q, dim=1)
    kn = nn.functional.normalize(k, dim=1)
    logits = qn @ kn.t() / temp
    labels = torch.arange(len(q), device=q.device)
    return 0.5 * (nn.functional.cross_entropy(logits, labels)
                  + nn.functional.cross_entropy(logits.t(), labels))


class DeltaAE(nn.Module):
    def __init__(self, action_dim, n_chunk, latent_dim=768, hidden=512,
                 layers=4, dropout=0.0, state_cond=False, delta_dim=None,
                 cond_dim=None, enc_state_cond=None, dec_state_cond=None):
        super().__init__()
        # 인코더/디코더 조건화를 분리 제어 (기본: state_cond를 양쪽에)
        enc_sc = state_cond if enc_state_cond is None else enc_state_cond
        dec_sc = state_cond if dec_state_cond is None else dec_state_cond
        self.state_cond = enc_sc or dec_sc
        self.delta_dim = delta_dim or latent_dim
        self.g = ChunkEncoder(action_dim, latent_dim, hidden, layers, dropout,
                              enc_sc, cond_dim)
        self.h = ChunkDecoder(action_dim, n_chunk, latent_dim, hidden,
                              layers, dropout, dec_sc, delta_dim, cond_dim)
        # 디코더 입력이 pooled Δz(768)와 다른 표현이면 cycle 경로에 사영 필요
        self.g2dec = (nn.Linear(latent_dim, self.delta_dim)
                      if self.delta_dim != latent_dim else nn.Identity())

    def losses(self, chunk, delta_z, w, z_t=None, delta_dec=None):
        """chunk (B,T,D) / delta_z (B,768) 정렬 타깃(pooled Δz) /
        z_t 상태조건(옵션) / delta_dec 디코더 입력 표현(기본 = delta_z).

        VITA 대응: align=FM자리(직접매핑), recon=FLD, cycle=L_AE. 복원은 L1.
        align_type: mse_cos(기본) | infonce | infonce_mse
        """
        if delta_dec is None:
            delta_dec = delta_z
        ghat = self.g(chunk, z_t)                # 액션(+상태) -> 잠재
        ahat = self.h(delta_dec, z_t)            # 실제 Δ표현(+상태) -> 액션 (FLD 대응)
        acyc = self.h(self.g2dec(ghat), z_t)     # 왕복 (L_AE 대응)

        align_type = w.get("align_type", "mse_cos")
        temp = w.get("temp", 0.1)
        if align_type == "infonce":
            l_align = info_nce(ghat, delta_z, temp)
        elif align_type == "infonce_mse":
            l_align = (info_nce(ghat, delta_z, temp)
                       + nn.functional.mse_loss(ghat, delta_z))
        else:  # mse_cos (기본)
            cos = nn.functional.cosine_similarity(ghat, delta_z, dim=1)
            l_align = (nn.functional.mse_loss(ghat, delta_z)
                       + 0.5 * (1 - cos).mean())
        l_recon = nn.functional.l1_loss(ahat, chunk)
        l_cycle = nn.functional.l1_loss(acyc, chunk)
        total = w["align"] * l_align + w["recon"] * l_recon + w["cycle"] * l_cycle
        return total, {"align": l_align.item(), "recon": l_recon.item(),
                       "cycle": l_cycle.item()}
