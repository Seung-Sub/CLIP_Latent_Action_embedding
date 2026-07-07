"""GT 에피소드 전체 시계열 추론 평가 — 예측 궤적이 GT 그래프를 따라가는지.

에피소드 하나를 처음부터 끝까지 receding-horizon 방식으로 추론:
  t = 16, 24, 32, ... 마다 (z_{t−16}, z_t, g(A_{t−16:t})) → f → h → Â_{t:t+16}
  각 예측의 앞 8스텝을 이어붙여 전체 예측 궤적을 구성 (GT 이미지 사용 = 개루프 액션)
14차원(양팔 관절 12 + 그리퍼 2) 전부를 GT와 겹쳐 그린다.

사용 (clip env):
  python src/eval_aloha/rollout_dataset.py                       # val 첫 에피소드
  python src/eval_aloha/rollout_dataset.py --episode 3 --task sim_insertion
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os

import matplotlib
import numpy as np
import torch
import yaml
from matplotlib import font_manager

for _f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    try:
        font_manager.fontManager.addfont(_f)
    except FileNotFoundError:
        pass
matplotlib.rcParams.update({"font.family": ["Noto Sans CJK KR", "sans-serif"],
                            "axes.unicode_minus": False})
import matplotlib.pyplot as plt

from core import chunkrep
from core.anchor import get_anchor
from data.act_sim import ActSimDataset
from models.networks import DeltaAE
from models.policy import build_policy_from_cfg

WS = Path(__file__).resolve().parents[2]
JOINT_NAMES = ([f"L{i}" for i in range(1, 7)] + ["L_grip"]
               + [f"R{i}" for i in range(1, 7)] + ["R_grip"])


def load_models(cfg, device):
    ck1 = torch.load(os.path.expanduser(cfg["phase1_ckpt"]),
                     map_location="cpu", weights_only=False)
    p1 = ck1["config"]
    latent = ck1.get("latent_dim", p1["model"]["latent_dim"])
    ae = DeltaAE(ck1["action_dim"], ck1["n_chunk"], latent,
                 p1["model"]["hidden"], p1["model"]["layers"],
                 p1["model"]["dropout"],
                 p1["model"].get("state_cond", True),
                 align_mode=p1["model"].get("align_mode", "dz"),
                 g_state_cond=p1["model"].get("g_state_cond"),
                 h_state_cond=p1["model"].get("h_state_cond"),
                 encoder_kind=p1["model"].get("encoder_kind", "cnn")).to(device).eval()
    ae.load_state_dict(ck1["state_dict"])
    ck2 = torch.load(os.path.expanduser(cfg["train"]["checkpoint"]),
                     map_location="cpu", weights_only=False)
    m = ck2["config"]["module"]
    policy = build_policy_from_cfg(m, latent=latent).to(device).eval()
    policy.load_state_dict(ck2["state_dict"])
    return (ae, policy, ck1["a_mean"], ck1["a_std"], ck1["n_chunk"],
            ck1["action_dim"], ck1.get("chunk_repr", "time"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(WS / "configs" / "phase2.yaml"))
    ap.add_argument("--task", default="sim_transfer_cube",
                    choices=["sim_transfer_cube", "sim_insertion"])
    ap.add_argument("--episode", type=int, default=None,
                    help="에피소드 번호 (기본: val 분할의 첫 에피소드)")
    ap.add_argument("--exec-horizon", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae, policy, a_mean, a_std, n_chunk, act_dim, repr_kind = load_models(cfg, device)
    ds = ActSimDataset(cfg)
    clip = get_anchor(cfg)

    # 에피소드 선택: 지정 없으면 학습과 동일한 분할 재현 후 해당 task의 val 첫 번째
    files = ds.episode_files()
    if args.episode is not None:
        path = Path(os.path.expanduser(
            f"~/clip_ws/data/act_sim/{args.task}/episode_{args.episode}.hdf5"))
    else:
        rng = np.random.RandomState(cfg["train"]["seed"])
        perm = rng.permutation(len(files))
        v = cfg["data"]["val_episodes"]
        n_val = max(1, round(len(files) * v)) if v < 1 else int(v)
        val_files = [files[i] for i in perm[:n_val]]
        path = next(p for p in val_files if args.task in str(p))
    print(f"에피소드: {path}")

    acts = ds.load_actions(path)                       # (T, 14) GT
    Z = ds.embeddings(clip, path)                      # (T, 768) 캐시
    T = len(acts)
    span, H = ds.span, args.exec_horizon

    def norm(a):
        return ((a - a_mean) / a_std).astype(np.float32)

    pred = np.full_like(acts, np.nan)
    t = span
    with torch.no_grad():
        while t + span <= T:
            z_prev = torch.tensor(Z[t - span][None], device=device)
            z_cur = torch.tensor(Z[t][None], device=device)
            past = chunkrep.to_repr(
                norm(ds.resample_chunk(acts[t - span:t])), repr_kind)[None]
            a_emb = ae.g(torch.tensor(past, device=device), z_prev)
            tokens = torch.stack([z_prev, z_cur, a_emb], dim=1)
            zeta = policy(tokens)
            ahat = ae.h(zeta, z_cur).cpu().numpy()[0]                   # (16, 14) repr
            ahat = chunkrep.from_repr(ahat, repr_kind) * a_std + a_mean
            n_exec = min(H, T - t)
            # n_chunk(16) 예측을 실제 span 스텝에 대응 (여기선 1:1)
            pred[t:t + n_exec] = ahat[:n_exec]
            t += H

    valid = ~np.isnan(pred[:, 0])
    arm = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    mae_deg = np.abs(pred[valid][:, arm] - acts[valid][:, arm]).mean() * 180 / np.pi
    grip_acc = ((pred[valid][:, [6, 13]] > 0.5)
                == (acts[valid][:, [6, 13]] > 0.5)).mean() * 100
    print(f"전체 시계열 추론: 관절 MAE {mae_deg:.2f}° | 그리퍼 {grip_acc:.1f}% "
          f"| 추론 구간 {valid.sum()}/{T} steps")

    # ---- 14차원 전부 plot ----
    fig, axes = plt.subplots(7, 2, figsize=(14, 16), dpi=110, sharex=True)
    tt = np.arange(T) / 50.0
    for d in range(act_dim):
        ax = axes[d % 7, d // 7]
        ax.plot(tt, acts[:, d], color="#4477AA", lw=1.6, label="GT")
        ax.plot(tt, pred[:, d], color="#EE6677", lw=1.2, ls="--", label="예측")
        ax.set_title(JOINT_NAMES[d], fontsize=9)
        ax.grid(color="#E5E7EB", lw=0.5)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"{path.parent.name}/{path.stem} — 전체 시계열 추론 (GT 이미지, "
                 f"{H}스텝 receding)\n관절 MAE {mae_deg:.2f}° | 그리퍼 {grip_acc:.1f}%",
                 fontsize=12)
    fig.tight_layout()
    out = WS / "outputs" / "eval" / f"rollout_dataset_{path.parent.name}_{path.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
