"""정렬 리포트 (시각화 표준 §7) — phase1 런의 정렬 상태 공식 뷰.

구성: ① per-sample cos(g, Δz) 히스토그램 ② 노름비 ‖g‖/‖Δz‖ 산점(vs ‖Δz‖)
③ 태스크별 요약표 ④ retrieval 요약. PCA 산점 인상 대신 이 그림이 공식 뷰
(근거: docs/upgrade_ledger.md '정렬의 의미론' — align 항은 접지 정규화).

사용: python src/diagnosis/alignment_report.py --ckpt checkpoints/grid/<tag>.pt
      [--unnorm]  # 앵커 비정규화 레시피 런이면 지정 (캐시 키 일치용)
산출: outputs/report/alignment_<tag>.png + .json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from core.anchor import get_anchor
from data.libero import LiberoDataset
from models.networks import DeltaAE

WS = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=str(WS / "configs" / "phase1_libero.yaml"))
    ap.add_argument("--unnorm", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(WS / args.ckpt if not Path(args.ckpt).is_absolute()
                    else args.ckpt, map_location="cpu", weights_only=False)
    p1 = ck["config"]["model"]
    cfg = yaml.safe_load(open(args.config))
    # 앵커 설정은 체크포인트가 진실 (없으면 CLI 플래그)
    ck_anchor = ck["config"].get("anchor") or {}
    if args.unnorm:
        ck_anchor = {**ck_anchor, "normalize": False}
    cfg["anchor"] = ck_anchor
    cfg["data"]["chunk_sec"] = ck["config"]["data"].get("chunk_sec", 0.8)
    ae = DeltaAE(ck["action_dim"], ck["n_chunk"],
                 ck.get("latent_dim", p1["latent_dim"]), p1["hidden"],
                 p1["layers"], p1["dropout"], p1.get("state_cond", True),
                 align_mode=p1.get("align_mode", "dz"),
                 contrast_head=p1.get("contrast_head", False)).to(device).eval()
    ae.load_state_dict(ck["state_dict"])
    ds = LiberoDataset(cfg)
    clip = get_anchor(cfg)

    rng = np.random.RandomState(ck["config"]["train"]["seed"])
    files = ds.episode_files()
    perm = rng.permutation(len(files))
    v = cfg["data"]["val_episodes"]
    n_val = max(1, round(len(files) * v)) if v < 1 else int(v)
    val_files = [files[i] for i in perm[:n_val]]
    eps = ds.build(clip, val_files, verbose=False)

    coss, ratios, dzn, tasks = [], [], [], []
    for ep, (Zt, Ztn, A) in zip(val_files, eps):
        C = ((A.reshape(len(A), ck["n_chunk"], ck["action_dim"]) - ck["a_mean"])
             / ck["a_std"]).astype(np.float32)
        D = Ztn - Zt
        with torch.no_grad():
            G = ae.g(torch.tensor(C, device=device),
                     torch.tensor(Zt, device=device)).cpu().numpy()
        cos = (G * D).sum(1) / (np.linalg.norm(G, axis=1)
                                * np.linalg.norm(D, axis=1) + 1e-9)
        coss.append(cos)
        ratios.append(np.linalg.norm(G, axis=1)
                      / (np.linalg.norm(D, axis=1) + 1e-9))
        dzn.append(np.linalg.norm(D, axis=1))
        tasks += [str(ep[0].stem)[:40]] * len(cos)
    cos = np.concatenate(coss)
    ratio = np.concatenate(ratios)
    dzn = np.concatenate(dzn)
    tasks = np.array(tasks)

    # retrieval (train_phase1과 동일 규격, val 전체)
    Zt_all = np.concatenate([e[0] for e in eps])
    D_all = np.concatenate([e[1] - e[0] for e in eps])
    A_all = np.concatenate([e[2] for e in eps])
    C_all = ((A_all.reshape(len(A_all), ck["n_chunk"], -1) - ck["a_mean"])
             / ck["a_std"]).astype(np.float32)
    with torch.no_grad():
        G_all = ae.g(torch.tensor(C_all, device=device),
                     torch.tensor(Zt_all, device=device)).cpu().numpy()
    def retr(Q, K):
        Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        Kn = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-8)
        rank = (-(Qn @ Kn.T)).argsort(1)
        m = np.arange(len(Q))
        return (float((rank[:, 0] == m).mean() * 100),
                float((rank[:, :5] == m[:, None]).any(1).mean() * 100))
    t1a, t5a = retr(G_all, D_all)

    tag = Path(args.ckpt).stem
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].hist(cos, bins=80, color="#4477aa")
    axes[0].axvline(np.median(cos), color="r", ls="--",
                    label=f"median {np.median(cos):.3f}")
    axes[0].set(xlabel="per-sample cos(g, Δz)", ylabel="samples",
                title=f"alignment dist | a→Δz retr top-1 {t1a:.1f}% top-5 {t5a:.1f}%")
    axes[0].legend()
    sc = axes[1].scatter(dzn, ratio, s=3, alpha=0.3, c=cos, cmap="coolwarm",
                         vmin=-1, vmax=1)
    axes[1].axhline(1.0, color="k", lw=0.5)
    axes[1].set(xlabel="‖Δz‖", ylabel="‖g‖/‖Δz‖", yscale="log",
                title="norm ratio vs Δz magnitude (색=cos)")
    plt.colorbar(sc, ax=axes[1])
    uniq = sorted(set(tasks))
    med = [float(np.median(cos[tasks == t])) for t in uniq]
    axes[2].barh(range(len(uniq)), med, color="#4477aa")
    axes[2].set_yticks(range(len(uniq)))
    axes[2].set_yticklabels([t[:28] for t in uniq], fontsize=7)
    axes[2].set(xlabel="median cos(g, Δz)", title="per-task alignment")
    fig.suptitle(f"정렬 리포트: {tag} (align은 접지 정규화 — ledger '정렬의 의미론' 참조)",
                 fontsize=10)
    fig.tight_layout()
    png = WS / "outputs" / "report" / f"alignment_{tag}.png"
    fig.savefig(png, dpi=120)
    out = {"tag": tag, "cos_median": float(np.median(cos)),
           "cos_p10_p90": [float(np.percentile(cos, 10)),
                           float(np.percentile(cos, 90))],
           "norm_ratio_median": float(np.median(ratio)),
           "retrieval_a2z": [t1a, t5a],
           "per_task_cos_median": dict(zip(uniq, med)), "n_val": int(len(cos))}
    (WS / "outputs" / "report" / f"alignment_{tag}.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print(f"cos 중앙값 {np.median(cos):.3f} | 노름비 중앙값 {np.median(ratio):.2f} "
          f"| a→Δz top-1 {t1a:.1f}%")
    print(f"저장: {png}")


if __name__ == "__main__":
    main()
