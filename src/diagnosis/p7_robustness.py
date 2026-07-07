"""P7 관측 강건성 프로브 — 앵커별 뉘앙스 SNR + 방문-z 이탈도 (분석자 신설).

(a) 뉘앙스 SNR: 데모의 연속 프레임 1-스텝 ‖Δz‖를 액션 norm 하위 5%(정지 근사) vs
    상위 50%(액션 구간)로 분리 — 앵커별 SNR = median(action)/median(nuisance).
    캐시 임베딩만 사용 (unnorm 캐시 = 확정 레시피 공간).
(b) 방문-z 이탈도: 롤아웃 영상 프레임을 해당 앵커로 인코딩 → 데모 캐시 kNN(k=10)
    cosine 거리 분포. 성공/실패 에피소드 분리 (영상 파일명의 ok/fail).
    한계: z 트레이스 미저장 → 저장된 영상(성공 일부+실패 2/task)만 표본. CLIP-DZ는
    성공·실패 양쪽, DINOv2-DZ는 실패 위주(당시 --save-video 0) — 표본 구성 명기.
(c) 이탈도 시계열: 실패 에피소드에서 스텝별 이탈도 궤적 (급등 여부).

사전 등록 예측(분석자): DINOv2 이탈 분포 우측 편이 + 실패 에피소드에서 증폭.
산출: outputs/report/p7_robustness.json + p7_fig_dist.png + p7_fig_traj.png
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from core.anchor import ClipAnchor, Dinov2Anchor
from data.libero import LiberoDataset

WS = Path(__file__).resolve().parents[2]
VID = WS / "outputs" / "eval" / "videos"


def nuisance_snr(cache_key, ds, files):
    """(a) 캐시 임베딩 1-스텝 Δz: 저액션(하위5%) vs 액션(상위50%) 노름."""
    nuis, act = [], []
    for ep in files:
        cache = (Path(ds.cache_dir) / cache_key /
                 (ds._key(ep) + "_agentview_rgb.npz"))
        if not cache.exists():
            continue
        Z = np.load(cache)["Z"]
        a = ds.load_actions(ep)
        n = min(len(Z) - 1, len(a) - 1)
        dz = np.linalg.norm(Z[1:n+1] - Z[:n], axis=1)
        an = np.linalg.norm(a[:n, :-1], axis=1)
        lo, hi = np.percentile(an, 5), np.percentile(an, 50)
        nuis.append(dz[an <= lo])
        act.append(dz[an >= hi])
    nuis, act = np.concatenate(nuis), np.concatenate(act)
    return {"nuisance_dz_median": float(np.median(nuis)),
            "action_dz_median": float(np.median(act)),
            "snr": float(np.median(act) / (np.median(nuis) + 1e-12)),
            "n_nuis": int(len(nuis)), "n_act": int(len(act))}, nuis, act


def demo_bank(cache_key, ds, files):
    Zs = []
    for ep in files:
        cache = (Path(ds.cache_dir) / cache_key /
                 (ds._key(ep) + "_agentview_rgb.npz"))
        if cache.exists():
            Zs.append(np.load(cache)["Z"])
    Z = np.concatenate(Zs)
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def video_deviation(anchor, bank, pattern):
    """(b)(c) 영상 프레임 → 앵커 인코딩 → 데모 kNN(k=10) cos 거리."""
    per_ep = []
    for vp in sorted(VID.glob(pattern)):
        m = re.match(r".*_t(\d+)_ep(\d+)_(ok|fail)\.mp4", vp.name)
        if not m:
            continue
        frames = imageio.mimread(vp, memtest=False)[::4]     # 4스텝 서브샘플
        Z = []
        for i in range(0, len(frames), 64):
            Z.append(anchor.encode_images(
                [Image.fromarray(f) for f in frames[i:i+64]])["embeds"])
        Z = np.concatenate(Z)
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        sim = Zn @ bank.T
        top = np.sort(sim, axis=1)[:, -10:]
        dist = 1 - top.mean(axis=1)                          # kNN(10) 평균 cos 거리
        per_ep.append({"task": int(m[1]), "ep": int(m[2]), "ok": m[3] == "ok",
                       "dist": dist})
    return per_ep


def main():
    cfg = yaml.safe_load(open(WS / "configs" / "phase1_libero.yaml"))
    ds = LiberoDataset(cfg)
    files = ds.episode_files()

    report = {"a_nuisance_snr": {}, "b_deviation": {}, "protocol": {
        "snr": "1-step ‖Δz‖, 저액션=action norm 하위5% vs 액션=상위50%, unnorm 캐시",
        "deviation": "롤아웃 영상 4스텝 서브샘플 → kNN(k=10) 평균 cos 거리 vs 데모 뱅크",
        "sample_caveat": "z 트레이스 미저장 → 저장 영상만: CLIP-DZ 성공+실패 / "
                         "DINOv2-DZ 실패 위주 (--save-video 0이었음)"}}

    # (a) 앵커 3종 SNR
    dists_a = {}
    for name, key in (("clip", "clip-vit-l-14/joint/raw"),
                      ("siglip2", "siglip2-so400m/joint/raw"),
                      ("dinov2", "dinov2-large/pre/raw")):
        stats, nuis, act = nuisance_snr(key, ds, files)
        report["a_nuisance_snr"][name] = stats
        dists_a[name] = (nuis, act)
        print(f"[SNR] {name}: nuisance {stats['nuisance_dz_median']:.4f} vs "
              f"action {stats['action_dz_median']:.4f} → SNR {stats['snr']:.2f}")

    # (b)(c) 이탈도: CLIP-DZ vs DINOv2-DZ — mtime 귀속 아카이브 사용
    # (파일명 run_id 부재 → 덮어쓰기 충돌 발견, presentation/videos_p7/에 귀속 스냅샷.
    #  향후 런은 run_id 접두사로 수정 완료)
    global VID
    clip_anchor = ClipAnchor(normalize=False)
    bank_c = demo_bank("clip-vit-l-14/joint/raw", ds, files)
    VID = WS / "outputs" / "presentation" / "videos_p7" / "clip_dz"
    dev_clip = video_deviation(clip_anchor, bank_c, "libero_t*_ep*.mp4")
    del clip_anchor
    import torch; torch.cuda.empty_cache()
    dino = Dinov2Anchor(normalize=False)
    bank_d = demo_bank("dinov2-large/pre/raw", ds, files)
    VID = WS / "outputs" / "presentation" / "videos_p7" / "dinov2_dz"
    dev_dino = video_deviation(dino, bank_d, "libero_t*_ep*.mp4")

    for name, dev in (("clip_dz", dev_clip), ("dinov2_dz", dev_dino)):
        ok = np.concatenate([e["dist"] for e in dev if e["ok"]]) \
            if any(e["ok"] for e in dev) else np.array([])
        fail = np.concatenate([e["dist"] for e in dev if not e["ok"]]) \
            if any(not e["ok"] for e in dev) else np.array([])
        report["b_deviation"][name] = {
            "n_episodes": len(dev),
            "ok_median": float(np.median(ok)) if len(ok) else None,
            "fail_median": float(np.median(fail)) if len(fail) else None,
            "fail_p90": float(np.percentile(fail, 90)) if len(fail) else None}
        print(f"[dev] {name}: eps {len(dev)} | ok med "
              f"{report['b_deviation'][name]['ok_median']} | fail med "
              f"{report['b_deviation'][name]['fail_median']}")

    # 그림 1: 분포 (SNR 2패널 + 이탈도 1패널)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, (nuis, act) in dists_a.items():
        axes[0].hist(np.log10(nuis + 1e-9), bins=60, alpha=0.5, label=name,
                     density=True)
    axes[0].set(title="(a) nuisance 1-step ‖Δz‖ (log10)", xlabel="log10 ‖Δz‖")
    axes[0].legend()
    for name, (nuis, act) in dists_a.items():
        snr = report["a_nuisance_snr"][name]["snr"]
        axes[1].bar(name, snr)
        axes[1].text(name, snr, f"{snr:.2f}", ha="center", va="bottom")
    axes[1].set(title="(a) SNR = med(action Δz)/med(nuisance Δz)")
    for name, dev, color in (("CLIP-DZ", dev_clip, "#4477aa"),
                             ("DINOv2-DZ", dev_dino, "#ee6677")):
        alld = np.concatenate([e["dist"] for e in dev]) if dev else np.array([])
        if len(alld):
            axes[2].hist(alld, bins=50, alpha=0.5, label=name, density=True,
                         color=color)
    axes[2].set(title="(b) 방문-z kNN10 cos 거리 (영상 표본)", xlabel="1 − mean cos")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(WS / "outputs" / "report" / "p7_fig_dist.png", dpi=120)

    # 그림 2: 실패 에피소드 이탈도 시계열 예시 (각 4편)
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (name, dev) in zip(axes2, (("CLIP-DZ", dev_clip),
                                       ("DINOv2-DZ", dev_dino))):
        fails = [e for e in dev if not e["ok"]][:4]
        for e in fails:
            ax.plot(e["dist"], alpha=0.8, label=f"t{e['task']}ep{e['ep']}")
        ax.set(title=f"(c) {name} 실패 이탈도 궤적", xlabel="frame (×4 step)")
        ax.legend(fontsize=7)
    axes2[0].set_ylabel("1 − mean cos (kNN10)")
    fig2.tight_layout()
    fig2.savefig(WS / "outputs" / "report" / "p7_fig_traj.png", dpi=120)

    (WS / "outputs" / "report" / "p7_robustness.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False))
    print("저장: p7_robustness.json + p7_fig_dist.png + p7_fig_traj.png")


if __name__ == "__main__":
    main()
