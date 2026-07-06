"""C8 모션 문장 생성기의 bin 경계 근거 — 청크 축별 누적 변위 분포 분석.

청크(16스텝×7D)의 축별 누적 변위 |Σa[:,k]| 분포 → 지배축 판정 마진과
크기 2-bin(small/large) 경계를 데이터에서 결정하고 근거를 기록한다.

산출: outputs/report/c8_chunk_axis_stats.json + .png
사용 (clip_libero env): python src/diagnosis/c8_chunk_axis_stats.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from data.libero import LiberoDataset

WS = Path(__file__).resolve().parents[2]
AXES = ["x", "y", "z", "roll", "pitch", "yaw"]


def main():
    cfg = yaml.safe_load(open(WS / "configs" / "phase1_libero.yaml"))
    ds = LiberoDataset(cfg)
    files = ds.episode_files()
    cums, grip_events = [], 0
    n_chunks = 0
    for ep in files:
        acts = ds.load_actions(ep)
        for t in range(0, len(acts) - ds.span, ds.stride):
            seg = ds.resample_chunk(acts[t:t + ds.span])
            cums.append(seg[:, :6].sum(0))                 # 축별 누적 변위
            grip_events += int((np.diff(np.sign(seg[:, 6])) != 0).any())
            n_chunks += 1
    C = np.abs(np.stack(cums))                             # (N, 6)
    dom = C.argmax(1)
    dom_mag = C.max(1)
    # 지배 마진: 1등/2등 비율 (문장의 축 단정이 정당한 정도)
    sorted_C = np.sort(C, 1)
    margin = sorted_C[:, -1] / (sorted_C[:, -2] + 1e-9)

    stats = {
        "n_chunks": n_chunks,
        "gripper_event_rate": grip_events / n_chunks,
        "dominant_axis_freq": {AXES[i]: float((dom == i).mean()) for i in range(6)},
        "dominant_margin_median": float(np.median(margin)),
        "dominant_magnitude_percentiles": {
            p: float(np.percentile(dom_mag, p)) for p in (10, 25, 50, 75, 90)},
        "per_axis_cum_median": {AXES[i]: float(np.median(C[:, i])) for i in range(6)},
        "bin_boundary_decision": {
            "magnitude_2bin": float(np.median(dom_mag)),
            "rationale": "지배축 누적 변위의 중앙값으로 small/large 이분 — 두 bin의 "
                         "표본 수가 균형(50/50)이고 스케일 의존 임계 자의성 제거. "
                         "회전축은 pos와 단위가 다르므로 축별이 아닌 지배축 크기 기준 "
                         "단일 경계 사용(문장은 지배축만 서술).",
        },
    }
    fig, axes_ = plt.subplots(1, 2, figsize=(12, 4))
    axes_[0].hist(dom_mag, bins=80, color="#4477aa")
    axes_[0].axvline(np.median(dom_mag), color="r", ls="--",
                     label=f"median {np.median(dom_mag):.2f} = bin 경계")
    axes_[0].set(xlabel="|cum| of dominant axis", ylabel="chunks",
                 title="dominant-axis magnitude")
    axes_[0].legend()
    axes_[1].bar(AXES, [(dom == i).mean() for i in range(6)], color="#4477aa")
    axes_[1].set(title=f"dominant axis freq (grip event rate "
                       f"{stats['gripper_event_rate']:.2f})")
    out = WS / "outputs" / "report"
    fig.tight_layout(); fig.savefig(out / "c8_chunk_axis_stats.png", dpi=120)
    (out / "c8_chunk_axis_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
