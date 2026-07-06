"""Phase 1.5 D1 — no-op 필터 임계 ε 결정용 액션 norm 분포 분석.

계획서 규약: ε를 임의로 정하지 않고 분포 히스토그램을 먼저 뽑아 근거를 남긴다.
OpenVLA 공식 규칙(검증 노트 §1): is_noop = ‖a[:-1]‖₂ < ε AND 그리퍼 명령 직전과 동일 (ε=1e-4).

산출: outputs/report/d1_action_norm_hist.png + d1_action_norm_stats.json

사용 (clip_libero env):
  python src/diagnosis/d1_action_hist.py --root ~/clip_ws/data/libero/libero_spatial
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WS = Path(__file__).resolve().parents[2]


def collect_norms(root):
    """모든 데모의 스텝별 ‖a[:-1]‖₂ 와 그리퍼 변화 여부."""
    norms, grip_change = [], []
    per_demo_len = []
    for f in sorted(Path(root).expanduser().glob("*.hdf5")):
        with h5py.File(f, "r") as h:
            for k in h["data"]:
                a = h[f"data/{k}/actions"][:]
                n = np.linalg.norm(a[:, :-1], axis=1)
                gc = np.ones(len(a), bool)
                gc[1:] = a[1:, -1] != a[:-1, -1]      # 그리퍼 명령 변화
                norms.append(n)
                grip_change.append(gc)
                per_demo_len.append(len(a))
    return np.concatenate(norms), np.concatenate(grip_change), per_demo_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/clip_ws/data/libero/libero_spatial")
    ap.add_argument("--tag", default="libero_spatial_raw")
    args = ap.parse_args()

    norms, grip_change, lens = collect_norms(args.root)
    T = len(norms)
    print(f"steps: {T} (demos {len(lens)}, mean len {np.mean(lens):.0f})")

    # 후보 ε별 필터 비율 (OpenVLA 규칙: norm<ε AND 그리퍼 불변)
    stats = {"n_steps": int(T), "n_demos": len(lens),
             "mean_demo_len": float(np.mean(lens)),
             "norm_percentiles": {p: float(np.percentile(norms, p))
                                  for p in (1, 5, 10, 25, 50, 75, 90)},
             "filter_rate_by_eps": {}}
    for eps in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        noop = (norms < eps) & ~grip_change
        stats["filter_rate_by_eps"][f"{eps:.0e}"] = float(noop.mean())
        print(f"  ε={eps:.0e}: 필터 대상 {100*noop.mean():5.2f}% "
              f"(norm<ε만: {100*(norms<eps).mean():5.2f}%)")

    # 히스토그램 (log-x, 0은 별도 bin)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    nz = norms[norms > 0]
    axes[0].hist(np.log10(nz + 1e-12), bins=120, color="#4477aa")
    axes[0].axvline(np.log10(1e-4), color="r", ls="--", label="OpenVLA ε=1e-4")
    axes[0].set(xlabel="log10 ‖a[:-1]‖₂", ylabel="steps",
                title=f"action norm dist ({args.tag}) | zero-steps: "
                      f"{100*(norms==0).mean():.1f}%")
    axes[0].legend()
    axes[1].hist(np.log10(nz + 1e-12), bins=120, color="#4477aa", log=True)
    axes[1].axvline(np.log10(1e-4), color="r", ls="--")
    axes[1].set(xlabel="log10 ‖a[:-1]‖₂", ylabel="steps (log)", title="log-y")
    out_dir = WS / "outputs" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"d1_action_norm_hist_{args.tag}.png"
    fig.tight_layout(); fig.savefig(png, dpi=120)
    stats["zero_norm_rate"] = float((norms == 0).mean())
    js = out_dir / f"d1_action_norm_stats_{args.tag}.json"
    js.write_text(json.dumps(stats, indent=1))
    print(f"저장: {png}\n저장: {js}")


if __name__ == "__main__":
    main()
