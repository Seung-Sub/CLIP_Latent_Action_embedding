"""S1.v2 §2 — D4 정밀화: E[Var(A|·)] 조건 3종 {z}, {z+z_wrist}, {z+z_wrist+proprio}.

조건부 정보가 늘수록 잔여 액션 분산이 얼마나 줄어드는지 → 관측 병목 vs 본질적
다봉성 분해. proprio = joint_states(7)+gripper_states(2) (데모·env 공통 확보 가능 9D).

특징 결합: 각 블록을 L2 정규화 후 concat (등가중) — z·z_wrist는 CLIP 단위노름,
proprio는 표준화 후 정규화. 이웃 = 동일 태스크·타 에피소드 cosine kNN (D4 원판 동일).

GPU 미사용 (체인 경합 금지): 임베딩은 캐시 히트 전용 스텁으로 로드, CLIP 미로딩.

사용 (clip_libero env): python src/diagnosis/d4_refined.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import yaml

from data.libero import LiberoDataset
from diagnosis.d4_multimodality import knn_conditional_variance

WS = Path(__file__).resolve().parents[2]


class CacheOnlyEncoder:
    """캐시 히트 전용 — encode 호출 시 즉시 실패 (GPU 로딩 회피 검증 겸용)."""
    cache_key = "clip-vit-l-14/joint/norm"

    def encode_images(self, *_):
        raise RuntimeError("캐시 미스 — GPU 경합 금지 조건에서 인코딩 불가")


def load_proprio(ep, starts):
    path, demo = ep
    with h5py.File(path, "r") as h:
        j = h[f"data/{demo}/obs/joint_states"][:]
        g = h[f"data/{demo}/obs/gripper_states"][:]
    P = np.concatenate([j, g], axis=1)
    return np.stack([P[t] for t in starts])


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def main():
    cfg = yaml.safe_load(open(WS / "configs" / "phase1_libero.yaml"))
    ds = LiberoDataset(cfg)
    enc = CacheOnlyEncoder()
    files = ds.episode_files()

    Zs, Ws, Ps, As, Gs = [], [], [], [], []
    for i, ep in enumerate(files):
        acts = ds.load_actions(ep)
        Z = ds.embeddings(enc, ep)
        Zw = ds.embeddings(enc, ep, "eye_in_hand_rgb")
        starts = list(range(0, len(acts) - ds.span, ds.stride))
        Zs.append(np.stack([Z[t] for t in starts]))
        Ws.append(np.stack([Zw[t] for t in starts]))
        Ps.append(load_proprio(ep, starts))
        As.append(np.stack([ds.resample_chunk(acts[t:t + ds.span]).ravel()
                            for t in starts]))
        Gs.append(np.stack([[str(ep[0].stem)] * len(starts),
                            [str(i)] * len(starts)], axis=1))
    Z, W, P = np.concatenate(Zs), np.concatenate(Ws), np.concatenate(Ps)
    A, G = np.concatenate(As), np.concatenate(Gs)
    A = (A - A.mean(0)) / np.maximum(A.std(0), 1e-6)
    P = (P - P.mean(0)) / np.maximum(P.std(0), 1e-6)     # 표준화 후 블록 정규화
    print(f"samples {len(Z)} | 태스크 {len(np.unique(G[:, 0]))}")

    conds = {
        "z": unit(Z),
        "z+wrist": np.concatenate([unit(Z), unit(W)], axis=1),
        "z+wrist+proprio": np.concatenate([unit(Z), unit(W), unit(P)], axis=1),
    }
    out = {"k": 10, "n_samples": int(len(Z)),
           "feature_combine": "블록별 L2 정규화 후 concat (등가중)",
           "proprio_def": "joint_states(7)+gripper_states(2) 표준화",
           "conditions": {}}
    for name, F in conds.items():
        ev, total = knn_conditional_variance(F, A, G, k=10)
        ceiling = 1 - ev / total
        out["conditions"][name] = {"e_var_a_given_c": round(ev, 4),
                                   "r2_ceiling_estimate": round(ceiling, 4)}
        print(f"{name:18s} E[Var(A|·)] = {ev:.4f} → 상한 ≈ {ceiling:.3f}")

    p = WS / "outputs" / "report" / "d4_refined_conditions.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"저장: {p}")


if __name__ == "__main__":
    main()
