"""D4 — 인간 데모 다봉성 정량화: 결정론 디코더의 R² 이론 상한 추정.

원리: R²_ceiling ≈ 1 − E[Var(A | z)] / Var(A).
z_t가 같아도 액션이 다르면(다봉) 어떤 결정론 f(z)도 그 분산만큼은 설명 불가.
Var(A|z)는 z-공간 k-NN(동일 태스크·타 데모 이웃)의 청크 분산으로 추정.

대조군: ALOHA(스크립트 전문가, R² 0.98 레짐)에 동일 추정 → 추정기 자체 검증.
결과는 flow/CVAE 디코더 도입(계획 §5 4.3)의 근거 수치가 된다.

사용:
  (clip_libero env) python src/diagnosis/d4_multimodality.py --track libero
  (clip env)        python src/diagnosis/d4_multimodality.py --track aloha
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from core.anchor import get_anchor
from data import get_dataset

WS = Path(__file__).resolve().parents[2]


def knn_conditional_variance(Z, A, groups, k=10):
    """샘플별: 같은 그룹(태스크) 내 타 에피소드 z-이웃 k개의 액션 분산.

    returns (E[Var(A|z)] (정규화 액션공간, 차원 평균), Var(A) 동일 규격)
    """
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    cond_vars = []
    rng = np.random.RandomState(0)
    MAX_G = 4000                            # 그룹당 상한 (유사도 행렬 메모리 보호)
    for g in np.unique(groups[:, 0]):
        m = groups[:, 0] == g
        Zg, Ag, ep_g = Zn[m], A[m], groups[m, 1]
        if len(Zg) > MAX_G:
            sel = rng.choice(len(Zg), MAX_G, replace=False)
            Zg, Ag, ep_g = Zg[sel], Ag[sel], ep_g[sel]
        S = Zg @ Zg.T                       # cosine 유사도
        # 같은 에피소드 이웃 제외 (시간 인접 = 사실상 동일 샘플 → 분산 과소추정 방지)
        same_ep = ep_g[:, None] == ep_g[None, :]
        S[same_ep] = -np.inf
        idx = np.argsort(-S, axis=1)[:, :k]
        neigh = Ag[idx]                     # (n, k, D)
        cond_vars.append(neigh.var(axis=1).mean(axis=1))   # 차원 평균 분산
    ev = float(np.concatenate(cond_vars).mean())
    total = float(A.var(axis=0).mean())
    return ev, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["libero", "aloha"], default="libero")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-eps-per-group", type=int, default=50)
    args = ap.parse_args()

    cfg_file = WS / "configs" / ("phase1_libero.yaml" if args.track == "libero"
                                 else "phase1.yaml")
    cfg = yaml.safe_load(open(cfg_file))
    ds = get_dataset(cfg)
    clip = get_anchor(cfg)
    files = ds.episode_files()

    # 그룹 = 태스크 (libero: hdf5 파일 / aloha: 데이터셋 루트)
    def group_of(ep):
        if args.track == "libero":
            return str(ep[0].stem)
        return str(ep.parent.name)

    print(f"{args.track}: {len(files)} episodes 임베딩 로드/인코딩 중...")
    eps = ds.build(clip, files, verbose=False)
    Zs, As, Gs = [], [], []
    for i, (ep, (Zt, _, A)) in enumerate(zip(files, eps)):
        Zs.append(Zt); As.append(A)
        Gs.append(np.stack([[group_of(ep)] * len(Zt),
                            [str(i)] * len(Zt)], axis=1))
    Z = np.concatenate(Zs)
    A = np.concatenate(As)
    G = np.concatenate(Gs)
    # 액션 정규화 (전체 통계 — R² 규격과 일치)
    A = A.reshape(len(A), -1)
    A = (A - A.mean(0)) / np.maximum(A.std(0), 1e-6)
    print(f"samples: {len(Z)} | groups: {len(np.unique(G[:, 0]))}")

    ev, total = knn_conditional_variance(Z, A, G, k=args.k)
    ceiling = 1 - ev / total
    print(f"\nE[Var(A|z)] = {ev:.4f} | Var(A) = {total:.4f}")
    print(f"결정론 디코더 R² 이론 상한(추정) ≈ {ceiling:.3f}  (k={args.k})")

    out = WS / "outputs" / "report" / f"d4_multimodality_{args.track}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "track": args.track, "k": args.k, "n_samples": int(len(Z)),
        "e_var_a_given_z": ev, "var_a": total,
        "r2_ceiling_estimate": ceiling,
        "note": "kNN(z-cos, 동일 태스크·타 에피소드) 조건부 분산 기반. "
                "이웃 반경 유한 → 상한의 보수적(낮은 쪽) 추정치. "
                "aloha(스크립트) 수치가 추정기 검증 대조군."}, indent=1))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
