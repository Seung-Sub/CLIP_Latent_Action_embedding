"""C8 zero-shot 언어→액션: hold-out 모션 문장 → h 디코딩 → 방향 정확도.

절차: hold-out 문장 20종(학습 미사용, 어휘·문체도 상이) → E_text → ζ로 간주 →
h(ζ, z_t) 디코딩 (val 에피소드에서 무작위 z_t M개) → 비정규화 청크의 지배축·방향이
문장 카테고리와 일치하는 비율. 그리퍼 문장은 그리퍼 이벤트 코드 일치로 판정.

스케일 주의: E_text는 단위노름, Δz는 노름 ≪1 (단위벡터 차) — h 입력 스케일 OOD 방지 위해
주 지표는 "val Δz 중앙노름으로 재스케일" 입력, 참고로 raw 입력도 병기 (모든 팔 동일 처리).

ARM-DZ는 낮게 나올 것으로 사전 등록됨 — 그대로 보고.

사용: python src/eval_libero/c8_zeroshot.py --ckpt checkpoints/grid/c8_arm_da.pt
"""
import argparse
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "src"))

import numpy as np
import torch
import yaml

from core.anchor import get_anchor
from data.libero import LiberoDataset
from data.motion_lang import MotionSentences, chunk_category
from models.networks import DeltaAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=str(WS / "configs" / "phase1_libero.yaml"))
    ap.add_argument("--n-states", type=int, default=100, help="z_t 표본 수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(WS / args.ckpt if not Path(args.ckpt).is_absolute()
                    else args.ckpt, map_location="cpu", weights_only=False)
    p1 = ck["config"]["model"]
    ae = DeltaAE(ck["action_dim"], ck["n_chunk"],
                 ck.get("latent_dim", p1["latent_dim"]), p1["hidden"],
                 p1["layers"], p1["dropout"], p1.get("state_cond", True),
                 align_mode=p1.get("align_mode", "dz")).to(device).eval()
    ae.load_state_dict(ck["state_dict"])
    a_mean, a_std = ck["a_mean"], ck["a_std"]
    align_mode = p1.get("align_mode", "dz")

    ds = LiberoDataset(cfg)
    clip = get_anchor(cfg)
    ms = MotionSentences()

    # val z_t 표본 + Δz 중앙노름 (train_phase1과 동일 시드 규약의 val split)
    rng = np.random.RandomState(cfg["train"]["seed"])
    files = ds.episode_files()
    perm = rng.permutation(len(files))
    v = cfg["data"]["val_episodes"]
    n_val = max(1, round(len(files) * v)) if v < 1 else int(v)
    val_files = [files[i] for i in perm[:n_val]]
    r2 = np.random.RandomState(args.seed)
    zs, dz_norms = [], []
    for ep in [val_files[i] for i in r2.choice(len(val_files),
                                               min(20, len(val_files)), False)]:
        Z = ds.embeddings(clip, ep)
        idx = r2.choice(len(Z) - ds.span, min(5, len(Z) - ds.span), False)
        zs.extend(Z[i] for i in idx)
        dz_norms.extend(np.linalg.norm(Z[i + ds.span] - Z[i]) for i in idx)
    Zt = torch.tensor(np.stack(zs[:args.n_states]), dtype=torch.float32,
                      device=device)
    dz_med = float(np.median(dz_norms))

    sents = [s for _, s in ms.holdout]
    cats = [c for c, _ in ms.holdout]
    E = clip.encode_texts(sents)["embeds"]                    # (20, 768) 단위노름
    results = {}
    for scale_name, scale in (("rescaled_dz_median", dz_med), ("raw_unit", 1.0)):
        T = torch.tensor(E * scale, dtype=torch.float32, device=device)
        hits, per_sent = 0, {}
        with torch.no_grad():
            for si in range(len(sents)):
                zeta = T[si:si + 1].expand(len(Zt), -1)
                chunks = ae.h(zeta, Zt).cpu().numpy() * a_std + a_mean
                ok = 0
                for ch in chunks:
                    cat, grip = chunk_category(ch)
                    if cats[si].startswith("grip"):
                        ok += (grip == 1) if cats[si] == "grip+" else (grip == 2)
                    else:
                        ok += cat.split("|")[0] == cats[si].split("|")[0]  # 축+방향
                per_sent[sents[si]] = {"target": cats[si],
                                       "dir_acc": ok / len(Zt)}
                hits += ok
        results[scale_name] = {
            "dir_acc_mean": hits / (len(sents) * len(Zt)),
            "per_sentence": per_sent}
        print(f"[{scale_name}] 방향 정확도: {100*hits/(len(sents)*len(Zt)):.1f}%")

    tag = Path(args.ckpt).stem
    out = WS / "outputs" / "report" / f"c8_zeroshot_{tag}.json"
    out.write_text(json.dumps({
        "ckpt": str(args.ckpt), "align_mode": align_mode,
        "n_holdout_sentences": len(sents), "n_states": len(Zt),
        "dz_median_norm": dz_med,
        "chance_level": "축·방향 6방향 가정 시 ~16.7% (분포 편중 고려 전)",
        **{k: {"dir_acc_mean": v["dir_acc_mean"],
               "per_sentence": v["per_sentence"]} for k, v in results.items()},
    }, indent=1, ensure_ascii=False))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
