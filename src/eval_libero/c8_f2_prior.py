"""S1.v2 F2 — 소형 prior: E_text → ζ 2층 MLP (unCLIP prior 레시피 축소판).

학습 쌍: train 청크별 (E_text(할당 문장), ζ = g(chunk, z_t)) — 문장당 다수 ζ
→ MLP는 문장 조건부 평균 ζ를 학습 (손실 MSE + 0.5(1−cos)).
평가: hold-out 문장 20종 → prior → h(ζ̂, z_t) → 방향 정확도 (기준 ≥50%).
F1(무학습 보정) 실패 후 단계 — 팔별 독립 학습 (ζ 분포가 팔마다 다름).

사용: python src/eval_libero/c8_f2_prior.py
"""
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "src"))

import numpy as np
import torch
import torch.nn as nn
import yaml

from core.anchor import get_anchor
from data.libero import LiberoDataset
from data.motion_lang import MotionSentences, chunk_category
from models.networks import DeltaAE

ARMS = ["dz", "da", "hy01", "hy03"]


def load_arm(arm, device):
    ck = torch.load(WS / f"checkpoints/grid/c8_arm_{arm}.pt",
                    map_location="cpu", weights_only=False)
    p1 = ck["config"]["model"]
    ae = DeltaAE(ck["action_dim"], ck["n_chunk"],
                 ck.get("latent_dim", p1["latent_dim"]), p1["hidden"],
                 p1["layers"], p1["dropout"], p1.get("state_cond", True),
                 align_mode=p1.get("align_mode", "dz")).to(device).eval()
    ae.load_state_dict(ck["state_dict"])
    return ae, ck


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(WS / "configs" / "phase1_libero.yaml"))
    ds = LiberoDataset(cfg)
    clip = get_anchor(cfg)
    ms = MotionSentences()

    rng = np.random.RandomState(cfg["train"]["seed"])
    files = ds.episode_files()
    perm = rng.permutation(len(files))
    n_val = round(len(files) * cfg["data"]["val_episodes"])
    tr_files = [files[i] for i in perm[n_val:]]
    va_files = [files[i] for i in perm[:n_val]]
    eps_tr = ds.build(clip, tr_files, verbose=False)
    eps_va = ds.build(clip, va_files, verbose=False)
    Zt_tr = np.concatenate([e[0] for e in eps_tr])
    A_tr = np.concatenate([e[2] for e in eps_tr])
    Zt_va = np.concatenate([e[0] for e in eps_va])

    S_all = ms.embed_all(clip)
    holdout_sents = [s for _, s in ms.holdout]
    holdout_cats = [c for c, _ in ms.holdout]
    S_hold = clip.encode_texts(holdout_sents)["embeds"]

    r2 = np.random.RandomState(0)
    Zs = torch.tensor(Zt_va[r2.choice(len(Zt_va), 100, False)], device=device)

    results = {}
    for arm in ARMS:
        ae, ck = load_arm(arm, device)
        n_chunk, act_dim = ck["n_chunk"], ck["action_dim"]
        C = ((A_tr.reshape(len(A_tr), n_chunk, act_dim) - ck["a_mean"])
             / ck["a_std"]).astype(np.float32)
        ids = ms.assign(A_tr.reshape(-1, n_chunk, act_dim))
        with torch.no_grad():
            Zeta = []
            for i in range(0, len(C), 4096):
                Zeta.append(ae.g(torch.tensor(C[i:i+4096], device=device),
                                 torch.tensor(Zt_tr[i:i+4096], device=device)))
            Zeta = torch.cat(Zeta)
        X = torch.tensor(S_all[ids], device=device)       # (N, 768)

        prior = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 1024),
                              nn.GELU(), nn.Linear(1024, 768)).to(device)
        opt = torch.optim.Adam(prior.parameters(), lr=1e-3)
        idx = np.arange(len(X))
        for ep in range(30):
            np.random.shuffle(idx)
            for i in range(0, len(idx), 1024):
                b = idx[i:i+1024]
                pred = prior(X[b])
                tgt = Zeta[b]
                cos = nn.functional.cosine_similarity(pred, tgt, dim=1)
                loss = nn.functional.mse_loss(pred, tgt) + 0.5 * (1 - cos).mean()
                opt.zero_grad(); loss.backward(); opt.step()

        prior.eval()
        hits = total = 0
        per_sent = {}
        with torch.no_grad():
            zeta_h = prior(torch.tensor(S_hold, dtype=torch.float32,
                                        device=device))
            for si in range(len(holdout_sents)):
                zeta = zeta_h[si:si+1].expand(len(Zs), -1)
                chunks = ae.h(zeta, Zs).cpu().numpy() * ck["a_std"] + ck["a_mean"]
                ok = 0
                for ch in chunks:
                    cat, grip = chunk_category(ch)
                    if holdout_cats[si].startswith("grip"):
                        ok += (grip == 1) if holdout_cats[si] == "grip+" else (grip == 2)
                    else:
                        ok += cat.split("|")[0] == holdout_cats[si].split("|")[0]
                hits += ok; total += len(Zs)
                per_sent[holdout_sents[si]] = round(ok / len(Zs), 3)
        acc = hits / total
        results[arm] = {"dir_acc": round(acc, 4), "per_sentence": per_sent,
                        "prior": "LN-768-1024-GELU-768, 30ep, MSE+0.5(1-cos)"}
        print(f"[{arm:5s}] F2 방향정확도 {100*acc:.1f}% "
              f"({'PASS' if acc >= 0.5 else 'below'} 기준 50%)")
        del ae; torch.cuda.empty_cache()

    p = WS / "outputs" / "report" / "c8_gapfix_f2.json"
    p.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"저장: {p}")


if __name__ == "__main__":
    main()
