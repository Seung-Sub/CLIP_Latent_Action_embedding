"""S1.v2 F0+F1 — zero-shot 디코드 갭 진단 및 무학습 보정.

F0 진단: 노름·센트로이드 통계 (E_text vs g출력 vs Δz) — "E_text가 h 입력 매니폴드
밖" 해석의 정량 근거.
F1 보정: E'_text = (E_text − μ_text + μ_g), 노름을 ‖Δz‖ 중앙값으로 재스케일
→ h 디코딩 → hold-out 방향 정확도 재측정. 성공 기준 ≥50% (우연 ~15%).

μ_text = train 문장 330종 임베딩 평균 / μ_g = val 청크 g출력 평균 (팔별).
사용: python src/eval_libero/c8_gapfix.py   (4팔 일괄, 결과 JSON + 표 출력)
"""
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

    # ---- val 분할 (train_phase1 시드 규약 재현) ----
    rng = np.random.RandomState(cfg["train"]["seed"])
    files = ds.episode_files()
    perm = rng.permutation(len(files))
    n_val = round(len(files) * cfg["data"]["val_episodes"])
    val_files = [files[i] for i in perm[:n_val]]
    eps = ds.build(clip, val_files, verbose=False)
    Zt = np.concatenate([e[0] for e in eps])
    Ztn = np.concatenate([e[1] for e in eps])
    A = np.concatenate([e[2] for e in eps])
    Dz = Ztn - Zt
    dz_norms = np.linalg.norm(Dz, axis=1)

    # ---- 텍스트 임베딩 (train 330 + hold-out 20) ----
    S_train = ms.embed_all(clip)
    holdout_sents = [s for _, s in ms.holdout]
    holdout_cats = [c for c, _ in ms.holdout]
    S_hold = clip.encode_texts(holdout_sents)["embeds"]
    mu_text = S_train.mean(0)

    # z_t 표본 (디코딩 조건, 팔 공통)
    r2 = np.random.RandomState(0)
    zidx = r2.choice(len(Zt), 100, False)
    Zs = torch.tensor(Zt[zidx], device=device)

    report = {"F0": {
        "dz_norm": {"median": float(np.median(dz_norms)),
                    "mean": float(dz_norms.mean()), "std": float(dz_norms.std())},
        "text_norm": 1.0, "n_val_pairs": int(len(Zt)),
        "mu_text_norm": float(np.linalg.norm(mu_text)),
    }, "F1": {}}

    n_chunk = None
    for arm in ARMS:
        ae, ck = load_arm(arm, device)
        n_chunk, act_dim = ck["n_chunk"], ck["action_dim"]
        C = ((A.reshape(len(A), n_chunk, act_dim) - ck["a_mean"]) / ck["a_std"]
             ).astype(np.float32)
        with torch.no_grad():
            G = ae.g(torch.tensor(C, device=device),
                     torch.tensor(Zt, device=device)).cpu().numpy()
        g_norms = np.linalg.norm(G, axis=1)
        mu_g = G.mean(0)
        cos_mu = float(mu_text @ mu_g /
                       (np.linalg.norm(mu_text) * np.linalg.norm(mu_g) + 1e-9))
        report["F0"][f"arm_{arm}"] = {
            "g_norm_median": float(np.median(g_norms)),
            "mu_g_norm": float(np.linalg.norm(mu_g)),
            "cos(mu_text, mu_g)": round(cos_mu, 4),
            "norm_ratio_text_over_g": round(1.0 / float(np.median(g_norms)), 1),
        }

        # ---- F1: 재중심 + 재스케일 → hold-out 디코딩 ----
        med_dz = float(np.median(dz_norms))
        E = S_hold - mu_text + mu_g                       # 재중심
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8) * med_dz
        hits = total = 0
        per_cat = {}
        with torch.no_grad():
            for si in range(len(holdout_sents)):
                zeta = torch.tensor(E[si:si + 1], dtype=torch.float32,
                                    device=device).expand(len(Zs), -1)
                chunks = ae.h(zeta, Zs).cpu().numpy() * ck["a_std"] + ck["a_mean"]
                ok = 0
                for ch in chunks:
                    cat, grip = chunk_category(ch)
                    if holdout_cats[si].startswith("grip"):
                        ok += (grip == 1) if holdout_cats[si] == "grip+" else (grip == 2)
                    else:
                        ok += cat.split("|")[0] == holdout_cats[si].split("|")[0]
                hits += ok; total += len(Zs)
                per_cat[holdout_sents[si]] = round(ok / len(Zs), 3)
        acc = hits / total
        report["F1"][f"arm_{arm}"] = {"dir_acc": round(acc, 4),
                                      "per_sentence": per_cat}
        print(f"[{arm:5s}] F0: ‖g‖중앙 {np.median(g_norms):.4f} "
              f"(text 1.0 = {1/np.median(g_norms):.0f}배) | cos(μt,μg) {cos_mu:+.3f} "
              f"|| F1 방향정확도 {100*acc:.1f}%")
        del ae
        torch.cuda.empty_cache()

    p = WS / "outputs" / "report" / "c8_gapfix_f0f1.json"
    p.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"저장: {p}")


if __name__ == "__main__":
    main()
