"""P6 지름길 프로브 (신설 표준) — 후보 토큰 '단독' 오프라인 정책의 val MAE.

정의(분석자 지시): 후보 토큰만 입력으로 받는 소형 정책을 학습해 val MAE를 측정.
시각 단독 대비 근접·우수하면 지름길 적신호 → 폐루프 전 드롭아웃/병목 설계 의무.

프로토콜: phase2와 동일 split(seed)·stride·정규화, 타깃 = 미래 청크(16×7 정규화),
프로브 = 2층 MLP(512) 30ep, MAE는 역정규화 후 phase2 리포트와 동일 규격(그리퍼 제외 평균).

사용:
  python src/diagnosis/p6_shortcut_probe.py --features proprio
  python src/diagnosis/p6_shortcut_probe.py --features z          # 시각 단독 기준선
  python src/diagnosis/p6_shortcut_probe.py --features gripper    # (a2) 대응 2D
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import yaml

from core.anchor import get_anchor
from data.libero import LiberoDataset

WS = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["proprio", "gripper", "z", "wrist"],
                    required=True)
    ap.add_argument("--config", default=str(WS / "configs" / "phase2_libero.yaml"))
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LiberoDataset(cfg)
    files = ds.episode_files()
    rng = np.random.RandomState(cfg["train"]["seed"])
    perm = rng.permutation(len(files))
    v = cfg["data"]["val_episodes"]
    n_val = max(1, round(len(files) * v)) if v < 1 else int(v)
    val_ids, tr_ids = perm[:n_val], perm[n_val:]
    stride = cfg["data"].get("stride", 2)

    # 특징 구성 (build_policy_samples와 동일 starts 정렬)
    if args.features in ("proprio", "gripper"):
        fields = ["gripper_states"] if args.features == "gripper" \
            else ["joint_states", "gripper_states"]
        X_eps = ds.build_proprio(files, stride=stride, fields=fields)
    else:
        clip = get_anchor(cfg)
        cam = None if args.features == "z" else "eye_in_hand_rgb"
        X_eps = []
        for ep in files:
            acts, Z = (ds._filtered(clip, ep) if cam is None
                       else (ds.load_actions(ep)[ds.keep_indices(ds.load_actions(ep))],
                             ds.embeddings(clip, ep, cam)[
                                 ds.keep_indices(ds.load_actions(ep))]))
            starts = list(range(0, len(acts) - ds.span, stride))
            X_eps.append(np.stack([Z[t] for t in starts]).astype(np.float32))

    # 타깃: 미래 청크 (정규화)
    Y_eps = []
    for ep in files:
        acts = ds.load_actions(ep)
        acts = acts[ds.keep_indices(acts)]
        starts = list(range(0, len(acts) - ds.span, stride))
        Y_eps.append(np.stack([ds.resample_chunk(acts[t:t + ds.span]).ravel()
                               for t in starts]).astype(np.float32))

    X_tr = np.concatenate([X_eps[i] for i in tr_ids])
    X_va = np.concatenate([X_eps[i] for i in val_ids])
    Y_tr = np.concatenate([Y_eps[i] for i in tr_ids])
    Y_va = np.concatenate([Y_eps[i] for i in val_ids])
    n_chunk = ds.n_chunk
    act_dim = Y_tr.shape[1] // n_chunk
    y_mean = Y_tr.reshape(-1, act_dim).mean(0)
    y_std = np.maximum(Y_tr.reshape(-1, act_dim).std(0), 1e-6)
    x_mean, x_std = X_tr.mean(0), np.maximum(X_tr.std(0), 1e-6)

    def nx(X): return ((X - x_mean) / x_std).astype(np.float32)
    def ny(Y): return ((Y.reshape(len(Y), n_chunk, act_dim) - y_mean) / y_std
                       ).reshape(len(Y), -1).astype(np.float32)

    Xt, Xv = torch.tensor(nx(X_tr), device=device), torch.tensor(nx(X_va), device=device)
    Yt, Yv = torch.tensor(ny(Y_tr), device=device), torch.tensor(ny(Y_va), device=device)

    torch.manual_seed(0)
    probe = nn.Sequential(nn.Linear(Xt.shape[1], 512), nn.GELU(),
                          nn.Linear(512, 512), nn.GELU(),
                          nn.Linear(512, Yt.shape[1])).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    idx = np.arange(len(Xt))
    for ep in range(args.epochs):
        np.random.shuffle(idx)
        for i in range(0, len(idx), 1024):
            b = idx[i:i + 1024]
            loss = nn.functional.l1_loss(probe(Xt[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(Xv).cpu().numpy()
    pr = pred.reshape(len(pred), n_chunk, act_dim) * y_std + y_mean
    gt = Y_va.reshape(len(Y_va), n_chunk, act_dim)
    arm = list(range(act_dim - 1))
    mae = float(np.abs(pr[:, :, arm] - gt[:, :, arm]).mean())
    grip_acc = float(((pr[:, :, -1] > 0) == (gt[:, :, -1] > 0)).mean() * 100)
    print(f"P6[{args.features}] 단독 val MAE {mae:.4f} | 그리퍼 {grip_acc:.1f}% "
          f"(dim {Xt.shape[1]}, {len(Xv)} val)")

    out = WS / "outputs" / "report" / "p6_shortcut_probe.json"
    data = json.loads(out.read_text()) if out.exists() else {}
    data[args.features] = {"val_mae": round(mae, 4), "grip_acc": round(grip_acc, 1),
                           "feat_dim": int(Xt.shape[1]), "n_val": int(len(Xv)),
                           "protocol": "2층 MLP512 30ep, phase2 동일 split/stride"}
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
