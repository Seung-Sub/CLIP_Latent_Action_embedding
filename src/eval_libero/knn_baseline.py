"""k-NN(VINN식) 바닥선 — 학습 없는 최근접 데모 청크 재생 (계획서 Phase 0.3).

각 재계획 시점: agentview 렌더 → CLIP z_t → 해당 태스크 데모 DB에서 cosine 최근접
k개 스텝의 액션 청크를 유사도 가중 평균 → 앞 H스텝 실행 (receding horizon).
정책 f·DeltaAE가 기여하는 부분을 분리하기 위한 '표현만으로 가능한 성능' 바닥선.

프로토콜: rollout_sim.py와 동일 (공식 init_states paired, wait 10, Wilson CI, §8 JSON).

사용 (clip_libero env):
  MUJOCO_GL=egl python src/eval_libero/knn_baseline.py --suite libero_spatial --episodes 50
"""
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "src"))

import argparse
import hashlib
import json
import os
import time

import numpy as np
import yaml
from PIL import Image

from core.anchor import get_anchor
from data.libero import LiberoDataset
from eval_libero.rollout_sim import wilson


def build_task_db(ds, clip, task_file):
    """태스크 파일의 전체 데모 → (Z (N,768), 청크 (N, n_chunk*D)) DB."""
    eps = [(task_file, k) for k in _demo_keys(task_file)]
    Zs, chunks = [], []
    for ep in eps:
        acts, Z = ds._filtered(clip, ep)
        T = len(acts)
        for t in range(0, T - ds.span):
            Zs.append(Z[t])
            chunks.append(ds.resample_chunk(acts[t:t + ds.span]).ravel())
    Z = np.stack(Zs).astype(np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    return Z, np.stack(chunks).astype(np.float32)


def _demo_keys(f):
    import h5py
    with h5py.File(f, "r") as h:
        return sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(WS / "configs" / "phase2_libero.yaml"))
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--k", type=int, default=5, help="최근접 이웃 수 (유사도 가중 평균)")
    ap.add_argument("--exec-horizon", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--wait-steps", type=int, default=10)
    ap.add_argument("--save-fail-videos", type=int, default=2)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--data-variant", default="raw",
                    choices=["raw", "openvla_modified"])
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import collections

    cfg = yaml.safe_load(open(args.config))
    ds = LiberoDataset(cfg)
    clip = get_anchor(cfg)
    span, H = ds.span, args.exec_horizon

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [args.task_id] if args.task_id is not None \
        else list(range(suite.get_num_tasks()))
    # 태스크 파일 매칭: 데모 파일명 = 태스크명 기반
    root = ds.roots[0]
    videos_dir = WS / "outputs" / "eval" / "videos"
    results, infer_ms = {}, []

    for tid in task_ids:
        task = suite.get_task(tid)
        task_file = root / f"{task.name}_demo.hdf5"
        if not task_file.exists():
            print(f"[task {tid}] 데모 파일 없음: {task_file} — skip (blocker 기록)")
            continue
        Zdb, Adb = build_task_db(ds, clip, task_file)
        print(f"[task {tid}] DB {len(Zdb)} steps | {task.language[:50]}")

        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl,
                                 camera_heights=128, camera_widths=128)
        init_states = suite.get_task_init_states(tid)
        succ, grasp_flags = [], []
        n_fail_saved = 0

        def encode(obs):
            z = clip.encode_images([Image.fromarray(obs["agentview_image"])])["embeds"][0]
            return z / (np.linalg.norm(z) + 1e-8)

        for ep in range(args.episodes):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(args.wait_steps):
                obs, *_ = env.step([0.0] * 6 + [-1.0])
            frames, done, t, grasp_cmd = [], False, 0, False
            while t < args.max_steps and not done:
                t0 = time.time()
                z = encode(obs)
                sim = Zdb @ z
                idx = np.argsort(-sim)[:args.k]
                wgt = np.maximum(sim[idx], 0) + 1e-8
                ahat = (Adb[idx] * wgt[:, None]).sum(0) / wgt.sum()
                ahat = np.clip(ahat.reshape(ds.n_chunk, -1), -1.0, 1.0)
                infer_ms.append((time.time() - t0) * 1000)
                for k_ in range(min(H, args.max_steps - t)):
                    obs, r, done, info = env.step(ahat[k_])
                    grasp_cmd = grasp_cmd or ahat[k_][-1] > 0
                    frames.append(obs["agentview_image"])
                    t += 1
                    if done:
                        break
            ok = bool(done)
            succ.append(ok); grasp_flags.append(grasp_cmd)
            print(f"[task {tid}] ep {ep:2d} | {'SUCCESS' if ok else 'fail'} | steps {t}",
                  flush=True)
            if not ok and n_fail_saved < args.save_fail_videos and frames:
                import imageio
                videos_dir.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(videos_dir / f"knn_t{tid}_ep{ep}_fail.mp4",
                                frames, fps=20)
                n_fail_saved += 1
        env.close()
        k_succ = int(np.sum(succ))
        lo, hi = wilson(k_succ, len(succ))
        fails = [(s, g) for s, g in zip(succ, grasp_flags) if not s]
        results[tid] = {
            "sr": k_succ / len(succ), "n": len(succ), "wilson_ci": [lo, hi],
            "language": task.language, "n_init_states": len(init_states),
            "failure_modes": {"reach": sum(1 for _, g in fails if not g),
                              "grasp": sum(1 for _, g in fails if g),
                              "wrong_object": 0, "wrong_goal": 0}}
        print(f"== task {tid}: {k_succ}/{len(succ)} = {100*k_succ/len(succ):.0f}% "
              f"(CI {100*lo:.0f}–{100*hi:.0f}%)", flush=True)

    srs = [r["sr"] for r in results.values()]
    K = sum(int(r["sr"] * r["n"]) for r in results.values())
    N = sum(r["n"] for r in results.values())
    s_lo, s_hi = wilson(K, N)
    print(f"\n=== k-NN(k={args.k}) {args.suite}: 평균 {100*np.mean(srs):.1f}% "
          f"(suite CI {100*s_lo:.1f}–{100*s_hi:.1f}%, {K}/{N}) ===")

    run_id = args.run_id or f"p0_knn{args.k}_{args.suite}_{args.data_variant}"
    report = {
        "run_id": run_id, "phase": "phase0", "track": "libero",
        "condition": {"anchor": clip.id, "anchor_cache_key": clip.cache_key,
                      "obs": clip.id, "lang": "L0(per-task DB)",
                      "data_variant": args.data_variant,
                      "extras": [f"knn_baseline_k{args.k}"]},
        "suite": args.suite, "train_seed": None,
        "eval": {"n_per_task": args.episodes,
                 "per_task_sr": [round(r["sr"], 4) for r in results.values()],
                 "suite_sr": round(float(np.mean(srs)), 4),
                 "wilson_ci": [round(s_lo, 4), round(s_hi, 4)],
                 "max_steps": args.max_steps, "wait_steps": args.wait_steps,
                 "paired_init_states": True, "exec_horizon": H,
                 "per_task": {str(t): r for t, r in results.items()},
                 "failure_modes": {m: sum(r["failure_modes"][m] for r in results.values())
                                   for m in ("reach", "grasp", "wrong_object", "wrong_goal")},
                 "infer_ms_mean": round(float(np.mean(infer_ms)), 2)},
        "offline": {}, "train": None,
        "config_hash": hashlib.md5(
            json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12],
        "wandb": None, "notes": args.notes,
    }
    rep_dir = WS / "outputs" / "report"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"{run_id}.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False))
    print(f"리포트 저장: {rep_dir / (run_id + '.json')}")


if __name__ == "__main__":
    main()
