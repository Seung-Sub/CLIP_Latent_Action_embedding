"""LIBERO 폐루프 평가 — suite의 각 태스크를 실시간 추론으로 수행, 성공률 측정.

루프 (20Hz, receding horizon):
  8스텝마다: agentview 렌더 → (방향 보정) → CLIP 인코딩 z_t
             → f(z_{t−16}, z_t, g(A_past)) → h(ζ̂, z_t) → 16스텝 → 앞 8스텝 실행
성공 판정: env.check_success() (LIBERO 표준, 태스크별 고정 초기상태 세트 사용)

참고: 데모와 env 렌더는 동일 방향임을 실측으로 확인 (공식 코드의 [::-1]은 영상표시용).

사용 (clip_libero env):
  MUJOCO_GL=egl python src/eval_libero/rollout_sim.py --suite libero_spatial --episodes 10
  MUJOCO_GL=egl python src/eval_libero/rollout_sim.py --task-id 0 --episodes 20 --save-video 2
"""
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "src"))

import argparse
import collections
import hashlib
import json
import os
import time

import numpy as np
import torch
import yaml
from PIL import Image

from core import chunkrep
from core.anchor import get_anchor
from data.libero import LiberoDataset
from eval_libero.rollout_dataset import load_models


def wilson(k, n, z=1.96):
    """이항 성공률 Wilson 95% CI."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def classify_failure(goal_state, init_pos, final_pos, target_max_lift,
                     grasp_cmd):
    """sim 상태 기반 실패 분류 (S1.v2 §3).

    규칙(순서 고정): 타깃 미변위·미리프트 → reach/grasp (그리퍼 명령 유무로 구분)
    / 타깃 미변위 + 타물체 대변위 → wrong_object / 타깃 리프트 후 목표 미충족 →
    wrong_goal / 그 외 → grasp. 임계: 변위 2cm, 리프트 5cm, 타물체 5cm.
    """
    try:
        target = goal_state[0][1]
    except (IndexError, TypeError):
        return "grasp" if grasp_cmd else "reach"
    t_disp = np.linalg.norm(final_pos.get(target, np.zeros(3))
                            - init_pos.get(target, np.zeros(3)))
    others_moved = any(
        np.linalg.norm(final_pos[o] - init_pos[o]) > 0.05
        for o in init_pos if o != target)
    if t_disp < 0.02 and target_max_lift < 0.05:
        if others_moved:
            return "wrong_object"
        return "grasp" if grasp_cmd else "reach"
    if target_max_lift >= 0.05:
        return "wrong_goal"        # 파지·리프트까지 성공, 배치 실패
    return "grasp"


def object_positions(obs):
    """obs에서 물체 위치 dict 추출 (robot·상대좌표 키 제외)."""
    return {k[:-4]: np.array(obs[k]) for k in obs
            if k.endswith("_pos") and not k.startswith("robot0")
            and "_to_" not in k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(WS / "configs" / "phase2_libero.yaml"))
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=None,
                    help="특정 태스크만 (기본: suite 전체)")
    ap.add_argument("--episodes", type=int, default=50, help="태스크당 롤아웃 수 (공표 관행 50)")
    ap.add_argument("--exec-horizon", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--wait-steps", type=int, default=10,
                    help="시작 물리 안정화 더미 스텝 (OpenVLA 관행 10, max_steps 불포함)")
    ap.add_argument("--save-fail-videos", type=int, default=2,
                    help="태스크당 실패 영상 저장 편수 (리포트 규약)")
    ap.add_argument("--run-id", default=None, help="§8 JSON run_id (기본: 자동 생성)")
    ap.add_argument("--data-variant", default="raw",
                    choices=["raw", "openvla_modified"],
                    help="학습 데이터 판본 — 모든 결과에 명기 의무")
    ap.add_argument("--phase", default="phase0", help="리포트 phase 필드")
    ap.add_argument("--notes", default="", help="리포트 notes 필드")
    ap.add_argument("--flip", action=argparse.BooleanOptionalAction, default=False,
                    help="env 렌더 상하반전 (실측: 데모와 동일 방향 — 기본 off)")
    ap.add_argument("--save-video", type=int, default=0)
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (ae, policy, a_mean, a_std, n_chunk, act_dim, use_lang,
     repr_kind, wrist_cam, proprio) = load_models(cfg, device)
    ds = LiberoDataset(cfg)          # span/resample 재사용
    clip = get_anchor(cfg)
    lang_enc = clip
    if use_lang and not clip.has_text:
        from core.anchor import ClipAnchor
        lang_enc = ClipAnchor()      # 교차 앵커 언어 폴백 (policy.lang_proj가 사영)
    span, H = ds.span, args.exec_horizon

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [args.task_id] if args.task_id is not None \
        else list(range(suite.get_num_tasks()))
    videos_dir = WS / "outputs" / "eval" / "videos"
    results = {}

    for tid in task_ids:
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl,
                                 camera_heights=128, camera_widths=128)
        init_states = suite.get_task_init_states(tid)
        lang = torch.tensor(lang_enc.encode_texts([task.language])["embeds"][0][None],
                            device=device) if use_lang else None
        if lang is not None and hasattr(policy, "lang_proj"):
            with torch.no_grad():
                lang = policy.lang_proj(lang)
        succ, infer_ms = [], []

        def frame(obs):
            img = obs["agentview_image"]
            return img[::-1].copy() if args.flip else img

        def encode(obs):
            return clip.encode_images([Image.fromarray(frame(obs))])["embeds"][0]

        def encode_wrist(obs):
            img = obs["robot0_eye_in_hand_image"]
            img = img[::-1].copy() if args.flip else img
            return clip.encode_images([Image.fromarray(img)])["embeds"][0]

        n_fail_saved = 0
        grasp_flags, ep_steps, fail_tags = [], [], []
        for ep in range(args.episodes):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(args.wait_steps):         # 물리 안정화 (OpenVLA 관행 10)
                obs, *_ = env.step([0.0] * 6 + [-1.0])
            init_pos = object_positions(obs)
            goal_state = env.env.parsed_problem.get("goal_state", []) \
                if hasattr(env.env, "parsed_problem") else []
            try:
                target_name = goal_state[0][1]
            except (IndexError, TypeError):
                target_name = None
            target_max_lift = 0.0
            rest = np.array([0.0] * 6 + [-1.0])
            past_actions = collections.deque([rest.copy() for _ in range(span)],
                                             maxlen=span)
            z_hist = collections.deque([encode(obs)], maxlen=span // H + 1)
            frames, done, t, grasp_cmd = [], False, 0, False
            with torch.no_grad():
                while t < args.max_steps and not done:
                    t0 = time.time()
                    past = ds.resample_chunk(np.stack(past_actions))
                    past = ((past - a_mean) / a_std).astype(np.float32)
                    past = chunkrep.to_repr(past, repr_kind)
                    zp = torch.tensor(z_hist[0][None], device=device)
                    zc = torch.tensor(z_hist[-1][None], device=device)
                    a_emb = ae.g(torch.tensor(past[None], device=device), zp)
                    toks = [zp, zc, a_emb] + ([lang] if use_lang else []) \
                        + ([torch.tensor(encode_wrist(obs)[None], device=device)]
                           if wrist_cam else [])
                    if proprio is not None:
                        _pmap = {"joint_states": "robot0_joint_pos",
                                 "gripper_states": "robot0_gripper_qpos"}
                        p = np.concatenate([obs[_pmap[f]]
                                            for f in proprio["fields"]])
                        p = (p - proprio["mean"]) / proprio["std"]
                        toks.append(policy.proprio_proj(
                            torch.tensor(p[None], dtype=torch.float32,
                                         device=device)))
                    zeta = policy(torch.stack(toks, dim=1))
                    ahat = chunkrep.from_repr(
                        ae.h(zeta, zc).cpu().numpy()[0], repr_kind) \
                        * a_std + a_mean
                    ahat = np.clip(ahat, -1.0, 1.0)
                    infer_ms.append((time.time() - t0) * 1000)
                    for k in range(min(H, args.max_steps - t)):
                        obs, r, done, info = env.step(ahat[k])
                        past_actions.append(ahat[k].copy())
                        grasp_cmd = grasp_cmd or ahat[k][-1] > 0
                        if target_name and f"{target_name}_pos" in obs:
                            lift = obs[f"{target_name}_pos"][2] \
                                - init_pos[target_name][2]
                            target_max_lift = max(target_max_lift, float(lift))
                        frames.append(frame(obs))    # 실패 영상 쿼터용으로 항상 기록
                        t += 1
                        if done:
                            break
                    z_hist.append(encode(obs))
            ok = bool(done)                          # LIBERO: done == success
            succ.append(ok)
            ep_steps.append(t)                       # SR@220 사후 산출용 (S1.v2 §1)
            grasp_flags.append(grasp_cmd)
            if not ok:
                fail_tags.append(classify_failure(
                    goal_state, init_pos, object_positions(obs),
                    target_max_lift, grasp_cmd))
            print(f"[task {tid}] ep {ep:2d} | {'SUCCESS' if ok else 'fail'} "
                  f"| steps {t} | 추론 {np.mean(infer_ms):.1f}ms", flush=True)
            save_this = (ep < args.save_video) or \
                (not ok and n_fail_saved < args.save_fail_videos)
            if save_this and frames:
                import imageio
                videos_dir.mkdir(parents=True, exist_ok=True)
                vp = videos_dir / f"libero_t{tid}_ep{ep}_{'ok' if ok else 'fail'}.mp4"
                imageio.mimsave(vp, frames, fps=20)
                if not ok:
                    n_fail_saved += 1
        env.close()
        k = int(np.sum(succ))
        lo, hi = wilson(k, len(succ))
        # 실패 모드: 그리퍼 닫기 명령조차 없음=reach / 있었으나 실패=grasp
        # (wrong_object/wrong_goal은 영상 수동 태깅 대상 — 자동 근사 불가)
        sr220 = float(np.mean([o and st <= 220
                               for o, st in zip(succ, ep_steps)]))
        results[tid] = {
            "sr": k / len(succ), "n": len(succ), "wilson_ci": [lo, hi],
            "sr_at_220": sr220,
            "episodes": [{"ok": bool(o), "steps": int(st)}
                         for o, st in zip(succ, ep_steps)],
            "language": task.language, "n_init_states": len(init_states),
            "failure_modes": {m: fail_tags.count(m)
                              for m in ("reach", "grasp",
                                        "wrong_object", "wrong_goal")},
        }
        print(f"== task {tid} [{task.language[:50]}]: {k}/{len(succ)} "
              f"= {100*k/len(succ):.0f}% (CI {100*lo:.0f}–{100*hi:.0f}%)", flush=True)

    srs = [r["sr"] for r in results.values()]
    K = sum(int(r["sr"] * r["n"]) for r in results.values())
    N = sum(r["n"] for r in results.values())
    s_lo, s_hi = wilson(K, N)
    print(f"\n=== {args.suite} | 태스크당 {args.episodes} 롤아웃 (paired init_states) ===")
    for tid, r in results.items():
        print(f"task {tid:2d}: {100*r['sr']:5.1f}%  "
              f"[{100*r['wilson_ci'][0]:.0f}–{100*r['wilson_ci'][1]:.0f}]  "
              f"{r['language'][:55]}")
    print(f"평균 성공률: {100*np.mean(srs):.1f}%  "
          f"(suite CI {100*s_lo:.1f}–{100*s_hi:.1f}%, {K}/{N})")

    # ---- §8 JSON 리포트 ----
    ck1 = torch.load(os.path.expanduser(cfg["phase1_ckpt"]),
                     map_location="cpu", weights_only=False)
    ck2 = torch.load(os.path.expanduser(cfg["train"]["checkpoint"]),
                     map_location="cpu", weights_only=False)
    p1m = ck1.get("metrics", {})
    run_id = args.run_id or (f"{args.phase}_{args.suite}"
                             f"_{args.data_variant}_s{cfg['train']['seed']}")
    anchor_info = ck1.get("anchor", {"id": clip.id, "cache_key": clip.cache_key})
    report = {
        "run_id": run_id, "phase": args.phase, "track": "libero",
        "condition": {"anchor": anchor_info.get("id"),
                      "anchor_cache_key": anchor_info.get("cache_key"),
                      "obs": anchor_info.get("id"),      # 현행: 앵커=관측 인코더 겸용
                      "lang": "L1" if use_lang else "L0",
                      "data_variant": args.data_variant, "extras": []},
        "suite": args.suite, "train_seed": cfg["train"]["seed"],
        "eval": {"n_per_task": args.episodes,
                 "per_task_sr": [round(r["sr"], 4) for r in results.values()],
                 "suite_sr": round(float(np.mean(srs)), 4),
                 "wilson_ci": [round(s_lo, 4), round(s_hi, 4)],
                 "max_steps": args.max_steps, "wait_steps": args.wait_steps,
                 "paired_init_states": True, "exec_horizon": H,
                 "per_task": {str(t): r for t, r in results.items()},
                 "sr_at_220": round(float(np.mean(
                     [r["sr_at_220"] for r in results.values()])), 4),
                 "failure_modes": {m: sum(r["failure_modes"][m] for r in results.values())
                                   for m in ("reach", "grasp", "wrong_object", "wrong_goal")},
                 "infer_ms_mean": round(float(np.mean(infer_ms)), 2)},
        "offline": {"phase1_r2": p1m.get("decoder_r2"),
                    "phase1_cycle_r2": p1m.get("cycle_r2"),
                    "retr_top1": (p1m.get("retrieval_a2z") or [None])[0],
                    "latcos": ck2.get("metrics", {}).get("lat_cos")},
        "train": {k: cfg["train"].get(k) for k in ("epochs", "lr", "batch_size")},
        "config_hash": hashlib.md5(
            json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12],
        "wandb": cfg.get("wandb", {}).get("run_name"),
        "notes": args.notes,
    }
    rep_dir = WS / "outputs" / "report"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep_path = rep_dir / f"{run_id}.json"
    rep_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"리포트 저장: {rep_path}")


if __name__ == "__main__":
    main()
