"""D2/데이터판본 — LIBERO 데모 고해상 재렌더 (검증 노트 §1 기반).

두 모드:
  state_rerender  [D2 전용]  데모의 states를 스텝별로 env에 주입해 256² 렌더.
                  스텝 구성이 원본과 1:1 동일 → 해상도 변수만 절제한 순수 비교.
  openvla_replay  [openvla_modified 판본] 공식 regenerate 스크립트 재현:
                  액션 재생 + 성공 에피소드만 유지 + no-op 스텝(ε=1e-4) 기록 제외 + 256².
                  (공식 RLDS판의 180° 회전은 OpenVLA 플랫폼 특이사항 — 비적용, 노트 참조)

산출: <out>/<task>_demo.hdf5 — data/demo_K/{obs/agentview_rgb(256²), actions[, states]}
      + 재생성 통계 json (제거 스텝·실패 데모 수)

사용 (clip_libero env):
  MUJOCO_GL=egl python src/diagnosis/d2_rerender.py --mode state_rerender \
      --suite libero_spatial --out ~/clip_ws/data/libero_raw256/libero_spatial
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np

WS = Path(__file__).resolve().parents[2]
NOOP_EPS = 1e-4          # OpenVLA regenerate 스크립트 상수 (검증 노트 §1)


def is_noop(action, prev_action=None, threshold=NOOP_EPS):
    """OpenVLA regenerate_libero_dataset.py 원문 규칙."""
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return (np.linalg.norm(action[:-1]) < threshold
            and action[-1] == prev_action[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["state_rerender", "openvla_replay"],
                    default="state_rerender")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--src", default="~/clip_ws/data/libero/libero_spatial")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--tasks", type=int, default=None, help="앞 N개 태스크만 (점검용)")
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    src = Path(os.path.expanduser(args.src))
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    stats = {"mode": args.mode, "resolution": args.resolution, "per_task": {}}

    n_tasks = args.tasks or suite.get_num_tasks()
    for tid in range(n_tasks):
        task = suite.get_task(tid)
        f_src = src / f"{task.name}_demo.hdf5"
        f_out = out / f"{task.name}_demo.hdf5"
        if f_out.exists():
            print(f"[{tid}] 이미 존재 — skip: {f_out.name}")
            continue
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl,
                                 camera_heights=args.resolution,
                                 camera_widths=args.resolution)
        kept, dropped_steps, failed_demos = 0, 0, 0
        with h5py.File(f_src, "r") as hs, h5py.File(f_out, "w") as ho:
            g = ho.create_group("data")
            demos = sorted(hs["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            for dk in demos:
                acts = hs[f"data/{dk}/actions"][:]
                states = hs[f"data/{dk}/states"][:]
                frames, keep_acts = [], []
                if args.mode == "state_rerender":
                    env.reset()
                    for t in range(len(acts)):
                        obs = env.set_init_state(states[t])
                        frames.append(obs["agentview_image"])
                        keep_acts.append(acts[t])
                    ok = True
                else:                                     # openvla_replay
                    env.reset()
                    obs = env.set_init_state(states[0])
                    for _ in range(5):                    # 공식 스크립트 안정화 관례
                        obs, *_ = env.step([0.0] * 6 + [-1.0])
                    done, prev = False, None
                    for t in range(len(acts)):
                        if is_noop(acts[t], prev):
                            dropped_steps += 1
                            prev = acts[t]
                            # no-op은 기록만 생략, 실행은 유지 (상태 일관성)
                            obs, r, done, info = env.step(acts[t])
                            continue
                        frames.append(obs["agentview_image"])
                        keep_acts.append(acts[t])
                        obs, r, done, info = env.step(acts[t])
                        prev = acts[t]
                    ok = bool(done)
                if not ok:
                    failed_demos += 1
                    continue
                gd = g.create_group(f"demo_{kept}")
                gd.create_dataset("obs/agentview_rgb",
                                  data=np.stack(frames).astype(np.uint8),
                                  compression="gzip", compression_opts=4)
                gd.create_dataset("actions", data=np.stack(keep_acts))
                kept += 1
        env.close()
        stats["per_task"][task.name] = {"kept_demos": kept,
                                        "failed_demos": failed_demos,
                                        "dropped_noop_steps": dropped_steps}
        print(f"[{tid}] {task.name[:45]}: kept {kept} | fail-filtered {failed_demos} "
              f"| noop-dropped {dropped_steps}", flush=True)

    sp = out / "regen_stats.json"
    sp.write_text(json.dumps(stats, indent=1))
    print(f"통계 저장: {sp}")


if __name__ == "__main__":
    main()
