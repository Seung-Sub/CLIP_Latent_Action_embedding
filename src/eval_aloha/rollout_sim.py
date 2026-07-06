"""시뮬레이션 폐루프 평가 — 실시간 추론으로 작업 수행, 성공률 측정.

루프 (50Hz, receding horizon):
  8스텝마다: angle 캠 렌더 → CLIP 인코딩 z_t → f(z_{t−16}, z_t, g(A_past)) → ζ̂
             → h(ζ̂, z_t) → 16스텝 액션 예측 → 앞 8스텝 실행 → 재예측
성공 판정: 에피소드 중 최대 reward == task.max_reward (수집 스크립트와 동일 기준)

사용 (clip env, MUJOCO_GL=egl 필요):
  MUJOCO_GL=egl python src/eval_aloha/rollout_sim.py --task sim_transfer_cube --episodes 50
  MUJOCO_GL=egl python src/eval_aloha/rollout_sim.py --task sim_insertion --episodes 50 --save-video 3
"""
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "src"))
sys.path.insert(0, str(WS / "aloha"))          # env/, utils/ (시뮬 환경)

import argparse
import collections
import os
import time

import numpy as np
import torch
import yaml
from PIL import Image

from core.anchor import get_anchor
from data.act_sim import ActSimDataset
from eval_aloha.rollout_dataset import load_models


def make_env(task):
    from env.sim_env import make_sim_env, BOX_POSE
    from utils.utils import sample_box_pose, sample_insertion_pose
    env = make_sim_env(task)
    def reset(seed):
        np.random.seed(seed)
        if "insertion" in task:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())
        else:
            BOX_POSE[0] = sample_box_pose()
        return env.reset()
    return env, reset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(WS / "configs" / "phase2.yaml"))
    ap.add_argument("--task", default="sim_transfer_cube",
                    choices=["sim_transfer_cube", "sim_insertion"])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--exec-horizon", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=10000, help="롤아웃 시드 시작값")
    ap.add_argument("--save-video", type=int, default=0, help="앞 N개 에피소드 mp4 저장")
    ap.add_argument("--onscreen", action="store_true",
                    help="실시간 창으로 롤아웃 표시 (데스크톱 세션 필요)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae, policy, a_mean, a_std, n_chunk, act_dim = load_models(cfg, device)
    ds = ActSimDataset(cfg)          # span/resample 유틸 재사용
    clip = get_anchor(cfg)
    env, reset = make_env(args.task)
    span, H = ds.span, args.exec_horizon

    def encode(ts):
        img = ts.observation["images"]["angle"]
        return clip.encode_images([Image.fromarray(img)])["embeds"][0]

    successes, rewards, infer_ms = [], [], []
    videos_dir = WS / "outputs" / "eval" / "videos"
    for ep in range(args.episodes):
        ts = env.reset() if False else reset(args.seed0 + ep)
        # 초기화: 과거청크 = 시작 qpos 반복, z 히스토리 = 시작 프레임
        qpos0 = ts.observation["qpos"][:act_dim]
        past_actions = collections.deque([qpos0.copy() for _ in range(span)],
                                         maxlen=span)
        z_hist = collections.deque([encode(ts)], maxlen=span // H + 1)
        frames = []
        viewer = None
        if args.onscreen:
            import matplotlib.pyplot as plt
            plt.ion()
            fig_v, ax_v = plt.subplots(figsize=(8, 6))
            viewer = ax_v.imshow(ts.observation["images"]["angle"])
            ax_v.set_title(f"ep {ep} | live rollout")
            ax_v.axis("off")
        max_reward, t = 0, 0

        def predict(z_prev, z_cur):
            """(z_{t−16}, z_t, 과거16액션) → 16스텝 액션 (실단위)."""
            past = ds.resample_chunk(np.stack(past_actions))
            past = ((past - a_mean) / a_std).astype(np.float32)
            zp = torch.tensor(z_prev[None], device=device)
            zc = torch.tensor(z_cur[None], device=device)
            a_emb = ae.g(torch.tensor(past[None], device=device), zp)
            zeta = policy(torch.stack([zp, zc, a_emb], dim=1))
            return ae.h(zeta, zc).cpu().numpy()[0] * a_std + a_mean

        def show(t):
            if ep < args.save_video:
                frames.append(ts.observation["images"]["angle"])
            if viewer is not None and t % 4 == 0:
                import matplotlib.pyplot as plt
                viewer.set_data(ts.observation["images"]["angle"])
                ax_v.set_title(f"ep {ep} | t={t} | reward {max_reward}/"
                               f"{env.task.max_reward}")
                plt.pause(0.001)

        with torch.no_grad():
            # receding horizon: 16 예측 → 앞 H 실행 → 재예측
            while t < args.max_steps:
                t0 = time.time()
                ahat = predict(z_hist[0], z_hist[-1])
                infer_ms.append((time.time() - t0) * 1000)
                for k in range(min(H, args.max_steps - t)):
                    ts = env.step(ahat[k])
                    past_actions.append(ahat[k].copy())
                    max_reward = max(max_reward, ts.reward or 0)
                    show(t)
                    t += 1
                z_hist.append(encode(ts))
        if viewer is not None:
            import matplotlib.pyplot as plt
            plt.close("all")
        success = max_reward == env.task.max_reward
        successes.append(success)
        rewards.append(max_reward)
        print(f"ep {ep:3d} | max_reward {max_reward}/{env.task.max_reward} | "
              f"{'SUCCESS' if success else 'fail'} | "
              f"추론 {np.mean(infer_ms):.1f}ms", flush=True)
        if ep < args.save_video and frames:
            import cv2
            videos_dir.mkdir(parents=True, exist_ok=True)
            vp = videos_dir / f"{args.task}_ep{ep}_{'ok' if success else 'fail'}.mp4"
            h, wd = frames[0].shape[:2]
            vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 50, (wd, h))
            for f in frames:
                vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            vw.release()
            print(f"  video: {vp}")

    sr = float(np.mean(successes)) * 100
    print(f"\n=== {args.task} | {args.episodes} rollouts ===")
    print(f"성공률: {sr:.1f}%  ({int(np.sum(successes))}/{args.episodes})")
    print(f"평균 max reward: {np.mean(rewards):.2f}/{env.task.max_reward}")
    print(f"추론 시간: {np.mean(infer_ms):.1f}ms/청크 "
          f"({1000/max(np.mean(infer_ms),1e-9)*H:.0f}Hz 상당, 제어 50Hz 요건 충족 여부 확인)")
    out = WS / "outputs" / "eval" / f"rollout_{args.task}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"task={args.task} episodes={args.episodes} "
                   f"success_rate={sr:.1f}% mean_reward={np.mean(rewards):.2f} "
                   f"infer_ms={np.mean(infer_ms):.1f}\n")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
