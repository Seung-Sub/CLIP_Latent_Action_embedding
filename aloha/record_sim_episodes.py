# MuJoCo EGL 렌더링 설정 (headless GPU 렌더링)
# 실행 전 환경변수 설정 필요:
#   export MUJOCO_GL=egl
#   export PYOPENGL_PLATFORM=egl
# 또는 스크립트 실행 시:
#   MUJOCO_GL=egl python3 record_sim_episodes.py ...
#
# 사용 예시:
#   # 기본: 5워커 × 10개 = 50 에피소드, transfer_cube, seed 42
#   python3 record_sim_episodes.py
#
#   # insertion 작업
#   python3 record_sim_episodes.py --task insertion
#
#   # 1워커 + 화면 표시 (테스트용)
#   python3 record_sim_episodes.py --num_workers 1 --num_episodes 3 --onscreen_render
#
#   # 커스텀 저장 경로
#   python3 record_sim_episodes.py --dataset_dir /path/to/data

import time
import os
import numpy as np
import argparse
import h5py
import multiprocessing as mp

from utils.constants import SIM_TASK_CONFIGS, DATA_DIR
from utils.utils import sample_box_pose, sample_insertion_pose

import IPython
e = IPython.embed

MAX_RETRIES = 30


def record_worker(worker_id, task_name, dataset_dir, episode_indices,
                  camera_names, episode_len,
                  policy_cls_name, onscreen_render, render_cam_name,
                  action_noise_scale=0.0, base_seed=0, max_retries=None):
    """단일 워커: 할당된 에피소드를 순차 녹화. 실패 시 재시도하여 성공만 저장."""
    if onscreen_render:
        import matplotlib.pyplot as plt
    from env.ee_sim_env import make_ee_sim_env
    from env.sim_env import make_sim_env, BOX_POSE
    from env.gripper_controller import GripperController

    if policy_cls_name == 'PickAndTransferPolicy':
        from env.scripted_policy import PickAndTransferPolicy as policy_cls
    else:
        from env.scripted_policy import InsertionPolicy as policy_cls

    inject_noise = False
    retries = max_retries if max_retries is not None else MAX_RETRIES
    # 통계: (episode_idx, attempts_needed)
    stats = []

    for episode_idx in episode_indices:
        saved = False

        for attempt in range(retries):
            # 시도마다 다른 랜덤 시드 → 다른 포즈 생성
            np.random.seed(base_seed * 100000 + episode_idx * 1000 + attempt)

            # --- Phase 1: EE space scripted policy rollout ---
            env = make_ee_sim_env(task_name)
            ts = env.reset()
            episode = [ts]
            policy = policy_cls(inject_noise)

            if onscreen_render:
                plt.ion()
                fig, ax = plt.subplots()
                plt_img = ax.imshow(ts.observation['images'][render_cam_name])
                ax.set_title(f'Phase 1 (EE) — {render_cam_name}')
                plt.show()
                plt.pause(0.1)

            for step in range(episode_len):
                action = policy(ts)
                left_cmd, right_cmd = policy.get_current_binary_states()
                env.task.set_binary_gripper_commands(left_cmd, right_cmd)
                ts = env.step(action)
                episode.append(ts)
                if onscreen_render:
                    plt_img.set_data(ts.observation['images'][render_cam_name])
                    plt.pause(0.02)

            if onscreen_render:
                plt.close()

            episode_return = np.sum([ts.reward for ts in episode[1:]])
            episode_max_reward = np.max([ts.reward for ts in episode[1:]])
            episode_final_reward = episode[-1].reward

            if 'sim_transfer_cube' in task_name \
               and episode_max_reward == env.task.max_reward and episode_final_reward != env.task.max_reward:
                episode_max_reward -= 1

            # Phase 1 (EE) 성공 불필요 — Phase 2 (Joint replay)에서만 판별

            # Phase 1 완료 → trajectory 추출
            action_traj = []
            binary_gripper_traj = []
            for ts in episode:
                qpos = ts.observation['qpos'].copy()
                action_traj.append(qpos)
                binary_gripper_traj.append((qpos[6], qpos[13]))

            subtask_info = episode[0].observation['env_state'].copy()
            del env, episode, policy

            # --- Phase 2: Joint space replay ---
            env = make_sim_env(task_name)
            BOX_POSE[0] = subtask_info
            ts = env.reset()

            replay_gripper_controller = GripperController(open_duration=50, close_duration=50)

            episode_replay = [ts]
            final_action_traj = []
            prev_left_cmd = 1.0
            prev_right_cmd = 1.0

            if onscreen_render:
                plt.ion()
                fig, ax = plt.subplots()
                plt_img = ax.imshow(ts.observation['images'][render_cam_name])
                ax.set_title(f'Phase 2 (Replay) — {render_cam_name}')
                plt.show()
                plt.pause(0.1)

            for t in range(len(action_traj)):
                target_qpos = action_traj[t]
                left_cmd, right_cmd = binary_gripper_traj[t]

                left_smooth = replay_gripper_controller.process_gripper('left', prev_left_cmd, left_cmd, t)
                right_smooth = replay_gripper_controller.process_gripper('right', prev_right_cmd, right_cmd, t)
                prev_left_cmd = left_cmd
                prev_right_cmd = right_cmd

                action_for_sim = np.concatenate([target_qpos[:6], [left_smooth], target_qpos[7:13], [right_smooth]])

                if action_noise_scale > 0:
                    arm_noise = np.random.normal(0, action_noise_scale, size=12)
                    action_for_sim[:6] += arm_noise[:6]
                    action_for_sim[7:13] += arm_noise[6:]

                action_for_save = np.concatenate([action_for_sim[:6], [left_cmd], action_for_sim[7:13], [right_cmd]])
                final_action_traj.append(action_for_save)

                ts = env.step(action_for_sim)
                episode_replay.append(ts)
                if onscreen_render:
                    plt_img.set_data(ts.observation['images'][render_cam_name])
                    plt.pause(0.02)

            if onscreen_render:
                plt.close()

            episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
            episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
            episode_final_reward = episode_replay[-1].reward

            if 'sim_transfer_cube' in task_name \
               and episode_max_reward == env.task.max_reward and episode_final_reward != env.task.max_reward:
                episode_max_reward -= 1

            if episode_max_reward != env.task.max_reward:
                print(f"[Worker {worker_id}] episode_idx={episode_idx} Replay Failed "
                      f"(attempt {attempt+1}, max={episode_max_reward}, final={episode_final_reward})")
                del env, episode_replay
                continue  # 재시도

            # --- Phase 2 성공 → Phase 3: Save HDF5 ---
            print(f"[Worker {worker_id}] episode_idx={episode_idx} Successful (attempt {attempt+1}), {episode_return=}")

            data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
            }
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'] = []

            final_action_traj = final_action_traj[:-1]
            episode_replay = episode_replay[:-1]
            episode_replay = episode_replay[1:]

            max_timesteps = len(final_action_traj)
            while final_action_traj:
                action = final_action_traj.pop(0)
                ts = episode_replay.pop(0)
                data_dict['/observations/qpos'].append(ts.observation['qpos'])
                data_dict['/observations/qvel'].append(ts.observation['qvel'])
                data_dict['/action'].append(action)
                for cam_name in camera_names:
                    data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

            t0 = time.time()
            dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                             chunks=(1, 480, 640, 3), compression='gzip', compression_opts=9)
                obs.create_dataset('qpos', (max_timesteps, 14))
                obs.create_dataset('qvel', (max_timesteps, 14))
                root.create_dataset('action', (max_timesteps, 14))

                for name, array in data_dict.items():
                    root[name][...] = array
            print(f'[Worker {worker_id}] episode_{episode_idx} saved ({time.time() - t0:.1f}s)')

            del env, episode_replay
            saved = True
            break  # 성공 → 다음 에피소드로

        stats.append((episode_idx, attempt + 1 if saved else -1))
        if not saved:
            print(f"[Worker {worker_id}] episode_idx={episode_idx} 최대 재시도({retries}) 초과!")

    return worker_id, stats


def main(args):
    """
    Generate demonstration data in simulation.
    Runs num_workers parallel MuJoCo environments, each recording a chunk of episodes.
    실패 에피소드는 재시도하여 성공 에피소드만 저장.
    """
    base_seed = args['seed']
    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    num_workers = args['num_workers']
    onscreen_render = args['onscreen_render']
    render_cam_name = args['render_cam']
    action_noise_scale = args['action_noise_scale']
    start_idx = args.get('start_idx', 0)

    # onscreen_render는 반드시 단일 워커
    if onscreen_render and num_workers > 1:
        print(f'[INFO] --onscreen_render 사용 → num_workers를 1로 변경')
        num_workers = 1

    # matplotlib 백엔드 설정
    if not onscreen_render:
        import matplotlib
        matplotlib.use('Agg')

    # task 설정
    if 'transfer' in task_name:
        policy_cls_name = 'PickAndTransferPolicy'
    else:
        policy_cls_name = 'InsertionPolicy'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']

    # 워커별 에피소드 분배 (연속 블록, start_idx 부터)
    base_chunk = num_episodes // num_workers
    remainder = num_episodes % num_workers
    chunks = []
    start = start_idx
    for w in range(num_workers):
        size = base_chunk + (1 if w < remainder else 0)
        chunks.append(list(range(start, start + size)))
        start += size

    print(f'\n{"=" * 60}')
    print(f' Recording {num_episodes} episodes | {num_workers} workers | seed={base_seed}')
    print(f' Task: {task_name}')
    print(f' Cameras: {camera_names}')
    print(f' Dataset: {dataset_dir}')
    print(f' Onscreen render: {onscreen_render}')
    print(f' 실패 시 재시도 (최대 {MAX_RETRIES}회)')
    print(f'{"=" * 60}')
    for w, chunk in enumerate(chunks):
        if chunk:
            print(f'  Worker {w}: episodes {chunk[0]}~{chunk[-1]} ({len(chunk)}개)')
    print()

    t_start = time.time()

    max_retries = args.get('max_retries')

    if num_workers == 1:
        _, all_stats = record_worker(
            0, task_name, dataset_dir, list(range(start_idx, start_idx + num_episodes)),
            camera_names, episode_len,
            policy_cls_name, onscreen_render, render_cam_name,
            action_noise_scale, base_seed, max_retries)
    else:
        pool = mp.Pool(processes=num_workers)
        results = []
        for w in range(num_workers):
            if not chunks[w]:
                continue
            r = pool.apply_async(record_worker,
                                 (w, task_name, dataset_dir, chunks[w],
                                  camera_names, episode_len,
                                  policy_cls_name, False, render_cam_name,
                                  action_noise_scale, base_seed, max_retries))
            results.append(r)
        pool.close()
        pool.join()

        all_stats = []
        for r in results:
            _, stats = r.get()
            all_stats.extend(stats)

    elapsed = time.time() - t_start

    # 통계 계산
    all_stats.sort(key=lambda x: x[0])  # episode_idx 순 정렬
    total = len(all_stats)
    first_try = sum(1 for _, a in all_stats if a == 1)
    retried = sum(1 for _, a in all_stats if a > 1)
    failed = sum(1 for _, a in all_stats if a == -1)
    saved = total - failed
    total_attempts = sum(a if a > 0 else MAX_RETRIES for _, a in all_stats)
    first_try_sr = first_try / total * 100 if total else 0

    # 결과 출력
    print(f'\n{"=" * 60}')
    print(f' 녹화 완료: {dataset_dir}')
    print(f' 저장: {saved}/{total}개 (1회 성공 {first_try} / 재시도 성공 {retried} / 실패 {failed})')
    print(f' 1회차 성공률: {first_try_sr:.1f}% ({first_try}/{total})')
    print(f' 총 시도 횟수: {total_attempts}회')
    print(f' 소요 시간: {elapsed:.1f}s ({elapsed/60:.1f}min)')
    if retried > 0:
        retry_details = [(idx, a) for idx, a in all_stats if a > 1]
        print(f' 재시도 에피소드: {[(idx, f"{a}회") for idx, a in retry_details]}')
    if failed > 0:
        failed_indices = [idx for idx, a in all_stats if a == -1]
        print(f' [ERROR] 최종 실패 에피소드: {failed_indices}')
    print(f'{"=" * 60}')

    # 성공률 txt 저장
    report_path = os.path.join(dataset_dir, 'record_stats.txt')
    with open(report_path, 'w') as f:
        f.write(f'task: {task_name}\n')
        f.write(f'seed: {base_seed}\n')
        f.write(f'num_episodes: {total}\n')
        f.write(f'saved: {saved}\n')
        f.write(f'first_try_success: {first_try}\n')
        f.write(f'retried_success: {retried}\n')
        f.write(f'failed: {failed}\n')
        f.write(f'first_try_sr: {first_try_sr:.1f}%\n')
        f.write(f'total_attempts: {total_attempts}\n')
        f.write(f'elapsed_sec: {elapsed:.1f}\n')
        f.write(f'\n--- per episode ---\n')
        for idx, attempts in all_stats:
            status = 'OK' if attempts > 0 else 'FAIL'
            f.write(f'episode_{idx}: {status} (attempts={attempts if attempts > 0 else MAX_RETRIES})\n')
    print(f' 통계 저장: {report_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sim 데이터 녹화 (병렬 MuJoCo)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python3 record_sim_episodes.py --task_name sim_transfer_cube_record --dataset_dir /path/to/data
  python3 record_sim_episodes.py --task_name sim_insertion_record --dataset_dir /path/to/data
  python3 record_sim_episodes.py --task_name sim_insertion_record --dataset_dir /path/to/data --onscreen_render --num_episodes 3
""")
    parser.add_argument('--task_name', type=str, default='sim_transfer_cube_record',
                        choices=['sim_transfer_cube_record', 'sim_insertion_record'],
                        help='작업 종류 (default: sim_transfer_cube_record)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='저장할 episode 번호 시작값 (default 0). e.g. --start_idx 30 --num_episodes 30 → episode_30~episode_59')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='저장 경로')
    parser.add_argument('--num_episodes', type=int, default=50,
                        help='총 에피소드 수 (default: 50)')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='병렬 워커 수 (default: 5)')
    parser.add_argument('--seed', type=int, default=0,
                        help='랜덤 시드 (default: 0)')
    parser.add_argument('--onscreen_render', action='store_true',
                        help='화면 렌더링 (자동으로 num_workers=1)')
    parser.add_argument('--render_cam', type=str, default='angle',
                        help='렌더링 카메라 (default: angle)')
    parser.add_argument('--action_noise_scale', type=float, default=0.0,
                        help='가우시안 관절 노이즈 표준편차 (rad, default: 0.0, 추천: 0.01)')
    parser.add_argument('--max_retries', type=int, default=None,
                        help=f'에피소드 별 최대 재시도 횟수 (default: {MAX_RETRIES}). 빠른 디버깅 시 작게.')

    main(vars(parser.parse_args()))
