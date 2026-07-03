# sim/ — ALOHA 시뮬 환경 (원본: /home/kist/act에서 최소 추출)
- env/    : MuJoCo 양팔 환경(sim_env, ee_sim_env) + 스크립트 정책 + 그리퍼 컨트롤러
- utils/  : constants.py(작업 설정 — 수집 카메라 ['angle']로 고정됨), utils.py(포즈 샘플링)
- assets/ : MuJoCo XML
- record_sim_episodes.py : 수집 (멀티워커, 성공만 저장, --start_idx로 이어붙이기)
- visualize_episodes.py  : 에피소드 → mp4 + qpos 플롯

수집: MUJOCO_GL=egl python record_sim_episodes.py --task_name sim_transfer_cube_record \
      --num_workers 5 --num_episodes 50 --dataset_dir ~/clipvp_ws/data/act_sim/sim_transfer_cube
