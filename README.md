# clip_ws — CLIP Visual-Policy 워크스페이스

frozen CLIP ViT-L/14 잠재공간에서 로봇 액션청크를 결합(Phase 1)하고,
그 잠재공간 위에서 미래 액션청크를 추론하는 정책(Phase 2)을 학습·평가하는 워크스페이스.
대상 작업: ALOHA 시뮬 양팔 조작 2종 — **transfer_cube**, **insertion** (50Hz, angle 단일 카메라).

## 모델 구조

```
Phase 1 (delta-AE):  g(A청크, z_t) ≈ Δz = z_{t+16} − z_t     [인코더: 1D-CNN]
                     h(Δz, z_t)  ≈ A청크                     [디코더: MLP]
Phase 2 (정책 f):    ζ̂ = f(z_{t−16}, z_t, g(A_past, z_{t−16}))   [MLP-concat d1024×4층]
                     Â_{t:t+16} = h(ζ̂, z_t)                  [동결 디코더로 복원]
추론(폐루프):        16스텝 예측 → 앞 8스텝 실행 → 재예측 (receding horizon)
```

- z = CLIP ViT-L/14 pooled 임베딩(768, frozen). 액션청크 = 16스텝 × 14관절(양팔 6+그리퍼 ×2) = 0.32초
- 손실: Phase1 = align(g≈Δz) + recon(L1) + cycle(L1) / Phase2 = act(L1, 주) + lat(잠재 GT, 보조)

## 0. 환경 설치 (최초 1회)

```bash
# 1) conda 환경 "clip" 생성 (torch cu128 포함 — RTX 5070 Ti급 Blackwell GPU 필수 빌드)
conda env create -f environment.yml
conda activate clip

# 2) CLIP 가중치 다운로드 (~1.6GB) — 경로는 configs/config.yaml의 clip.model_dir와 일치시킬 것
hf download openai/clip-vit-large-patch14 --local-dir ~/clip_ws/models/clip-vit-large-patch14
#   (이미 다른 경로에 있다면 configs/config.yaml의 model_dir만 수정)

# 3) wandb 로깅(선택): wandb login   — 안 하면 configs의 wandb.enabled를 false로
```

## 1~5. 전체 파이프라인

```bash
conda activate clip && cd ~/clip_ws

# 1. 데이터 취득 — 작업당 200 에피소드 (5워커 병렬, 성공 에피소드만 저장, ~15분)
cd aloha
MUJOCO_GL=egl python record_sim_episodes.py --task_name sim_transfer_cube_record \
    --num_workers 5 --num_episodes 200 --dataset_dir ~/clip_ws/data/act_sim/sim_transfer_cube
MUJOCO_GL=egl python record_sim_episodes.py --task_name sim_insertion_record \
    --num_workers 5 --num_episodes 200 --dataset_dir ~/clip_ws/data/act_sim/sim_insertion
cd ..
# ⚠️ 데이터를 재취득했다면 임베딩 캐시를 반드시 비울 것: rm -rf outputs/cache/act_sim_emb
# 에피소드 확인: cd aloha && python visualize_episodes.py --dataset_dir <dir> --episode_idx 0

# 2. Phase 1 학습 — 액션청크↔Δz 결합 AE (첫 실행 시 CLIP 인코딩 ~15분 + 학습 ~5분)
python src/training/train_phase1.py                    # 설정: configs/phase1.yaml

# 3. Phase 2 학습 — 잠재 정책 f (~3분, phase1 체크포인트 동결 사용)
python src/training/train_phase2.py                    # 설정: configs/phase2.yaml

# 4. GT 데이터셋 평가 — 전체 시계열 추론이 GT 그래프를 따라가는지 (14관절 플롯)
python src/eval/rollout_dataset.py --task sim_transfer_cube
python src/eval/rollout_dataset.py --task sim_insertion
# → outputs/eval/rollout_dataset_*.png

# 5. 시뮬레이션 평가 — 폐루프 실시간 추론 성공률
MUJOCO_GL=egl python src/eval/rollout_sim.py --task sim_transfer_cube --episodes 50
MUJOCO_GL=egl python src/eval/rollout_sim.py --task sim_insertion --episodes 50
#   옵션: --save-video 3 (앞 3편 mp4) / --onscreen (실시간 창) / --exec-horizon 8

# → outputs/eval/rollout_*.txt, videos/
```

전체 소요(재취득부터 평가까지): **약 1시간~1시간 30분** (롤아웃 25/50회 기준).

실험용 오버라이드: 모든 학습 스크립트는 `--set key=value --tag 이름` 지원
(예: `python src/training/train_phase2.py --set module.d_model=512 --tag d512`
→ `checkpoints/grid/`, `outputs/grid/<tag>.json`에 분리 저장).

## 디렉터리

| 폴더 | 내용 (각 폴더의 README.md 참조) |
|---|---|
| `configs/` | config.yaml(CLIP 경로) + phase1/phase2.yaml (확정 레시피, 근거 주석) |
| `src/core` | CLIP 래퍼 (pooled 768 임베딩) |
| `src/data` | act_sim HDF5 로더 — 임베딩 캐시, 학습쌍/삼중쌍 생성(경계 증강 포함) |
| `src/models` | DeltaAE(g/h), 정책 f(mlp/cls/pma) + 손실 |
| `src/training` | train_phase1.py / train_phase2.py |
| `src/eval` | rollout_dataset.py(GT 시계열 추론 그래프) / rollout_sim.py(폐루프 성공률) |
| `aloha/` | ALOHA MuJoCo 환경(2작업 전용 경량화) + 수집·시각화 스크립트 |
| `data/act_sim/` | 수집 데이터 (episode_N.hdf5: angle 이미지 + qpos + action, 50Hz) |
| `checkpoints/` | phase1_delta_ae.pt / phase2_policy.pt (확정 모델) |
| `outputs/` | cache(임베딩) / eval(그래프·영상·성공률) / grid(실험 런 기록) |

## 현재 성능 (2026-07-03, 작업당 200 eps 학습 기준)

| 평가 | transfer_cube |
|---|---|
| Phase1 복원 R² (held-out) | 0.981 |
| GT 시계열 추론 | 관절 MAE 0.22°, 그리퍼 98% (그래프가 GT 정밀 추종) |
| 폐루프 성공률 (25롤아웃) | **40%**, 평균 reward 2.04/4 |
| 추론 속도 | 스텝당 ~9ms (인코딩 8 + 정책 1) — 50Hz 실시간 여유 |

개선 이력: 0%(초기) → 경계샘플 증강+과거청크 노이즈 → pooled 단일공간 단순화 → **40%**.
(temporal ensemble은 구세대 정책의 결함을 가리던 목발로 판명 — 짝비교 28% vs 무-TE 40%로 제거됨)

## 알려진 이슈와 다음 개선 레버 (우선순위)

폐루프 실패가 이봉형(초기 큐브 정렬 성패가 결정) — 남은 병목은 시각의 물체 위치 정밀도 + 분포이동 잔여분.

1. **정책 시각입력 패치토큰화** — pooled 768은 공간 정보를 압축함 (이봉형 병목 정면 공략)
2. **proprio(qpos) 4번째 토큰** — 팔 상태를 직접 공급 (코드 수십 줄)
3. **DAgger 자동 루프** — 스크립트 전문가로 실패 상태 자동 재라벨 (시뮬 특권, 분포이동 정공법)
4. **데이터 증량** (200→1000/작업 — 수집 ~9분/작업이라 저비용)
5. flow matching 디코딩 (스크립트 데이터라 멀티모달리티 낮음 — 후순위)

연구 이력(그리드 200여 런, 보고서, 기각 목록): `~/clip_openx_ws/experiments/` 참조.
