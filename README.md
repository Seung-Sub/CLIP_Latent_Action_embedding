# clipvp_ws — CLIP Visual-Policy 워크스페이스

frozen CLIP ViT-L/14 잠재공간에서 로봇 액션청크를 결합(Phase 1)하고,
그 잠재공간 위에서 미래 액션을 추론하는 정책(Phase 2)을 학습·평가하는 전용 워크스페이스.

## 모델 구조 (한눈에)

```
Phase 1 (delta-AE):  g(A청크, z_t) ≈ Δz = z_{t+16} − z_t     [인코더]
                     h(Δ패치, z_t) ≈ A청크                    [디코더]
Phase 2 (정책 f):    ζ̂ = f(z_{t−16}, z_t, g(A_past))          [MLP-concat]
                     Â_{t:t+16} = h(g2dec(ζ̂), z_t)            [동결 디코더로 복원]
```
- 데이터: ALOHA 시뮬 2작업 (transfer_cube, insertion) × 200 eps, 50Hz, angle 단일캠, 16스텝 청크
- 확정 성능: Phase1 복원 R² 0.985 / Phase2 관절 MAE 0.42°(오프라인), 잠재 cos 0.97

## 파이프라인 5단계 (전부 clipx conda env)

```bash
conda activate clipx && cd ~/clipvp_ws

# 1. 데이터 취득 (시뮬, 성공 에피소드만 저장)
cd sim && MUJOCO_GL=egl python record_sim_episodes.py --task_name sim_transfer_cube_record \
    --num_workers 5 --num_episodes 200 --dataset_dir ~/clipvp_ws/data/act_sim/sim_transfer_cube; cd ..

# 2. Phase 1 학습 (액션↔Δz 결합 AE)
python src/training/train_phase1.py            # configs/phase1.yaml

# 3. Phase 2 학습 (잠재 정책 f)
python src/training/train_phase2.py            # configs/phase2.yaml

# 4. GT 데이터셋 평가 (전체 시계열 추론 → 14차원 그래프)
python src/eval/eval_gt_trace.py --task sim_transfer_cube

# 5. 시뮬레이션 평가 (폐루프 실시간 추론 → 성공률)
MUJOCO_GL=egl python src/eval/rollout_sim.py --task sim_transfer_cube --episodes 50 \
    --temporal-ensemble          # ACT식 매스텝 예측+지수가중 평균 (권장 — 첫 성공 달성)
#   --onscreen                   # 실시간 창 관전
#   --save-video 3               # 앞 3개 에피소드 mp4
```

## 디렉터리

| 폴더 | 내용 (각 폴더의 README.md 참조) |
|---|---|
| `configs/` | config.yaml(CLIP 공통) + phase1/phase2.yaml (확정 레시피, 근거 주석) |
| `src/` | core(CLIP 래퍼) / data(로더) / models(네트워크) / training / eval |
| `sim/` | ALOHA MuJoCo 환경 + 데이터 수집·시각화 스크립트 |
| `data/act_sim/` | 수집 데이터 (→ clip_openx_ws 심볼릭 링크, 5GB) |
| `checkpoints/` | phase1_delta_ae.pt / phase2_policy.pt (확정 모델) |
| `outputs/` | cache(임베딩, 심링크) / eval(그래프·영상·성공률) / grid(런 기록) |

## 현재 상태와 알려진 이슈 (2026-07-03)

- **오프라인 평가는 우수**: GT 이미지로 전체 시계열 추론 시 관절 MAE 0.22°, 그리퍼 98%,
  14차원 그래프가 GT를 정밀 추종 (`outputs/eval/gt_trace_*.png`)
- **폐루프 진행 중**: 개선 이력 — 초기 0/4 → 경계샘플+노이즈 주입 후 1.0/4 →
  **temporal ensemble 적용 후 첫 완전 성공(4/4)**. 정확한 성공률 측정 중
- 다음 후보(효과 예상순): ① proprio(qpos) 4번째 토큰 추가 ② 노이즈 스케일·z 교란 증강
  ③ 재예측 주기 단축(8→4) ④ 정책 시각입력을 패치토큰으로 ⑤ flow matching 디코딩
- 실험 이력·연구 배경: `~/clip_openx_ws/experiments/` (연구용 워크스페이스)
