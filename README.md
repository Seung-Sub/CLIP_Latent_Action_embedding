# clip_ws — CLIP Visual-Policy 워크스페이스

frozen CLIP ViT-L/14 잠재공간에서 로봇 액션청크를 결합(Phase 1)하고,
그 잠재공간 위에서 미래 액션청크를 추론하는 정책(Phase 2)을 학습·평가하는 워크스페이스.
평가 트랙 2개: **ALOHA**(이 문서 — transfer_cube/insertion, 50Hz) ·
**LIBERO**(→ README_libero.md — Franka 7D, 인간 데모, 언어 지시문). 공용층(core/models/training)은 동일.

정책 모델 2계열을 지원하며 **전부 config로 선택**됩니다 (코드 수정 불필요):

| 계열 | `module.name` | 설명 | 성능 (캠페인 실측) |
|---|---|---|---|
| 회귀 MLP (베이스라인) | `mlp` | 토큰 concat → MLP → ζ̂ 직접 회귀 | aloha 평균 35.5% / LIBERO-Spatial 36.5% |
| **Flow matching (기본)** | `flow` | CFM 속도장 + Euler 6스텝, 출발점 = 직전 액션 잠재(A2A식) | **LIBERO 80.0%** (+손목캠·d1536) / aloha는 하단 표 참조 |

상세 절제·근거: `docs/upgrade_report.md` (2026-07-04 캠페인 보고서), 시간순 일지: `docs/upgrade_journal.md`

## 모델 구조

```
Phase 1 (delta-AE):  g(A청크, z_t) ≈ Δz = z_{t+16} − z_t     [인코더: 1D-CNN]
                     h(Δz, z_t)  ≈ A청크                     [디코더: MLP]
                     ※ DCT 청크 표현(chunk_repr: dct)은 연구 보류 — 기본 time, 코드만 유지
Phase 2 (정책 f):    토큰 [z_{t−16}, z_t, g(A_past), (lang), (wrist)] → ζ̂
                     - mlp : concat-MLP 회귀
                     - flow: ctx 인코더(토큰 요약) + 속도장 v(x, ctx, t), x₀=g(A_past)에서 6스텝 적분
                     Â_{t:t+16} = h(ζ̂, z_t)                  [동결 디코더로 복원]
추론(폐루프):        16스텝 예측 → 앞 8스텝 실행 → 재예측 (receding horizon, H=8 최적 실측)
```

- z = CLIP ViT-L/14 pooled 임베딩(768, frozen). 액션청크 = 16스텝 × 14관절(양팔 6+그리퍼 ×2) = 0.32초
- 손실: Phase1 = align + recon + cycle / Phase2 = mlp는 act(L1)+lat(보조),
  flow는 **CFM(속도장 MSE) + FLD**(ODE 샘플 ζ̂를 동결 h로 디코딩한 액션 L1)

### Phase2 module 설정 키 (전 트랙 공통)

```yaml
module:
  name: flow            # mlp(베이스라인) | flow(권장) — 실험용 3종(cls/pma/resmlp)은 제거됨
  d_model: 1024         # aloha 승자 1024 / LIBERO 승자 1536
  layers: 4             # flow에선 v_net 잔차블록 수
  ctx_layers: 4         # flow 전용: ctx 인코더 블록 수 (aloha 승자 4, LIBERO 2)
  flow_source: past     # past(A2A식, 권장) | noise(π0식) | vision(VITA식) — 절제에서 past 일관 승
  flow_steps: 6         # Euler 적분 스텝 (12는 무이득 실측)
  source_noise: 0.1     # 학습 시 x₀ 교란 (x0_std 상대배율)
  lang_token: true      # LIBERO 전용 (CLIP 텍스트 임베딩 4번째 토큰)
  wrist_token: true     # 손목캠 토큰 (data.wrist_camera 필요 — 현재 LIBERO만 데이터 보유)
```

평가·롤아웃 스크립트는 **체크포인트에 저장된 config에서 구조를 자동 복원**하므로,
학습만 올바른 설정으로 하면 평가 명령은 모델 계열과 무관하게 동일합니다.

## 0. 환경 설치 (최초 1회)

```bash
# 1) conda 환경 "clip" 생성 (torch cu128 포함 — RTX 5070 Ti급 Blackwell GPU 필수 빌드)
conda env create -f environment.yml
conda activate clip

# 2) CLIP 가중치 다운로드 (~1.6GB) — 경로는 configs/config.yaml의 clip.model_dir와 일치시킬 것
hf download openai/clip-vit-large-patch14 --local-dir ~/clip_ws/models/clip-vit-large-patch14

# 3) wandb 로깅(선택): wandb login   — 안 하면 configs의 wandb.enabled를 false로
```

## 1~5. 전체 파이프라인 (ALOHA 트랙)

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

# 2. Phase 1 — 기본 = SOTA (DCT 표현, configs/phase1.yaml)
python src/training/train_phase1.py

# 3. Phase 2 — 기본 = SOTA (flow+ctx4, configs/phase2.yaml)
python src/training/train_phase2.py
#   베이스라인(mlp+time): --config configs/phase2_mlp.yaml (phase1도 time ckpt를 참조함)
#   실험 분리 저장: --tag 이름 → checkpoints/grid/<이름>.pt

# 4. GT 데이터셋 평가 (14관절 플롯) → outputs/eval/rollout_dataset_*.png
python src/eval_aloha/rollout_dataset.py --task sim_transfer_cube

# 5. 폐루프 성공률 (체크포인트가 구조를 결정 — mlp/flow 동일 명령)
MUJOCO_GL=egl python src/eval_aloha/rollout_sim.py --task sim_transfer_cube --episodes 50
MUJOCO_GL=egl python src/eval_aloha/rollout_sim.py --task sim_insertion --episodes 50
#   옵션: --save-video 3 / --onscreen / --exec-horizon 8(기본·최적)
```

**승자 체크포인트가 기본 경로에 승격되어 있어 재학습 없이 바로 평가 가능**:
기본 config가 곧 SOTA(`checkpoints/phase2_aloha_fm.pt`, 100롤 실측 transfer 67%/insertion 34%)이므로
위 5번 명령을 그대로 실행하면 됩니다. 베이스라인 평가는 `--config configs/phase2_mlp.yaml`.

실험용 오버라이드: 모든 학습 스크립트는 `--set key=value --tag 이름` 지원.
다중 arm 자동 실행(학습→평가→기록)은 `outputs/campaign/run_arms.py` + arm yaml 참조.

## 디렉터리

| 폴더 | 내용 (각 폴더의 README.md 참조) |
|---|---|
| `configs/` | config.yaml(CLIP 경로) + phase1/phase2*.yaml — **기본 = SOTA**, `*_mlp.yaml` = 베이스라인 |
| `src/core` | CLIP 래퍼 + `chunkrep.py`(청크 표현: time/DCT — phase1 ckpt가 결정, 하류 자동 추종) |
| `src/data` | act_sim/libero HDF5 로더 — 임베딩 캐시, 학습쌍/삼중쌍 생성(경계 증강, 손목캠 배열 포함) |
| `src/models` | DeltaAE(g/h) + 정책 `policy.py`(mlp/cls/pma/resmlp/**flow**) + 손실 |
| `src/training` | train_phase1.py / train_phase2.py (mlp·flow 분기, 손목캠·언어 토큰 지원) |
| `src/eval_aloha` `src/eval_libero` | GT 그래프 / 폐루프 성공률 (ckpt config에서 구조 자동 복원) |
| `aloha/` | ALOHA MuJoCo 환경(경량화) + 수집·시각화 (left/right_wrist 카메라 XML 보유 — 렌더 미연결) |
| `data/` | act_sim(angle, 50Hz) / libero(spatial·object·goal, agentview+eye_in_hand, 20Hz) |
| `checkpoints/` | **SOTA**: phase1_aloha_dct / phase2_aloha_fm / phase2_libero_fm · 베이스라인: phase1_delta_ae / phase2_policy / phase2_libero · `grid/`(캠페인 arm) |
| `outputs/campaign/` | 업그레이드 캠페인: 러너(run_arms.py)·arm yaml·평가 config(cfg/)·결과(jsonl/txt)·로그 |
| `docs/` | **upgrade_report.md**(캠페인 최종 보고서) · upgrade_journal.md(일지) — git 제외 영역 |

## 현재 성능 (기본 config 기준 실측)

| 벤치마크 | 베이스라인(mlp) | **기본(flow 레시피)** |
|---|---|---|
| aloha transfer_cube (50롤) | 43% | **52%** (time+flow+ctx4) |
| aloha insertion (50롤) | 28% | **32%** (time+flow+ctx4) |
| LIBERO-Spatial (20롤×10태스크) | 36.5% | **80.0%** (flow+손목캠+d1536, 시드3 평균) |
| LIBERO-Object (10롤×10태스크) | 35.0% | **85.7%** |
| LIBERO-Goal (10롤×10태스크) | 18.0% | **80.0%** |

(참고: 보류 중인 DCT 표현을 aloha에 조합하면 100롤 시드3 평균 58/36까지 실측됨 — 보고서 참조)

핵심 판정(절제 근거는 보고서): ① flow 헤드는 LIBERO에 결정적(+29pp), aloha에도 ctx4와 함께 유효(+9pp)
② flow 출발점은 past(A2A식)가 noise/vision 대비 일관 승 ③ 손목캠 토큰 +26pp(LIBERO)
④ 개루프 MAE는 폐루프 성공률을 예측하지 못함 — 모델 선택은 폐루프로.

## 다음 개선 레버 (우선순위)

1. **aloha 손목캠** — XML에 left/right_wrist 카메라 존재, 렌더 연결+재수집만 필요 (LIBERO +26pp의 aloha판)
2. **정책 시각입력 패치토큰화** — pooled 768의 공간 정밀도 한계 (잔여 병목 정면 공략)
3. **폐루프 기반 모델 선택** — val 손실이 시드 우열을 예측 못함 (Goal 시드 64~90% 실측)
4. DAgger 자동 루프 / 데이터 증량 (aloha)
5. phase1 g/h에 손목캠 조건 추가
6. (보류) DCT 청크 표현 연구 재개 — 캠페인에서 aloha 폐루프 +8pp 실측, 코드 유지 상태

연구 이력: `docs/upgrade_report.md`(이번 캠페인) · `~/clip_openx_ws/experiments/`(delta-AE 탐색 152런)
