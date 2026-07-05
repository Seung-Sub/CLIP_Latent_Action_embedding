# LIBERO 트랙 — 학습·평가 파이프라인

aloha 트랙(→ README.md)과 병렬 구조. 공용층(src/core·models·training)은 동일하고,
환경 의존 3점 세트만 다르다: conda env(`clip_libero`) · `src/data/libero.py`(로더) · `src/eval_libero/`(평가).

| | aloha 트랙 | **LIBERO 트랙** |
|---|---|---|
| 로봇/액션 | ALOHA 양팔, 관절 14D | Franka Panda, OSC 델타 7D (Δpos+Δrot+그리퍼) |
| 제어 | 50Hz | 20Hz |
| 데이터 | 직접 수집 (스크립트 정책) | **공식 인간 텔레옵 데모** (task당 50개) — 수집 불필요 |
| 카메라 | angle | agentview + **eye_in_hand(손목캠)** |
| 태스크 | 2개 (언어 불필요) | suite당 10개 + **언어 지시문** |
| 권장 정책 | DCT + flow(past) + ctx4 | **flow(past) + 손목캠 + d1536** (DCT 무기여) |
| conda | `clip` | **`clip_libero`** (robosuite/mujoco 충돌 격리) |

## 성능 (2026-07-04 캠페인, 시드 3 평균 — 절제 근거: docs/upgrade_report.md)

| suite | 베이스라인 (mlp 6M) | **flow+손목캠+d1536 (124M)** |
|---|---|---|
| libero_spatial (20롤/태스크) | 36.5% | **80.0%** {83, 73, 84} |
| libero_object (10롤/태스크) | 35.0% | **85.7%** {93, 76, 88} |
| libero_goal (10롤/태스크) | 18.0% | **80.0%** {64, 90, 86} |

## 0. 환경 설치

```bash
conda env create -f environment_libero.yml    # env "clip_libero" (pypi 'libero' = HF 재배포판)
conda activate clip_libero
# CLIP 가중치는 aloha 트랙과 공유 (configs/config.yaml의 clip.model_dir)
```

검증된 조합: libero 0.1.1(pip) + robosuite 1.4.0 + **mujoco 3.3.2** (2.3.x는 로봇 텍스처 붕괴, 3.10+는 API 크래시).
벤치마크 자산은 첫 실행 시 HF Hub 자동 다운로드(`~/.cache/libero/assets`) — 레포 클론 불필요.

## 1. 데모 데이터 다운로드

```bash
python -c "from libero.libero.utils.download_utils import download_from_huggingface; \
           download_from_huggingface('libero_spatial', 'data/libero', check_overwrite=False)"
# object/goal도 동일 ('libero_object', 'libero_goal') — suite당 6~7GB, task당 hdf5 1개×10
```

## 2~5. 학습·평가

```bash
conda activate clip_libero && cd ~/clip_ws

# 2. Phase 1 — 액션청크↔Δz 결합 AE (LIBERO는 time 표현 유지 — DCT 무기여 실측)
python src/training/train_phase1.py --config configs/phase1_libero.yaml

# 3. Phase 2 — 기본 = SOTA (flow + 손목캠 + d1536, configs/phase2_libero.yaml)
python src/training/train_phase2.py --config configs/phase2_libero.yaml
#    베이스라인(mlp): --config configs/phase2_libero_mlp.yaml

# 4. GT 데모 평가 (7D 플롯) → outputs/eval/rollout_dataset_libero_*.png
python src/eval_libero/rollout_dataset.py

# 5. 폐루프 suite 평가 (체크포인트가 구조를 자동 복원 — mlp/flow/손목캠 동일 명령)
MUJOCO_GL=egl python src/eval_libero/rollout_sim.py --suite libero_spatial --episodes 10
#   옵션: --task-id 0 (단일 태스크) / --save-video 2 / --exec-horizon 8(기본·최적)
```

**승자 체크포인트가 기본 경로에 승격**(`checkpoints/phase2_libero_fm.pt`, Spatial 20롤 84%)되어
있어 위 5번 명령 그대로 SOTA 평가가 됩니다. Object/Goal 승자 ckpt로 평가하려면:

```bash
MUJOCO_GL=egl python src/eval_libero/rollout_sim.py \
    --config outputs/campaign/cfg/o_win.yaml --suite libero_object --episodes 10   # 93%
MUJOCO_GL=egl python src/eval_libero/rollout_sim.py \
    --config outputs/campaign/cfg/g_win_s1.yaml --suite libero_goal --episodes 10  # 90%
```

다른 suite로 학습하려면 data 경로만 교체: `--set data.root=~/clip_ws/data/libero/libero_object
--set data.cache_dir=~/clip_ws/outputs/cache/libero_obj_emb` (+ 평가 시 `--suite libero_object`).

## 구현 노트 (재현 시 주의)

1. 멀티태스크 언어 조건화 필수 → `module.lang_token: true` (CLIP 텍스트 임베딩 4번째 토큰,
   교란실험으로 기여 검증됨: 정상 0.384 / 교란 0.517 MAE)
2. **손목캠 토큰** = `data.wrist_camera: eye_in_hand_rgb` + `module.wrist_token: true` —
   현재 시점 프레임 1장이 5번째 토큰. env 렌더 키는 `robot0_eye_in_hand_image`,
   데모와 동일 방향 실측(cos 0.992, flip 불필요). 단독 +26pp
3. mujoco 버전: **3.3.2 필수** (2.3.x=로봇 텍스처 붕괴 / 3.10+=API 크래시)
4. 데모와 env 렌더는 동일 방향 (공식 코드의 [::-1]은 영상표시용)
5. 첫 임포트 시 대화형 프롬프트 → 스크립트에선 `printf "N\n" |` 파이프
6. flow 학습은 val 손실이 시드 우열을 예측 못함(Goal 시드 64~90%) — 최종 선택은 폐루프 평가로

설계 문서: `docs/libero_plan.md` · 캠페인 보고서: `docs/upgrade_report.md`
