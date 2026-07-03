# LIBERO 트랙 — 학습·평가 파이프라인

aloha 트랙(→ README.md)과 병렬 구조. 공용층(src/core·models·training)은 동일하고,
환경 의존 3점 세트만 다르다: `libero/`(시뮬) · `src/data/libero.py`(로더) · `src/eval_libero/`(평가).

| | aloha 트랙 | **LIBERO 트랙** |
|---|---|---|
| 로봇/액션 | ALOHA 양팔, 관절 14D | Franka Panda, OSC 델타 7D (Δpos+Δrot+그리퍼) |
| 제어 | 50Hz | 20Hz |
| 데이터 | 직접 수집 (스크립트 정책) | **공식 인간 텔레옵 데모** (task당 50개) — 수집 불필요 |
| 카메라 | angle | agentview (3인칭) |
| 태스크 | 2개 (언어 불필요) | suite당 10개 + **언어 지시문** |
| conda | `clip` | **`clip_libero`** (robosuite/mujoco 충돌 격리) |

## 0. 환경 설치

```bash
conda env create -f environment_libero.yml    # env "clip_libero" (pypi 'libero' = HF 재배포판)
conda activate clip_libero
# CLIP 가중치는 aloha 트랙과 공유 (configs/config.yaml의 clip.model_dir)
```

검증된 조합: libero 0.1.1(pip) + robosuite 1.4.0 + **mujoco 3.3.2** (2.3.x는 로봇 텍스처 붕괴, 3.10+는 API 크래시).
벤치마크 자산(bddl/init_states는 패키지 내장, 물체 메시는 첫 실행 시 HF Hub 자동 다운로드
→ `~/.cache/libero/assets`) — 별도 레포 클론 불필요.

## 1. 데모 데이터 다운로드 (LIBERO-Spatial부터)

```bash
# (구현 예정) python libero_download.py --suite libero_spatial --out data/libero/
# → data/libero/libero_spatial/<task>_demo.hdf5 × 10
```

## 2~5. 학습·평가 (aloha와 동일한 명령 체계)

```bash
# 2. Phase 1 — 액션청크↔Δz 결합 AE
python src/training/train_phase1.py --config configs/phase1_libero.yaml

# 3. Phase 2 — 잠재 정책 f
python src/training/train_phase2.py --config configs/phase2_libero.yaml

# 4. GT 데이터셋 평가 (7D 플롯: xyz / 회전 / 그리퍼)
python src/eval_libero/rollout_dataset.py --task <task_name>

# 5. 폐루프 평가 (OffScreenRenderEnv, 20Hz receding-8, env success 판정)
MUJOCO_GL=egl python src/eval_libero/rollout_sim.py --suite libero_spatial --episodes 20
```

## 상태

- [x] conda 환경 `clip_libero` 생성·검증 (env 생성 + agentview 렌더 스모크 통과)
- [x] 데모 다운로드 (HF Hub 경유 — `download_from_huggingface('libero_spatial', ...)`)
- [x] `src/data/libero.py` 로더 (+ 지시문 임베딩 캐시)
- [x] configs 2종 + Phase1/2 학습 (P1: R² 0.68·검색 top-1 45% / P2: MAE 0.08·grip 94%)
- [x] `src/eval_libero/` 2종 (rollout_dataset 7D 그래프 / rollout_sim suite 평가)
- [x] **Spatial 10태스크 성공률: 평균 38.0%** (0~90%, outputs/eval/rollout_libero_spatial.txt)

## 통합 과정에서 잡은 결함 3개 (재현 시 주의)
1. 멀티태스크 언어 조건화 필수 → 정책 4번째 토큰 = CLIP 텍스트 임베딩 (`module.lang_token: true`)
2. mujoco 버전: **3.3.2 필수** (2.3.x=로봇 텍스처 붕괴 → 학습분포 불일치 / 3.10+=API 크래시)
3. 데모와 env 렌더는 동일 방향 (플립 불필요 — 공식 코드의 [::-1]은 영상표시용)

설계 문서: `docs/libero_plan.md`
