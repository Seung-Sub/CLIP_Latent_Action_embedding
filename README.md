# clip_ws — CLIP 잠재공간 Visual-Policy (ALOHA · LIBERO)

frozen CLIP ViT-L/14 잠재공간에서 액션청크를 결합(Phase 1)하고, 그 위에서 미래
액션청크를 추론하는 flow matching 정책(Phase 2)을 학습·평가한다.

```
Phase 1:  g(A청크, z_t) ≈ Δz = z_{t+16} − z_t   /   h(Δz, z_t) ≈ A청크
Phase 2:  토큰 [z_{t−16}, z_t, g(A_past), (언어), (손목캠)] → flow matching → ζ̂ → h(ζ̂, z_t) = Â
폐루프:   16스텝 예측 → 앞 8스텝 실행 → 재예측 (receding horizon)
```

실험 트랙 2개 — 각각 독립된 conda 환경을 사용한다:

| | A. ALOHA | B. LIBERO |
|---|---|---|
| 로봇/액션 | 양팔 ViperX, 관절 14D, 50Hz | Franka Panda, OSC 델타 7D, 20Hz |
| 데이터 | 스크립트 정책으로 직접 수집 | 공식 인간 데모 다운로드 |
| 정책 입력 | 시각 3토큰 | +언어, +손목캠 토큰 |
| conda env | `clip` | `clip_libero` |

## 0. 요구사항

- Linux + NVIDIA GPU (VRAM 12GB 이상 권장; RTX 5070 Ti급 Blackwell은 포함된 cu128 torch 필수)
- conda(Anaconda/Miniconda), 디스크 여유 약 30GB (LIBERO 3개 suite 시 +20GB)

## 1. 공통 설치 (최초 1회)

```bash
git clone <repo-url> ~/clip_ws && cd ~/clip_ws

# CLIP 가중치 (~1.6GB) — 경로는 configs/config.yaml의 clip.model_dir와 일치시킬 것
pip install -U "huggingface_hub[cli]"
hf download openai/clip-vit-large-patch14 --local-dir ~/clip_ws/models/clip-vit-large-patch14

# wandb 로깅을 쓰지 않으면 모든 학습 명령에 --set wandb.enabled=false 를 붙인다
```

---

## 2. 실험 A — ALOHA (transfer_cube / insertion)

```bash
conda env create -f environment.yml        # env "clip" (최초 1회)
conda activate clip && cd ~/clip_ws
```

### A-1. 데이터 수집 (작업당 200 에피소드, 5워커 병렬, ~15분/작업)

```bash
cd aloha
MUJOCO_GL=egl python record_sim_episodes.py --task_name sim_transfer_cube_record \
    --num_workers 5 --num_episodes 200 --dataset_dir ~/clip_ws/data/act_sim/sim_transfer_cube
MUJOCO_GL=egl python record_sim_episodes.py --task_name sim_insertion_record \
    --num_workers 5 --num_episodes 200 --dataset_dir ~/clip_ws/data/act_sim/sim_insertion
cd ..
# 확인(선택): cd aloha && python visualize_episodes.py --dataset_dir ~/clip_ws/data/act_sim/sim_transfer_cube --episode_idx 0
```

### A-2. Phase 1 학습 (첫 실행 시 CLIP 인코딩 ~15분 + 학습 ~5분)

```bash
python src/training/train_phase1.py
# → checkpoints/phase1_delta_ae.pt   (기대: 디코더 R² ≈ 0.98)
```

### A-3. Phase 2 학습 (~10분)

```bash
python src/training/train_phase2.py
# → checkpoints/phase2_aloha_fm.pt   (flow matching 정책, 72M)
```

### A-4. GT 데이터셋 평가 — 예측 궤적이 GT를 따라가는지 (14관절 플롯)

```bash
python src/eval_aloha/rollout_dataset.py --task sim_transfer_cube
# → outputs/eval/rollout_dataset_*.png   (기대: 관절 MAE < 1°)
```

### A-5. 폐루프 성공률

```bash
MUJOCO_GL=egl python src/eval_aloha/rollout_sim.py --task sim_transfer_cube --episodes 50
MUJOCO_GL=egl python src/eval_aloha/rollout_sim.py --task sim_insertion --episodes 50
# 기대(50롤): transfer ≈ 50% / insertion ≈ 30%   (베이스라인 mlp: 43 / 28)
# 옵션: --save-video 3 (앞 3편 mp4) / --onscreen (실시간 창)
```

---

## 3. 실험 B — LIBERO (spatial / object / goal suite)

```bash
conda env create -f environment_libero.yml   # env "clip_libero" (최초 1회)
conda activate clip_libero && cd ~/clip_ws
# 첫 libero 임포트 시 설정 프롬프트가 뜨면 N 입력 (스크립트에선 printf "N\n" | 파이프)
# 벤치마크 자산은 첫 실행 시 ~/.cache/libero 로 자동 다운로드된다
```

### B-1. 데모 데이터 다운로드 (suite당 6~7GB, task당 hdf5 1개 × 10)

```bash
python -c "from libero.libero.utils.download_utils import download_from_huggingface; \
           download_from_huggingface('libero_spatial', 'data/libero', check_overwrite=False)"
# object/goal suite도 동일: 'libero_object', 'libero_goal'
```

### B-2. Phase 1 학습 (첫 실행 시 CLIP 인코딩 ~30분 + 학습 ~5분)

```bash
python src/training/train_phase1.py --config configs/phase1_libero.yaml
# → checkpoints/phase1_libero.pt
```

### B-3. Phase 2 학습 (~15분 — 손목캠 인코딩 캐시 포함)

```bash
python src/training/train_phase2.py --config configs/phase2_libero.yaml
# → checkpoints/phase2_libero_fm.pt   (flow + 언어 + 손목캠 토큰, 124M)
```

### B-4. GT 데모 평가 (7D 액션 플롯)

```bash
python src/eval_libero/rollout_dataset.py
# → outputs/eval/rollout_dataset_libero_*.png
```

### B-5. 폐루프 suite 평가

```bash
MUJOCO_GL=egl python src/eval_libero/rollout_sim.py --suite libero_spatial --episodes 10
# 기대(10롤×10태스크 평균): spatial ≈ 80%   (베이스라인 mlp: ≈ 37%)
# 옵션: --task-id 0 (단일 태스크) / --save-video 2
```

### B-6. 잠재공간 맵핑 시각화 (선택, 대화형 창)

```bash
python src/eval_libero/latent_mapping.py
# phase1 잠재공간에 3인칭 전/후·Δz 화살표·g(액션청크) 화살표·그리퍼 델타·언어 cmd를
# PCA 2D/3D로 표시. 우측에서 태스크/에피소드/시작 시점 선택, [전체 구성]·[3D]·[확대] 토글
```

다른 suite(object/goal)로 실험하려면 **config 사본에서 경로만 교체** 후 같은 절차 (예: object):

```bash
sed 's/libero_spatial/libero_object/; s/libero_emb/libero_obj_emb/; s/phase1_libero/phase1_libero_obj/' \
    configs/phase1_libero.yaml > configs/phase1_libero_obj.yaml
sed 's/libero_spatial/libero_object/; s/libero_emb/libero_obj_emb/; s/phase1_libero/phase1_libero_obj/; s/phase2_libero_fm/phase2_libero_obj/' \
    configs/phase2_libero.yaml > configs/phase2_libero_obj.yaml
# B-2/B-3을 --config configs/phase1_libero_obj.yaml / phase2_libero_obj.yaml 로 실행하고
# B-5는 --config configs/phase2_libero_obj.yaml --suite libero_object
```

---

## 4. 모델 변형과 실험 옵션

- **기본 = flow matching 정책** (권장). **베이스라인(MLP 회귀)** 비교는 config만 교체:
  ```bash
  python src/training/train_phase2.py --config configs/phase2_mlp.yaml          # ALOHA
  python src/training/train_phase2.py --config configs/phase2_libero_mlp.yaml   # LIBERO
  ```
- 평가 스크립트는 **체크포인트에 저장된 config로 모델 구조를 자동 복원**한다 —
  같은 평가 명령으로 어떤 변형이든 평가된다 (config의 `train.checkpoint`가 평가 대상을 지정).
- 모든 학습 스크립트는 오버라이드/분리 저장을 지원한다:
  ```bash
  python src/training/train_phase2.py --set module.d_model=512 --tag my_run
  # → checkpoints/grid/my_run.pt, outputs/grid/my_run.json (기본 체크포인트를 건드리지 않음)
  ```
- 주요 `module` 키: `name`(mlp|flow) · `d_model` · `layers` · `ctx_layers`(flow 문맥 인코더) ·
  `flow_steps`(Euler 스텝) · `lang_token`(LIBERO 언어) · `wrist_token`+`data.wrist_camera`(LIBERO 손목캠)

## 5. 디렉터리

| 경로 | 내용 |
|---|---|
| `configs/` | phase1/phase2 설정 (ALOHA: `phase1.yaml`·`phase2.yaml` / LIBERO: `*_libero.yaml` / 베이스라인: `*_mlp.yaml`) |
| `src/core` `src/data` `src/models` `src/training` | CLIP 래퍼 · 로더(임베딩 캐시) · DeltaAE+정책 · 트레이너 (두 트랙 공용) |
| `src/eval_aloha` `src/eval_libero` | GT 평가(`rollout_dataset.py`) / 폐루프 평가(`rollout_sim.py`) |
| `aloha/` | ALOHA MuJoCo 시뮬 + 수집·시각화 스크립트 |
| `data/` `checkpoints/` `outputs/` | 데이터 / 학습 결과 / 캐시·평가 산출물 (git 제외) |

## 6. 문제 해결

- **mujoco 버전 (LIBERO)**: 반드시 3.3.2 (environment_libero.yml에 고정됨 — 2.3.x는 로봇 렌더
  붕괴, 3.10+는 크래시). 임의 업그레이드 금지
- **렌더 오류**: 헤드리스/원격에서는 모든 시뮬 명령에 `MUJOCO_GL=egl` 필수
- **데이터를 다시 수집/다운로드한 경우**: 해당 임베딩 캐시를 비울 것
  (`rm -rf outputs/cache/act_sim_emb` 또는 `outputs/cache/libero_emb`)
- **처음부터 완전 재현**: `rm -rf data checkpoints outputs` 후 위 절차를 처음부터
- GPU 2장 병렬 실험 시 `CUDA_VISIBLE_DEVICES=0|1` 로 트랙별 분리
