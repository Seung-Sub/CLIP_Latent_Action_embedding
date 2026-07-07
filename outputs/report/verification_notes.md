# 확인 필요 항목 공식 검증 노트 (계획서 §CHANGELOG "확인 필요" 해소)

검증일: 2026-07-06 / 방법: 공식 GitHub·HuggingFace·논문 직접 확인 (리서치 에이전트)
모든 항목 공식 소스 인용. 아래 사실들은 D1·D2 구현과 평가 프로토콜에 그대로 반영함.

## 1. OpenVLA 수정판 LIBERO 데이터

- **배포**: HF `openvla/modified_libero_rlds` (MIT), **RLDS(TFRecord) 형식만 공식 배포**, 총 10.2GB.
  - libero_spatial_no_noops 1.91GB / object 2.82GB / goal 1.84GB / 10 3.66GB. LIBERO-90 없음.
- **해상도**: 재렌더 **256×256** (원본 128×128), 카메라 2종 저장(agentview_rgb, eye_in_hand_rgb).
  RLDS 변환 시에만 이미지 **180° 회전** 적용(OpenVLA 플랫폼 특이사항 — 재생성 HDF5에는 없음).
- **no-op 필터 규칙** (regenerate_libero_dataset.py 원문):
  `is_noop(a, prev) = ‖a[:-1]‖₂ < 1e-4 AND a[-1] == prev[-1]`
  즉 그리퍼 제외 6D의 L2 norm < **1e-4** 이고 그리퍼 명령이 직전과 동일할 때만 제거
  (정지 상태에서 그리퍼만 여닫는 스텝은 보존). 에피소드 첫 스텝은 norm 기준만.
- **실패 데모 제거**: 데모 액션을 env에 재생(replay) 후 `done`(성공)인 에피소드만 저장.
- **원본 HDF5 형식의 수정판은 공식 미배포** — 공식 재생성 스크립트(regenerate_libero_dataset.py)로
  로컬 재생성이 공식 경로. → **우리 D2는 이 스크립트와 동일 로직의 자체 재생성으로 확보**
  (RLDS→HDF5 역변환 대신; tensorflow 의존성 회피, 180° 회전은 우리 플랫폼 비해당).

## 2. OpenVLA 공식 평가 관행 (run_libero_eval.py)

- **suite별 max_steps**: spatial **220** / object 280 / goal 300 / libero_10 520 / libero_90 400
  (각각 최장 학습 데모 길이 기반). + 시작 시 **num_steps_wait=10** 더미 스텝
  (물체 낙하 안정화 대기, dummy action `[0,0,0,0,0,0,-1]`) — max_steps에 불포함.
- **태스크당 50 트라이얼**, 공표치는 3시드 × 500롤아웃 평균 (Spatial 84.7±0.9%).
- 평가 렌더 256², **agentview 단일 카메라**만 정책 입력(→224 리사이즈), 손목캠 미사용.
- 액션: 7D delta EEF, 그리퍼 [0,1]→[-1,1] 이진화.
- 제어 주파수(Hz)는 공식 코드에 명시 없음 (UNVERIFIED — env 기본 제어율 사용).

**우리 프로토콜에의 반영**:
- 현행 우리 max_steps=300 (고정) — 공표 비교 시 "OpenVLA 관행은 spatial 220+wait10"임을 명기.
  50롤아웃 승급에서 wait 스텝 부재 여부 점검(물체 낙하 중 시작 → 실패 원인 가능성).
- data_variant 필드: `raw`(원본 128² HDF5) / `openvla_modified`(no-op·실패 필터+256² 재렌더 규칙 재현).

## 3. V-JEPA2 체크포인트 (Phase 2 앵커 후보 A군)

- 공식 HF: `facebook/vjepa2-vitl-fpc64-256`(0.3B) / `vjepa2-vith-fpc64-256`(0.6~0.7B) /
  `vjepa2-vitg-fpc64-256`(1B) / `vjepa2-vitg-fpc64-384`(1B). crop 256(vitg-384만 384).
- **단일 이미지 모드 공식 지원**: 프레임 반복(`pixel_values.repeat`)으로 이미지 백본 사용 가능,
  `get_vision_features(skip_predictor=True)`. 별도 이미지 전용 가중치 없음.
- V-JEPA 2-AC(액션 조건)는 GitHub 배포. V-JEPA 2.1 계열(ViT-B~G, 384)도 추가됨.
- 논문: arXiv 2506.09985.

## 남은 미검증

- OpenVLA LIBERO 평가의 제어 주파수 Hz 수치 (공식 코드에 상수 없음).

## 4. (S1.v2 §8 추가) openvla_modified 재현 시 재생 실패 62 데모

- 집계: data/libero_openvla_mod/libero_spatial/regen_stats.json — 태스크별 실패 수
  1/4/6/6/12/8/1/13/4/2 (task 0~9 순), 계 62/500 (12.4%).
- **per-demo ID 목록은 미확보**: 재생성 스크립트가 카운트만 기록. ID 확보에는 재생
  재실행 필요 (GPU·EGL, 체인 경합으로 보류 — 비차단, 재실행 시 스크립트에 ID 로깅
  1줄 추가하면 됨: d2_rerender.py openvla_replay 분기).
- 함의: 공식 인간 데모의 12.4%가 결정론 재생으로 성공 재현 불가 (env 비결정성 또는
  데모 품질). OpenVLA 공표 수치의 학습 데이터도 동일 필터 통과분만임.

## 5. (S1.v2 §6) 캠페인 수치 소급 등재 한계

- docs/upgrade_report.md는 .gitignore(/docs/) 대상 — 이 머신에 파일 부재.
- 캠페인 수치(80.0/85.7/80.0, mlp 36.5/35/18)는 README·config 주석에서만 확인.
- 프로토콜 필드(n/task, wait, max_steps, init_states) 불명 → §8 스텁 등재 보류,
  병렬 연구자에게 원본 공유 요청 필요.

## 6. (H2-fair §1–2) 앵커 전처리 정합 검증 (2026-07-08)

- **SigLIP2**: `padding="max_length"` 적용 확인 + 실측 토큰 길이 64 = text max_position ✓
  (HF SigLIP 계열 대표 함정 비해당). 이미지 384², mean/std (0.5,0.5,0.5) — 체크포인트
  카드 일치 ✓. **224 변형 존재 확인** (google/siglip2-so400m-patch14-224) — 해상도 동등
  조건 1런 추가 가능 (대기).
- **DINOv2-large**: registers 0 (비레지스터 변형), pooled = pooler_output(CLS 기반) ✓.
  processor mean/std ImageNet 표준 — 카드 일치 ✓. 불일치 발견 없음 → 재산출 불요.
- **새니티 게이트**: 태스크 문장(공간 관계 구분) 검색은 CLIP 0% / SigLIP2 10% — **양 앵커
  모두 우연 수준이라 게이트 판별력 없음** (공간 추론 = zero-shot 공통 약점, 세팅 문제 아님).
  조악 캡션 대체 검증(로봇 장면 vs 개·거리·인물): 양쪽 100% — 세팅 건전 ✓.
- **vocab 중립성 (§5) — 편향 실측**: v2 문장 최근접=동일카테고리 비율 CLIP 36.7% vs
  SigLIP2 18.1% (2배 격차) → **모션 어휘가 CLIP 편향** 확인. dual-score v3 생성·재판정
  체인 가동 (기존 "목적함수 정합성" 판정은 v3 결과까지 보류로 격하).
