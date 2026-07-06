# 실험 설계서 v1.1 — "사전학습 임베딩 변위(Δz)를 액션 코드로 쓰는 정책" 검증 로드맵

작성: 2026-07-03 (v1.1, LIBERO 트랙 반영 개정) / 실행: Claude Code
대상 코드베이스: `clip_ws` — 2트랙 구조 (aloha / **libero**, 공용층 src/core·models·training)

---

## CHANGELOG (v1.0 → v1.1) — 코드베이스 최신화 반영

**이미 완료된 항목 (계획에서 '완료' 처리)**
- [완료] 언어 조건화 L1: CLIP 텍스트 임베딩 4번째 토큰 (`module.lang_token: true`).
  정성 관찰: 지시문에 따라 분기, 오지시 시 오동작 → **인과성은 있어 보이나 아직 정량화 안 됨** (→ §5).
- [완료] 고정 평가 스위트의 LIBERO판: `suite.get_task_init_states()` 공식 초기상태 사용,
  `env.check_success()` 판정 — v1.0 §0.2의 의도가 이미 표준 방식으로 충족됨.
- [완료] 데이터 소스 추상화 1단계: `data/act_sim.py` + `data/libero.py` 동일 인터페이스.

**전제 변경 (계획 수정 사항)**
- 주 벤치마크: ALOHA → **LIBERO-Spatial** (언어 내장, 태스크 10개 × 인간 텔레옵 데모 50개,
  공표 베이스라인 존재). ALOHA는 기제 연구 트랙으로 강등(§트랙 구분 참조).
- 신규 결정적 신호: **Phase1 R²가 ALOHA 0.981 → LIBERO 0.68로 급락** (retrieval top-1 45%).
  → 신규 Phase 1.5 "R² 붕괴 진단" 추가 (본 개정의 최우선 항목).
- 폐루프 38.0% (spatial 평균, 태스크당 20롤아웃)의 좌표: 공표 기준
  Diffusion Policy(from scratch) 78.3 / Octo(FT) 78.9 / OpenVLA(FT) 84.7 /
  OpenVLA-OFT 97.6 / π0 96.8–98.0 / LAPA 73.8 (%). → 현재 DP-scratch의 절반 수준.
  성능 바닥 작업(§6)의 우선순위 상향 + **flow/CVAE 디코더 보류 해제**(인간 데모 = 다봉 데이터,
  보류 조건이 이미 충족됨).
- Phase 0.1(앵커 추상화)은 **여전히 미완** (`policy.py LATENT=768` 하드코딩, CLIP 전용 래퍼).
- 평가 표본: 태스크당 20 → **50 롤아웃** (공표 프로토콜과 동일, suite 500회).
- DAgger는 LIBERO에 전문가가 없으므로 **ALOHA 트랙 전용**으로 이동.
- 언어 분기 시나리오: ALOHA 커스텀 제작 불필요 → **LIBERO-Goal suite** (동일 장면·상이 목표)로 대체.

---

## 트랙 구분 (v1.1 신설)

| | **LIBERO 트랙 (주)** | ALOHA 트랙 (기제 연구) |
|---|---|---|
| 목적 | 언어·벤치마크 비교·논문 메인 결과 | 통제 실험 (스크립트=결정론), DAgger, force 바인딩 |
| 데이터 | 공식 인간 데모 50/task (다봉) | 스크립트 전문가 (단봉, 질의 가능) |
| 특성 | R² 0.68 레짐 | R² 0.98 레짐 |
| 역할 | H1–H3 검증 | H4(force) + R² 레짐 대조 분석 |

※ "스크립트(0.98) vs 인간(0.68) 레짐에서 Δz 액션 코딩이 어떻게 달라지는가" 자체가
논문의 분석 축이 된다. 두 트랙 수치를 항상 병기할 것.

---

## 0. 연구 테제와 사전 등록 가설 (v1.0과 동일, H3 상태만 갱신)

- H1 (역할 분리): 관측 인코더와 액션 코드 앵커의 분리가 폐루프 성능을 유의하게 올린다.
- H2 (앵커 선택): 언어 정렬 앵커의 Δz는 무언어 앵커(DINOv2) 대비 액션 코딩 동급.
- H3 (언어 축): **[부분 달성 — 정량화 필요]** L1(원시 주입)로 정성적 인과 확인됨.
  갭 보정(L2–L4)의 추가 이득과 분기 준수율 ≥90%를 정량 검증.
- H4 (멀티모달 바인딩): force 바인딩 ≥ 토큰추가 (저데이터) or 창발 검색 성립. [ALOHA 트랙]

---

## 1. Phase 0 — 인프라 정비 (예상 1주) [일부 완료, 잔여 명시]

### 0.1 앵커 추상화 [미완 — 그대로 필수]
- `src/core/anchor.py`: `encode_images / encode_texts / dim / patch_dim / has_text / id`.
  구현체 ClipAnchor · Siglip2Anchor · Dinov2Anchor · ImageBindAnchor.
- `policy.py LATENT=768` 및 DeltaAE `latent_dim` → anchor.dim 일반화.
- 옵션: `projection {joint, pre}`, `normalize {true, false}`.
- 캐시 키: `{anchor_id}/{projection}/{normalize}/{camera}` (기존 libero_emb 캐시와 분리).

### 0.2 평가 프로토콜 승급 [반완료 → 표본 수·리포트만 보강]
- 태스크당 **50 롤아웃**(공식 init_states 순서 고정 = paired), suite 500회.
- 리포트: 태스크별 SR + Wilson 95% CI + suite 평균 CI(±~4%p) + 실패 모드 태깅
  (도달 실패 / 파지 실패 / 오브젝트 오인 / 지시 오해).
- max_steps: 현행 300 고정 — 공표 비교 시 suite별 관행 horizon과의 차이를 리포트에 명기
  (구현 시 OpenVLA 평가 코드의 suite별 max steps 확인 후 동일화 권장).

### 0.3 베이스라인 [교체]
- 외부(공표, 동일 50데모 레짐): DP-scratch 78.3 / Octo 78.9 / OpenVLA 84.7 /
  OpenVLA-OFT 97.6 / π0 96.8–98.0 / LAPA 73.8 (LIBERO-Spatial, %).
  ※ 주의: OpenVLA 계열 수치는 **수정판 데이터**(near-zero 액션·실패 데모 필터링, 재렌더) 기준.
  우리가 원본/수정판 어느 쪽을 쓰는지 모든 비교표에 명기할 것.
- 내부: k-NN(VINN식) 바닥선을 LIBERO에 이식 (z_t 최근접 데모 청크 재생).
- ALOHA 트랙: ACT@동일데이터 베이스라인 유지 (v1.0 §0.3 그대로).

### 0.4 데이터·로깅 보강
- LIBERO: 물체 GT 상태는 env/obs에서 추출 가능 → P1 프로브 타깃 로깅 스크립트.
- **데이터 판본 결정**: 원본 HDF5(저해상도, no-op 포함 알려짐) vs OpenVLA 수정판
  (no-op·실패 필터, 고해상도 재렌더). §1.5 진단 D1·D2와 직결 — 두 판본 모두 확보 권장.

---

## 2. Phase 1 — 사전 예측 프로브 5종 (v1.0 유지, LIBERO 우선 적용)

P1 pose 선형 프로브 / P2 Δz SNR / P3 zero-shot text↔Δz / P4 그리드 상관 / P5 k-NN.
세부는 v1.0 §3과 동일하되 적용 순서를 LIBERO → ALOHA로. P2는 §1.5의 D1과 결과 공유.

---

## 2.5 Phase 1.5 — R² 붕괴 진단 (v1.1 신설, 최우선) (예상 0.5–1주)

> 질문: 왜 같은 구조가 ALOHA 스크립트에선 R² 0.981인데 LIBERO 인간 데모에선 0.68인가?
> 아래 가설을 하나씩 절제. 각 실험은 Phase1 재학습(수 분)로 충분.

- **D1 무동작(no-op) 구간 오염 [1순위 가설]**: 인간 데모는 정지·머뭇거림 구간을 포함하고,
  해당 청크의 Δz ≈ 렌더 노이즈 → align/recon 타깃이 무의미해져 R²를 끌어내림.
  실험: near-zero 액션 필터링(OpenVLA 수정판 방식 재현: ‖a‖<ε 스텝 제거) 전후 R²·retrieval 비교.
  ※ 근거: OpenVLA 계열 LIBERO 결과가 정확히 이 필터링을 적용한 수정판 데이터 기준.
- **D2 관측 해상도**: 원본 HDF5 프레임 해상도 확인(128² 알려짐) — CLIP 입력 224²로 업스케일 시
  물체 디테일 손실. 실험: 수정판(고해상 재렌더) 또는 env 재렌더 256²로 캐시 재생성 후 비교.
- **D3 청크 시간 스케일**: 20Hz×16=0.8s (ALOHA 0.32s의 2.5배). chunk_sec {0.4, 0.8, 1.6} 스윕.
- **D4 인간 데모 다봉성 정량화**: 유사 상태(z_t 근접 이웃)에서의 액션 분산 측정 →
  결정론 디코더의 이론 상한 추정. 결과는 §6의 flow/CVAE 디코더 도입 근거 수치.
- **D5 백본 교차 확인**: D1·D2 최적 조건에서 앵커 4종 R² 재측정 (Phase 2 매트릭스와 결합).

산출물: `outputs/report/phase1_5_diagnosis.json` + "R² 레짐 표" (aloha vs libero × D1–D5).
게이트 G0(신설): D1–D3 조합으로 LIBERO R² ≥ 0.85 회복 시 현행 구조 유지;
0.85 미만 지속 시 Δz 정의 자체 재검토(비정규화/pre-projection/장기 델타)로 확장.

---

## 3. Phase 2 — 앵커 × 관측 인코더 매트릭스 (v1.0 §4 유지 + 프로토콜 조정)

- 백본 후보 A1–A4(+opt A5) 및 조건 C1–C7: v1.0과 동일.
- **주 무대 변경**: 스크리닝을 LIBERO-Spatial에서 수행 —
  Stage A: 7조건 × 10 롤아웃/task (paired, 100회/조건) / Stage B: 상위 3 × 50/task × 3시드.
- ALOHA 대각 4런(C1–C4)은 저비용 정합성 확인용으로 병행.
- 전제: §2.5의 D1·D2 최적 데이터 조건 확정 후 개시 (오염 데이터로 매트릭스 돌리지 말 것).

---

## 4. Phase 3 — 언어 축 정량화 (v1.0 §5 개정: L1 완료 반영)

### 4.1 조건 (L1은 완료 — 대조군으로 편입)
- L0 무언어 / **L1 원시 텍스트 토큰 [구현됨]** / L2 모달 centering / L3 학습형 선형 어댑터 /
  L4 문장차분 E(instruction)−E(neutral).

### 4.2 평가 (LIBERO 네이티브로 재설계)
- **분기 인과 테스트**: LIBERO-Goal suite (동일 장면 · 목표 10종) — 언어가 유일한 분기 신호.
  지표: 지시 준수율(달성한 goal 판정) ≥ 90% (G3).
- **지시 교차 혼동 행렬 [신설]**: 태스크 i 장면에 태스크 j 지시를 입력 → 행동이 어느 태스크를
  따르는지 10×10 행렬. 사용자의 정성 관찰("오지시 → 오동작")을 정량화하는 그림.
  대각 우세율 + off-diagonal에서 지시 추종률을 보고.
- 패러프레이즈 hold-out: task.language의 LLM 재작성 20종(5종 hold-out) 일반화.
- 갭 정량화: 텍스트/이미지 임베딩 중심 거리·cos + L1 vs L2–L4 성능 격차 병기.
- 스트레스(기록용): 공간 관계 지시 스왑 — spatial suite가 정확히 이 축이므로
  "between/next to/on" 교란 문장으로 bag-of-words 예측 검증.

---

## 5. Phase 4 — 성능 바닥 확보 (v1.0 §6 개정)

- 4.1 proprio 토큰: LIBERO obs의 로봇 상태(ee pose·그리퍼) → 5번째 토큰. [즉시]
- 4.2 patch 정식화: Phase 2 승자 구성에서 mean-patch → PMA k토큰.
- 4.3 **flow/CVAE 디코더 [보류 해제 — LIBERO 트랙 필수 후보]**:
  근거 (i) 인간 데모 = 다봉(§2.5 D4로 정량 근거 확보), (ii) ACT의 절제에서 CVAE 목적함수가
  인간 데이터에 결정적(제거 시 35.3%→2%)이었다는 결과, (iii) VITA FLD·CLASP one-to-many 논의.
  최소구현: h를 조건부 flow matching 디코더로 교체(ζ̂, z_t 조건) — VITA FLD 구조 참조.
- 4.4 DAgger: **ALOHA 트랙 전용** (LIBERO는 전문가 부재).
- 4.5 데이터: LIBERO는 50데모/task 고정(공정 비교 조건) — 증량 대신 위 레버로 승부.
  ALOHA는 200→1000 유지.
- 목표선(내부): LIBERO-Spatial ≥ 70% (DP-scratch 78.3의 사정권) 도달 후 언어·앵커 주장 전개.

---

## 6. Phase 5 — force 바인딩 (v1.0 §7 유지, ALOHA 트랙)

변경 없음. (선택) robosuite/Panda 손목 F/T 노출 여부 조사 — 가능하면 LIBERO 쪽 접촉
태스크에도 이식 검토. 촉각·오디오는 실기 단계 보류 유지.

## 6.5 (조건부) Phase 6 — 무라벨 비디오 사전학습 (v1.0 유지)

LIBERO에선 "액션 라벨 은닉" 시뮬레이션으로 1차 검증 가능 (데모 500개 중 400개 라벨 은닉).

---

## 7. 게이트 요약표 (v1.1)

| 게이트 | 시점 | 기준 | 실패 시 |
|---|---|---|---|
| **G0 (신설)** | Phase 1.5 후 | LIBERO Phase1 R² ≥ 0.85 (D1–D3 조합) | Δz 정의 재검토 (비정규화·pre-proj·장기 델타) |
| G1 | Phase 1 후 | pooled pose MAE ≥ 2× patch | obs 분리 전면 채택 |
| G2 | Phase 2 후 | 언어정렬 앵커 SR ≥ DINOv2 −10%p (동일 obs) | Phase 3 성과로 재판정 |
| G3 | Phase 3 후 | Goal-suite 분기 준수 ≥ 90% (L2–L4 중 1개) | 앵커 LoRA 트랙 |
| G-floor | Phase 4 후 | Spatial ≥ 70% | 원인 분석 후 디코더/관측 재설계 |
| G4 | Phase 5 후 | 바인딩 ≥ 토큰추가 or 창발 검색 | 부정 결과 보고 |
| 전역 중단 | 상시 | DINOv2 전 지표 압도 + G3 실패 | anchor-agnostic 프레임 피벗 |

---

## 8. 결과 리포트 규약 (v1.0 §9 + 필드 추가)

```json
{
  "run_id": "p2_C7_libero_spatial_s0",
  "phase": "phase2", "track": "libero",
  "condition": {"anchor": "siglip2-so400m", "projection": "joint", "normalize": true,
                 "obs": "dinov2-l/mean-patch", "lang": "L1",
                 "data_variant": "openvla_modified|raw", "extras": []},
  "suite": "libero_spatial", "train_seed": 0,
  "eval": {"n_per_task": 50, "per_task_sr": [/*10개*/], "suite_sr": 0.47,
            "wilson_ci": [0.43, 0.51], "max_steps": 300,
            "failure_modes": {"reach": 0, "grasp": 0, "wrong_object": 0, "wrong_goal": 0}},
  "offline": {"phase1_r2": 0.86, "retr_top1": 61.2, "latcos": 0.95,
               "p1_pose_mae_cm": 1.4, "p2_dz_snr": 6.3, "p3_text_dz_auc": 0.52,
               "d4_action_multimodality": 0.0},
  "lang_eval": {"goal_suite_compliance": null, "confusion_diag_rate": null,
                 "paraphrase_holdout_sr": null},
  "train": {"epochs": 50, "early_stop": 31, "wall_min": 8},
  "config_hash": "…", "wandb": "…", "notes": "…"
}
```

- 실패·기각 런 보고 의무, paired 검정 p값 병기, 조건당 실패 영상 2편: v1.0과 동일.
- **데이터 판본(raw/수정판)과 max_steps를 모든 공표 비교에 명기** (신설 의무).

---

## 9. 일정 (v1.1)

| 주차 | 내용 |
|---|---|
| 1 | Phase 0 잔여 (앵커 추상화·50롤아웃 승급·k-NN 이식·데이터 판본 확보) |
| 1–2 | **Phase 1.5 진단 (최우선)** + Phase 1 프로브 |
| 2–4 | Phase 2 매트릭스 (LIBERO 주무대) |
| 4–6 | Phase 3 언어 정량화 + Phase 4 (proprio·patch·flow 디코더) |
| 6–10 | Phase 5 force (ALOHA) + 조건부 Phase 6 |

리스크(갱신): (i) mujoco **3.3.2 고정** (2.3.x 텍스처 붕괴/3.10+ 크래시 — 코드베이스 실측),
(ii) 데이터 판본 혼용으로 인한 비교 불공정 — data_variant 필드로 통제,
(iii) LIBERO 원본 해상도 저하 — D2에서 판정, (iv) ImageBind 라이선스(비상업) — SigLIP2 대체 설계.
