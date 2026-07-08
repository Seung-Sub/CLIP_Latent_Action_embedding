# 재실험 지시서 v1 — 앵커 공정화 + DINOv2 역할 교정 + Δz 표현 검증

> **대상**: Claude Code(실행자). 코드베이스·`HANDOFF.md`·이전 실험 전량 숙지 전제.
> **작성**: cowork(이론·문헌 검증 파트너) / 2026-07-08.
> **정본·규율**: 모든 수치의 정본 = `outputs/report/*.json`(§8 스키마). 인용은 `outputs/presentation/NUMBER_CARD.md` 경유. `docs/eval_protocol.md`(fp16 비결정성 → suite 평균 n≥500만 공식, 태스크별은 참고치, <3pp 차이는 반복 규칙, 시드 스크리닝 10eps/task 후 승자만 전량) 준수.
> **감사 근거**: 본 지시서는 `HANDOFF.md` §3·§6·§7과 `src/core/anchor.py` 코드 대조 감사에서 도출됨. 감사 결론 요약은 §0.2 참조.

---

## 0. 이 지시서가 답하는 질문과 배경

### 0.1 사용자 질문
"실험했던 SigLIP2/DINOv2가 잘못 구현·실험된 결과인가? 각 모델에 맞게 파라미터·파이프라인을 수정하고 잠재공간 변위(Δz) 아이디어만 부위별로 적절히 적용해야 하는가? 아니면 이미 잘 수행됐고 CLIP이 우위인가?"

### 0.2 감사 기반 답 (재실험 설계의 근거)
- **SigLIP2**: 구현은 **틀리지 않음**(`get_image_features`=MAP head 풀링, `get_text_features`=Gemma 토크나이저, `siglip2-so400m-patch14-384` 정본 = 올바른 API 사용). **무효 아님.** 그러나 **비교가 공정·결론적이지 않음** — 확정 레시피(**비정규화 Δz**)가 CLIP 기하에서 튜닝됐는데 SigLIP2 native 공간은 정규화 cosine 초구면이므로 불리 가능. 3시드 미완 + CI 겹침이라 현재 상태는 **"CLIP≈SigLIP2(구별 불가)"**. → 재구현이 아니라 **native 기하 맞춤 파라미터 재설정 + 통계력 확보**가 필요.
- **DINOv2**: v1(center-crop+CLS)은 **실제 잘못된 구현**이었고 그 수치 폐기 정당. v2는 앵커 사용법을 고침. 그러나 더 큰 문제는 **역할** — (a) 무텍스트인데 global-semantic 앵커로 사용, (b) obs2에서 **mean-pool 단일 토큰**으로 사용, 둘 다 DINOv2의 강점(공간 패치)을 버림. mean-pool 토큰은 P6 shortcut(0.0987 적신호)·proprio식 causal-confusion을 유발. → **"무익"이 아니라 "잘못된 역할·풀링"**. 올바른 역할(관측 패치 토큰)로 재실험 필요.
- **"CLIP 우위?"**: **앵커 역할에 한해** DINOv2 대비 공정 v2 +7.8pp(벤치마크 조건부)로 방어 가능. 그러나 **SigLIP2 대비는 통계적 무승부**라 "CLIP 최선"은 과대 주장. 정직한 현재 결론: *global VL-semantic 앵커(CLIP/SigLIP2) > 무텍스트 dense 앵커, VL 앵커끼리는 CLIP≈SigLIP2, DINOv2의 자리는 관측(공간) 측면인데 아직 올바르게 테스트 안 됨.*

### 0.3 설계 원칙 (Δz 부위별 적용)
Δz(잠재공간 변위)는 본질적으로 **global-semantic 연산**이다. 따라서:
- **앵커(Δz 타깃 공간)** = 텍스트 정렬 가능한 global-semantic 인코더만 자격 있음 → **CLIP / SigLIP2**.
- **DINOv2의 dense 특징** = **관측(observation) 측면**에만, 그것도 **패치 토큰으로 정책이 attend**하게 (전역 pooling 금지). 이는 H1(역할 분리) 가설과 정확히 일치하고, OpenVLA/Prismatic이 검증한 융합 방식(두 인코더 패치 토큰 채널 concat, 전역 pooling 없음; DINOv2=공간, SigLIP=의미)과도 정합.

---

## Block 0 — 공통 규약 (전 실험 불변 통제)

- **통제 고정값**(오직 실험 변수만 변경): 데이터 판본 = raw 확정(`condition.data_variant` 필드에 명기), flow `source=past`, `d_model=1536`, 손목캠 토큰 on, 청크 0.8s(16스텝)/receding 8, receding·`max_steps`·`wait_steps` 현행 유지.
- **시드**: **순위/우열 주장에는 3시드(s0,s1,s2) 필수.** 스크리닝은 10eps/task, 승자만 50roll × 3seed.
- **비교 방식**: head-to-head는 **paired(동일 init states) + 차분(paired) CI**로 보고. marginal Wilson CI만으로 순위 매기지 말 것. **CI 겹치면 "구별 불가"로 판정하고 "우위" 서술 금지.**
- **프롬프트 규약**: 텍스트 정렬용 모션 문장에 **단일 프롬프트 컨벤션을 CLIP·SigLIP2에 동일(대칭) 적용.** 사전등록: 기본 = raw 문장, 보조 = SigLIP 캡션형 템플릿 1종. 두 앵커에 대칭 적용된 경우에만 언어 축 비교가 유효.
- **오프라인 지표로 선정 금지**: 백본/토큰/레시피 선정은 **폐루프 SR + P7 강건성** 기준. `HANDOFF.md` §6.2(오프라인↔폐루프 해리)에 의거, offline dec R²/retrieval 상승은 선정 근거가 아님.
- **산출물**: 모든 런 §8 JSON, `condition`에 `anchor / projection / normalize / pooled / obs_fusion` 필드 기록. **실패·기각 런도 보고**(paired p값 병기, 조건당 실패 영상 2편).

**재실행 금지(이미 정착 — 컴퓨트 낭비 방지)**: DINOv2-v1(crop+CLS), R² 붕괴 진단 D1–D4, ARM-AE 대조군, sigmoid 스케일 3형태 탐색. 컴퓨트는 아래 E0–E5에만 투입.

---

## Block 1 — 프리플라이트: Δz 자명해 프로브 (E0, 최우선·저비용)

**목적**: g(A, z_t)가 z_t를 무시하고 **타깃 프레임만 인코딩**하는 자명해(trivial solution)인지 검증. cycle R²(0.88) ≫ dec R²(0.68) 간극이 그 신호일 수 있음. 이게 붕괴하면 E1–E5 해석 전부의 전제가 무너지므로 **가장 먼저** 확인. (근거: LAM 이론 문헌의 "LAM이 타깃 프레임만 인코딩하는 자명해로 붕괴" 경고.)

**구현**:
- `train_phase1.py`에 z_t 절제 스위치 추가: g **입력의 z_t**를 `{intact / zeros / shuffle(배치 내 셔플)}` 3모드. h는 intact 유지(디코더는 z_t 사용이 정상).

**실행**: CLIP 앵커 + 확정 레시피, 3모드 × 1시드 phase1 재학습(각 수 분).

**사전등록 판정**:
- **정상**: intact 대비 zeros·shuffle에서 **align cos·dec R²·a2z retrieval가 유의하게 하락**(g가 z_t를 실제 조건으로 사용).
- **자명해 적신호**: 하락이 시드 노이즈 이하 → `outputs/report/e0_trivial_probe.json`에 `flag=true` 기록 + cowork 큐 회부. 이 경우 E2 이후 해석 시 "Δz가 조건부 dynamics가 아니라 정적 프레임 임베딩일 수 있음"을 명시.

**주의**: 이건 정책 성능이 아니라 **표현 타당성 게이트**. 통과해야 이후 실험 해석이 의미를 가짐.

---

## Block 2 — 앵커 공정 재판정: CLIP vs SigLIP2 (E1–E2, 핵심)

### E1. 앵커별 native 레시피 탐색 (레시피 편향 제거)

**목적**: 각 앵커에 CLIP-튜닝 레시피를 강제하지 않고 **자기 native 기하에서 최적 레시피**를 찾는다.

**구현**: 기존 config 키만으로 가능 —
- `anchor.normalize {true, false}` × 정렬 `{hybrid λ_c∈{0.1,0.3}, dz(순수)}`.
- **CLIP**: {norm∈[T,F]} × {λ_c∈[0.1,0.3], dz}
- **SigLIP2**: {norm∈[T,F]} × {λ_c∈[0.1,0.3], dz} — **주 가설: norm=true가 native(cosine 구면)**.

**실행**: 각 조합 phase1 + phase2 스크리닝(10eps/task, paired). **앵커별 폐루프 기준 최적 레시피 1개** 선정.

**주의**:
- 정규화가 Δz 스케일을 바꾸므로 phase2 `x0_std`(flow source 스케일) **재적합** 확인.
- 앵커별 캐시키 분리(`{id}/{projection}/{norm|raw}`) 이미 존재 — 구/신 캐시 혼합 금지.
- 선정은 **폐루프 기준**(offline 금지).

**사전등록 판정**: 앵커별 승자 레시피를 `outputs/report/e1_recipe_clip.json`, `e1_recipe_siglip2.json`에 확정. **예측 등록**(upgrade_ledger): SigLIP2는 norm=true에서 norm=false 대비 폐루프 개선. 개선 없으면 그대로 기록(반증도 결과).

### E2. 공정 head-to-head (사용자 핵심 질문의 답)

**목적**: E1 승자 레시피끼리 **동일 통제**로 CLIP vs SigLIP2 최종 판정.

**실행**:
- CLIP-승자 vs SigLIP2-승자, 각 **50roll × 3seed**, paired, **차분 CI** 보고.
- **sigmoid 2×2는 여기서 제외** — 그건 action↔text 정렬 손실함수 문제이지 앵커 품질과 직교(`HANDOFF.md` §6.4). 언어 축(t2a·zero-shot)은 **별도 표**로 병기하되 앵커 우열 판정에 섞지 말 것.

**사전등록 판정(3-way, 사전 고정)**:
- 차분 CI가 SigLIP2 유리로 분리 → **"SigLIP2 우위"**
- CI 겹침 → **"구별 불가"** (현 상태 유지, "CLIP 우위" 서술 금지)
- 차분 CI가 CLIP 유리로 분리 → **"CLIP 우위 + 기제 후보"**(합성 렌더 도메인갭 / 과제 부분공간 안정성)

**산출물**: `outputs/report/e2_anchor_headtohead.json` + 판정 문서 `outputs/report/e2_verdict.md`. **이 결과로 `HANDOFF.md` §6.4를 재작성**(현재 "SigLIP2 미승"은 통계력 부족으로 폐기/수정). cowork 검토 큐로 회부.

---

## Block 3 — DINOv2 역할 교정: 관측 패치 융합 (E3–E4)

### E3. mean-pool → attention-pool 교체 (최소 원칙 수정)

**목적**: OpenVLA/Prismatic는 두 인코더의 **패치 토큰을 채널 concat해 정책에 투입하고 전역 pooling을 하지 않는다**(DINOv2=공간, SigLIP=의미). 현재 obs2(DINOv2 mean-patch **단일 토큰**)를 **공간 보존형 소수 토큰**으로 교체해 P6 shortcut(0.0987 적신호)과 proprio식 causal-confusion을 회피.

**구현(신규)**:
1. **패치 토큰 노출**: `Siglip2Anchor.encode_images`가 vision tower `last_hidden_state`(패치 토큰)를 `tokens`로 반환하도록 추가(현재 `None`). DINOv2는 이미 반환, CLIP은 `save_tokens` 경로 존재.
2. **`ObsFusion` 모듈**(`src/models/policy.py` 또는 `networks.py`): `mode ∈ {none, meanpatch(현행), attnpool}`.
   - `attnpool`: **K개 학습 쿼리**가 (선택 인코더별 공통 차원 사영 후 **토큰축 concat**된) 패치 토큰에 cross-attend → **K개 obs 토큰** 생성. `n_query=8` 기본.
3. **config 키(신규)**: `obs_fusion: {mode: attnpool, encoders: [dinov2] | [siglip2, dinov2], n_query: 8}`. FlowPolicy 조건 토큰에 K개 obs 토큰 append(기존 obs2 단일 토큰 자리 대체).
4. **로드 동기화**: `n_tokens`·`load_models` 로드 크기 동기화(`HANDOFF.md` §7.4 size mismatch 크래시 재발 방지).
5. **표준화**: 패치 토큰도 사영 후 LayerNorm 유지(§7.3 std 지배 교훈).

**실행**: **앵커=CLIP 고정**(융합 효과 격리), 3팔 비교 — (a) obs 없음(순수), (b) meanpatch 현행, (c) attnpool. 스크리닝 후 승자 50roll × 3seed.

**사전등록 판정(게이트)**: attnpool이 **(a)순수 대비 폐루프 CI-분리 개선 AND (b)meanpatch 대비 개선 AND P7 강건성 비열화**면 통과 → "DINOv2는 올바른 방식이면 기여" 실증. 하나라도 미달이면 → 융합 무효라는 정직한 부정 결과(E4 확장 판단).

**주의**: 성능이 아니라 **인과 유용성**이 관건 — 반드시 폐루프 + P7로 판정(오프라인 dec R² 상승은 §6.2 함정). 손목캠도 동일 obs 인코더 경유하는지 확인.

### E4. (조건부) 밀집 cross-attention — E3가 부분 성공 시에만

**목적**: attnpool(K토큰 요약)로 부족하면, flow denoiser가 패치 토큰 memory에 직접 cross-attend하는 밀집 버전.

**구현**: `obs_fusion.mode=xattn` — FlowPolicy 블록에 cross-attention 추가(잠재/flow 상태 → 패치 토큰 memory). 토큰 예산·레이어는 d1536 기준 소규모부터.

**사전등록 판정**: E3 승자 대비 **CI-분리 개선 시에만 채택**(비용 대비). 아니면 E3 승자 유지.

---

## Block 4 — 통합: 목표 아키텍처 (E5, E2·E3 결과 조건부)

**목적**: "각 능력 활용 + Δz 부위별 적용"의 완성형 — **앵커 = E2 승자(CLIP 또는 SigLIP2), 관측 = E3/E4 승자 융합(SigLIP2+DINOv2 패치)**.

**전제**: E2에서 앵커 확정 **AND** E3에서 융합이 게이트 통과했을 때만 실행. (둘 중 하나라도 실패면 그 자체가 결론 — 통합 강행 금지.)

**실행**: 확정 아키텍처를 50roll × 3seed, LIBERO-Spatial. 통과 시 suite 확장(Object/Goal 캐시 준비됨, §9 참조)으로 일반화 확인 + G3 혼동 행렬(Goal suite).

**사전등록 판정**: 통합이 **각 단일 축(앵커-only, 융합-only) 최고 대비 CI-비열화 이상**이면 승격. 목표 좌표를 NUMBER_CARD에 명기(공표 기준: π0 96.8 / OpenVLA 84.7 / OpenVLA-OFT 97.6 / LAPA 73.8, LIBERO-Spatial %).

---

## 실행 순서 및 예측 장부 규율

1. **E0**(프리플라이트, 반나절) → 통과 확인.
2. **E1 → E2**(앵커 공정화, 주력) → 사용자 핵심 질문 답 확정.
3. **E3**(융합 교정) → (조건부 **E4**).
4. **E5**(통합, 조건부).

- 각 블록 **실행 전** `docs/upgrade_ledger.md` 예측 장부에 예측 등록(장부 규율: 폐루프 예측 3연속 실패 이력 → 조건부 등록 유지), 실행 후 적중/반증 기록.
- 사이클마다 `docs/verification_log.md`에 발견→조치→판정 기록.
- **E2·E5 판정 문서는 cowork(이론 파트너) 검토 큐로.**

---

## 부록 A — 판정 게이트 요약표

| 실험 | 게이트 기준 | 실패 시 |
|---|---|---|
| **E0** | intact 대비 zeros/shuffle에서 align·dec R²·a2z 유의 하락 | 자명해 flag + cowork 회부, 이후 해석에 명시 |
| **E1** | 앵커별 폐루프 최적 레시피 확정 (offline 선정 금지) | — (탐색 실험) |
| **E2** | 차분 CI 분리 여부로 3-way 판정 (겹침=구별 불가) | "CLIP 우위" 서술 금지, §6.4 재작성 |
| **E3** | attnpool > 순수 AND > meanpatch AND P7 비열화 | 융합 무효 부정 결과, E4 판단 |
| **E4** | E3 승자 대비 CI-분리 개선 | E3 승자 유지 |
| **E5** | 단일 축 최고 대비 CI-비열화 이상 | 통합 미승격, 원인 분석 |

## 부록 B — config 키 요약 (신규/변경)

- 기존: `anchor.{name, projection, normalize, pooled, center_crop, model_dir}` (E1에서 normalize·정렬 스윕).
- **신규 `obs_fusion.{mode, encoders, n_query}`** — E3/E4. `mode ∈ {none, meanpatch, attnpool, xattn}`.
- **신규 `train_phase1` z_t 절제 스위치** — E0. `g_ztcond ∈ {intact, zeros, shuffle}`.
- `condition` JSON 필드에 `anchor / projection / normalize / pooled / obs_fusion / data_variant` 기록 의무.

## 부록 C — 산출물 파일 색인

| 실험 | 산출물 |
|---|---|
| E0 | `outputs/report/e0_trivial_probe.json` |
| E1 | `outputs/report/e1_recipe_{clip,siglip2}.json` |
| E2 | `outputs/report/e2_anchor_headtohead.json` + `e2_verdict.md` |
| E3 | `outputs/report/e3_obs_fusion.json` |
| E4 | `outputs/report/e4_xattn.json`(조건부) |
| E5 | `outputs/report/e5_integrated.json` + suite 확장 리포트 |

---

*본 지시서는 "각 모델 native 파라미터 재설정(E1) + 부위별 Δz 적용(앵커=E2 / 관측=E3) + 통합(E5)"을 사전등록 게이트로 검증하도록 구성됨. 모든 우열 주장은 3시드 + 차분 CI 통과를 전제로 하며, CI 겹침은 "구별 불가"로 판정한다.*
