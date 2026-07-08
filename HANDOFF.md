# 연구 핸드오프 — 원 코드베이스에서 현재까지 (감사용 상세본)

> 대상: 원 코드베이스를 아는 연구자. 목적: 실험 설계·구현·결과·분석을 **검증**하기 위함.
> 작성: 실행자(Claude Code) / 2026-07-08 / 브랜치 `exp/executor-log` (= repo main 스냅샷).
> **감사 원칙**: 모든 수치는 `outputs/report/*.json`(§8 스키마)이 정본. 본문 수치는 거기서
> 인용. 내가 만든 버그와 그 수정도 숨기지 않고 §7에 전부 기재했다 — 그 부분을 특히 검토 바람.

---

## 0. 이 문서를 읽는 법

- **§1–2**: 연구 아이디어와 파이프라인 (이미 아는 부분은 건너뛰어도 됨).
- **§3**: 원 코드베이스 대비 **내가 무엇을 어떻게 구현했는가** — 감사 1순위.
- **§4**: 실험 연대기 (왜 그 실험을 했고, 어떤 순서로).
- **§5**: 전체 결과표.
- **§6**: 핵심 발견 + **내 해석과 그 근거·반론** — 해석이 타당한지 검토 바람.
- **§7**: **내가 낸 버그와 수정** — 검증이 잡은 것들. 여기가 신뢰성의 핵심.
- **§7B**: **내 재설계 판단** — 버그는 아니나 설계가 틀렸을 수 있는 것. 자가 검증
  불가라 **연구자 판단이 가장 필요한 곳** (사용자 요청으로 신설).
- **§8**: 방법론 규율 (게이트·예측 장부·검증 이중화).
- **§9**: 미해결·주의·다음 단계.
- **§10**: 파일 색인.

---

## 1. 연구 아이디어

**테제**: 액션청크(7-DoF × 16스텝)를 frozen CLIP ViT-L/14 이미지 잠재공간의 **변위**
`Δz = z_{t+16} − z_t`에 접지한다. 인코더 g가 `g(A, z_t) ≈ Δz`를 배우면 CLIP의 시각-언어
공간이 액션의 의미 좌표계가 되고, flow matching 정책 f가 그 위에서 미래 잠재
`ζ = g(A_fut, z_t)`를 예측 → 동결 디코더 h가 액션 복원.

**신규성** (130편 서베이, `최종보고서_v2.md`): raw 액션청크를 frozen VL 공동공간 변위에
forward 회귀로 접지하는 조합은 선행 없음. 부품(DynE, DINO-WM, V-JEPA2-AC, MotionCLIP)은
개별 검증됨. **최대 리스크**: "문헌은 dynamics에 CLIP을 회피(DINOv2/SigLIP2 선호)". →
이에 대한 방어가 G2 앵커 대조 + ARM-AE 대조군 (§6).

## 2. 파이프라인 구조

```
phase1 (DeltaAE):  g: 액션청크(16×7) + z_t → ζ (768)   [1D-CNN, 상태조건]
                   h: Δz + z_t → 액션청크              [MLP, 상태조건]
   손실: align = g(a,z_t)≈Δz (MSE+cos) / recon = h(Δz,z_t)≈a (L1) / cycle = h(g(a,z_t),z_t)≈a
phase2 (정책 f):   토큰 [z_{t-16}, z_t, g(A_past), (lang), (wrist)] → flow matching → ζ̂
                   → h(ζ̂, z_t) = 액션청크,  receding horizon (16 예측 → 앞 8 실행, 20Hz)
```

- 데이터: LIBERO-Spatial (Franka 7D OSC, 인간 텔레옵 50데모/task × 10태스크, 20Hz, agentview).
  ALOHA 트랙은 기제 연구용 (스크립트 전문가, R² 0.98 레짐).
- **확정 레시피**: joint + **비정규화** Δz + **hybrid** 정렬(λ_c=0.3) + flow(source=past) +
  손목캠 토큰 + d1536. (비정규화·hybrid의 근거는 §4·§6.)

---

## 3. 원 코드베이스 대비 구현한 것 (감사 1순위)

원 코드베이스는 CLIP 기반 파이프라인이 이미 건전하게 구현돼 있었다(재현으로 확인, §4.0).
비교·검증을 위해 내가 추가/변경한 것:

### 3.1 앵커 추상화 (`src/core/anchor.py` — 신규)
- `BaseAnchor` 인터페이스: `encode_images/encode_texts/dim/patch_dim/has_text/id/cache_key`.
- 구현체: `ClipAnchor`(기존 ClipWrapper와 **출력 비트 동일** 검증), `Siglip2Anchor`,
  `Dinov2Anchor`. `get_anchor(cfg)` 팩토리.
- **캐시 키 분리** `{anchor_id}/{projection}/{normalize}` — 앵커·전처리 판별 혼합 방지.
- **감사 포인트**: `Dinov2Anchor`는 v2 보정됨(§7.2) — `do_center_crop=False`(기본),
  `pooled` ∈ {cls, clsmp}, register 부재 캐시키 접미사(`-nc`, `-clsmp`).

### 3.2 정렬 모드 (`src/models/networks.py` DeltaAE)
- `align_mode` ∈ {dz(회귀), direct(InfoNCE), hybrid(dz+λc·InfoNCE)}. C8 절제용.
- `contrast_loss` ∈ {infonce(SupCon 다중양성), sigmoid(SigLIP식 균형 쌍별, b₀ 설정)}.
- `contrast_head`(노름 분리 투영), `g_state_cond`/`h_state_cond`(QueST 절제),
  `encoder_kind` ∈ {cnn, strided, transformer, mlp}, `comp`/`vel` 손실.
- `LATENT=768` 하드코딩 제거 → `anchor.dim` 일반화 (policy.py 포함).

### 3.3 정책 확장 (`src/models/policy.py`, `train_phase2.py`)
- **병렬 연구자 캠페인에서 병합**: FlowPolicy(CFM, source noise/past/vision),
  build_policy_from_cfg, chunkrep(DCT, 보류), 손목캠 토큰. (git merge 35b7e95)
- 내가 추가: `lang_proj`(교차 앵커 언어 어댑터 — 무텍스트 앵커에 CLIP 텍스트 사영),
  `proprio_proj`(로봇상태 토큰, 표준화), `obs2_proj`(C7 제2 관측 토큰, 표준화).

### 3.4 데이터 로더 (`src/data/libero.py`)
- no-op 필터(OpenVLA 규칙), 앵커별 캐시 + 구 평면캐시 하위호환 폴백,
  `embeddings_meanpatch`(C7), `build_proprio`(필드 선택), `keep_indices`.

### 3.5 어휘 (`src/data/motion_lang.py` + json)
- 모션 문장 v1(템플릿) → v2(F2.5 증강) → **v3(dual-score 중립 — CLIP·SigLIP2 양쪽
  마진>0 문장만)**. bin 경계는 액션 분포에서 결정(§7 근거 기록).

### 3.6 평가·진단 스크립트 (`src/eval_libero/`, `src/diagnosis/`)
- `rollout_sim.py` 승급: 50롤 paired, Wilson CI, 상태 기반 실패 분류
  (reach/grasp/wrong_object/wrong_goal), SR@220, §8 JSON, 실패 영상 쿼터.
- `knn_baseline.py`(VINN 바닥선), `c8_zeroshot.py`, `c8_gapfix.py`(F0/F1), `c8_f2_prior.py`.
- 진단: `d1_action_hist.py`, `d2_rerender.py`, `d4_multimodality.py`/`d4_refined.py`,
  `p6_shortcut_probe.py`, `p7_robustness.py`, `alignment_report.py`, `make_presentation.py`.

---

## 4. 실험 연대기

### 4.0 재현 (원 코드베이스 건전성 확인)
ALOHA·LIBERO 양 트랙 재현 성공 (ALOHA phase1 R² 0.988, LIBERO 폐루프 flow 후 87%).
→ 원 CLIP 파이프라인은 건전. 이후 문제는 전부 "비교 확장"에서 발생.

### 4.1 Phase 1.5 — R² 붕괴 진단
질문: ALOHA 0.98 vs LIBERO 0.68 원인? D1(no-op) **기각**(raw에 no-op 0.12%뿐),
D2(해상도 256²) **경미**(+0.011), D3(청크 스케일) **트레이드오프**만, **D4(다봉성)**
= z-조건부 액션 분산이 스크립트의 3배 → 결정론 상한 0.59 vs 0.87. **flow 디코더 채택 근거.**

### 4.2 C8 — 정렬 방식 절제 (논문 중심 실험)
4팔 폐루프: DZ 87.0 / DA 76.0 / HY01 83.4 / **HY03 87.0**; 언어 축(t2a·zero-shot)은 역순.
→ **교차 패턴**: 폐루프는 Δz-접지, 언어는 직접 정렬 우세. **hybrid(λ0.3) 승격**(폐루프
무손실 + 언어 획득). ARM-AE 대조군(align=0): 오프라인 동급(0.679) but 폐루프 **73.6**
(−7.4pp CI분리) → **접지 자체의 기여 실증.**

### 4.3 앵커 매트릭스 (H2) + 오염 발견
오프라인은 DINOv2 압도(dec 0.740), 폐루프는 CLIP 압도. **그러나** 검증이 DINOv2 사용법
오염(center-crop + CLS 풀링)을 발견 → 조건부 격하 → **v2 공정 재실험**(§4.5).

### 4.4 부정 결과 (전부 사전 등록)
proprio −28pp(인과 혼동, P6가 사전 예측), quantile R² 0.767 = 지표 착시(동일공간 0.674),
노름 분리 기각, 3.2s 파일럿 언어 정렬 붕괴(→S2 설계 교정).

### 4.5 검증 사이클 (구현·평가 이중 검증 상설화)
- 사이클 1: 적대적 리뷰가 strided blind·obs2 누락·표준화 누락 검출; 앵커 감사가 DINOv2
  crop·CLS 오염 확정.
- 사이클 2: cowork(이론 파트너) 온보딩 회신 — sigmoid "일반 우위" 재해석, V-JEPA2-AC
  서술 정정, PosA-VLA 증거, 정본 문서 미푸시 지적.
- 사이클 3: **sigmoid 2×2** — 두 앵커 모두 InfoNCE 하회, SigLIP2≤CLIP 전 셀 (§6.4).

### 4.6 G2 최종 (DINOv2-v2 공정 재실험)
no-crop+clsmp: 폐루프 **79.2** [75.4–82.5] — 사전 등록 "혼합" 구간 적중. 오염 성분
+14.0pp(격차의 64%), 잔존 −7.8pp CI분리. **G2 무조건부 통과, 논문 수치 +7.8pp**(v1 +21.8 폐기).

---

## 5. 전체 결과표 (LIBERO-Spatial, 50롤/task paired, 시드2 조건부)

### 폐루프 SR (정본: outputs/report/*.json)
| 구성 | SR | CI | SR@220 | run_id |
|---|---|---|---|---|
| CLIP-HY03 (정규화, s2) | 87.0 | 84–90 | 86.4 | c8_closedloop_hy03 |
| CLIP-DZ (정규화, s2) | 87.0 | 84–90 | 86.2 | c8_closedloop_dz |
| CLIP-HY01 | 83.4 | 80–86 | 82.4 | c8_closedloop_hy01 |
| **CLIP-HY03 (비정규화=확정, s1)** | 81.0 | 77–84 | 80.4 | final_hy03_unnorm_s1 |
| SigLIP2-HY03 (비정규화, s2) | 80.2 | 76–83 | 78.8 | s2p_siglip2_hy03 |
| DINOv2-DZ v2 (no-crop+clsmp) | 79.2 | 75–83 | 78.0 | h2f_dinov2_v2 |
| CLIP-DA | 76.0 | 72–79 | 76.0 | c8_closedloop_da |
| **ARM-AE (align=0 대조군)** | 73.6 | 69–77 | 72.6 | c8ext_arm_ae |
| DINOv2-DZ v1 (crop+CLS, 오염) | 65.2 | 61–69 | 63.4 | s2p_dinov2_dz |
| +proprio (기각) | 59.0 | 55–63 | 56.8 | s1v2_dz_proprio |
| mlp 기준선 재평가 | 35.6 | 32–40 | 34.2 | p0_reeval |
| k-NN 바닥선 (VINN) | 18.2 | 15–22 | 18.0 | p0_knn5 |

### phase1 오프라인 (dec R² / a→Δz retrieval / align cos)
| 구성 | dec | a2z | align |
|---|---|---|---|
| DINOv2 v2 (no-crop+clsmp) | 0.764 | 56.1 | 0.739 |
| DINOv2 v1 (crop+CLS) | 0.740 | 56.2 | 0.753 |
| SigLIP2 (비정규화) | 0.693 | 37.6 | 0.689 |
| CLIP (비정규화, DZ) | 0.684 | 35.1 | 0.674 |
| CLIP-HY03 (비정규화) | 0.682 | 11.9 | 0.310 |
| ARM-AE (align=0) | 0.679 | 0.1 | 0.044 |

### 진단
- **D4 상한**: z 0.590 → +손목 0.698 → +proprio 0.746 (정보량↑). 폐루프: 손목 +26pp,
  proprio −28pp → **정보량 ≠ 인과적 유용성**.
- **P6 단독 MAE**(적신호 = 시각 0.1118 이하): proprio 0.0966🔴, gripper 0.1773⚪,
  dinov2-mp 0.0987🔴.
- **sigmoid 2×2 t2a** (v3): CLIP InfoNCE 52.7 / sig 42.4; SigLIP2 InfoNCE 40.8 / sig 33.4.

---

## 6. 핵심 발견과 내 해석 (타당성 검토 바람)

### 6.1 R² 붕괴 = 인간 데모 다봉성 [D4]
해석: LIBERO 인간 데모는 같은 상태에서 액션이 다봉(분산 3배) → 결정론 디코더 상한이
낮음. **근거**: D4 조건부 분산 + ALOHA 대조(0.87). **반론 가능성**: kNN 조건부 분산은
이웃 반경 유한으로 보수적 하계 → 절대값보다 트랙 간 비교로만 해석해야 함(문서에 명기).

### 6.2 오프라인↔폐루프 해리 (프로젝트 중심 주제)
DINOv2(오프라인 1위, 폐루프 최하)·proprio(정보↑ 폐루프↓)·quantile(지표 착시) — 3중.
해석: 오프라인 코딩 품질 ≠ 폐루프 강건성. **문헌 정합**(cowork §2): robomimic
(val loss 최적 ≠ 최고 정책), DINO-WM(CLS 폐루프 반토막), Burns/Temporal Trap. **함의**:
백본/토큰 선택을 오프라인 지표로 하면 안 됨 → 폐루프 SR이 유일 기준.

### 6.3 G2 = CLIP이 공정 구성에서도 폐루프 우위 (+7.8pp)
해석: 오프라인 코딩은 DINOv2 우세하나 폐루프는 언어 정렬 앵커(CLIP)가 이김. 잔존 격차
기제 후보 = 합성 렌더 도메인 갭(PosA-VLA 독립 보고 정합) 또는 과제 관련 부분공간 안정성.
**주의**: "DINOv2 원천 열위"로 서술하면 안 됨 — Theia(MuJoCo DINOv2≥CLIP)와 모순. **벤치마크
조건부, 파이프라인 성분 분해로 서술.** (v1 +21.8pp는 오염이라 폐기.)

### 6.4 SigLIP2가 CLIP을 이기지 못함 (사용자 직관 반증)
폐루프 80.2≈81.0, 오프라인 dec 동급, 언어 t2a는 CLIP 우위, sigmoid 2×2도 두 앵커 모두
InfoNCE 하회. 해석: **일반 VL 벤치마크 강도(SigLIP2>CLIP)가 이 Δz 파이프라인으로 비전이**
(§6.2 주제 반복). **cowork 확인 요청**: SAIL의 sigmoid 우위는 image·text **양타워 frozen**
정렬 — 우리는 g(액션 인코더)를 **학습**하는 비대칭이라 전이 안 될 disanalogy 가능성.

### 6.5 접지 자체의 기여 [ARM-AE]
align=0이면 오프라인 복원은 동급인데 폐루프 −7.4pp. → Δz 접지가 폐루프에 실질 기여.
논문의 "정렬 없이도 되는 것 아니냐" 반박용 핵심 대조군.

---

## 7. 내가 만든 버그와 수정 (검증이 잡은 것 — 신뢰성 핵심)

**이 섹션이 감사에서 가장 중요하다. 각 항목은 "증상 → 원인 → 수정 → 영향 런 처리".**

1. **`.gitignore data/` 비앵커 패턴** (초기): `src/data/` 로더 코드까지 커밋 제외.
   → `/data/` 앵커링. (원 머신에서 로더 파일 유실 원인이었음.)
2. **strided 인코더 마지막 3스텝 blind** [CRITICAL]: stride-2에 causal pad (2,0) 오사용
   → grad=0 실증. 수정 (k−s)=(1,0), 전 스텝 grad 검증. **buggy 런 무효화 → v2 재실행.**
3. **obs2 토큰 비표준화** [MED]: DINOv2 mean-patch(std 0.75)가 공유 LayerNorm을 ~20×
   지배 → C7 "무익" 오판 위험. 수정: o_mean/o_std 표준화 + ckpt 저장. (C7 phase2 학습
   시작 전 동기화.)
4. **load_models n_tok에 obs2 누락** [HIGH]: C7 폐루프 로드 시점 size mismatch 크래시.
   수정 (크래시 전).
5. **sigmoid 손실 스케일** [측정 무효]: 3형태 시도 — sum(~40× 과대→align 붕괴 a2z 0.5),
   mean-all(음성 지배→언어 붕괴 t2a 3.5), **balanced(pos/neg 평균)+b₀=−5.5 정착**.
   → 첫 2회 2×2 결과 무효 처리, 3회차로 재산출. (§6.4는 3회차 유효값.)
6. **DINOv2 center-crop + CLS** [앵커 감사, 비교 오염]: HF 기본 전처리가 렌더 테두리
   12.5% 삭제 + dynamics 부적합 풀링. → v2 재실험, v1은 "오염판" 딱지. (CLIP은 정사각
   입력에서 crop 무연산이라 **우연히** 무손상 — 비대칭 오염이었음.)
7. **quantile R² 지표 착시**: 정규화 공간이 바뀌면 R² 차원 가중이 달라짐 (0.767 자기공간
   vs 0.674 동일공간). → 절제에서 기각.
8. **모션 어휘 CLIP 편향**: v2 어휘 변별력 CLIP 36.7 vs SigLIP2 18.1 (2배) → v3 dual-score.

**메타**: 6·8은 "CLIP-first 코드베이스의 앵커 편향"(사용자 우려)이 코드/데이터 수준에서
실재했음을 보여줌. **"앵커 비교는 문헌 관행 구성으로만 승격" 규칙**을 프로토콜에 채택
(eval_protocol.md H2-fair v2).

---

## 7B. 재설계 판단 — 버그는 아니나 설계가 틀렸을 수 있는 것 (**최우선 검토**)

> §7은 "객관적으로 틀려서 잡은 것". 이 섹션은 **버그 없이 돌지만 설계 판단 자체가
> 틀렸을 수 있는 것**들이다. 나는 버그는 검증으로 잡을 수 있으나, "이 판단이 옳은가"는
> 자가 검증이 안 된다 — 연구자의 전문 판단이 필요한 지점. 각 항목:
> **[결정] / [대안] / [근거] / [틀렸을 수 있는 지점] / [확인법]**.

### 7B.1 교차 앵커 언어 어댑터 `lang_proj` (내가 발명 — 계획에 없음) ⚠️최고 위험
- [결정] 무텍스트 앵커(DINOv2)에 언어 조건화를 주려고 CLIP 텍스트(768)→anchor.dim 학습형
  선형 사영을 정책과 공동학습.
- [대안] (a) DINOv2엔 lang 토큰 자체를 빼기, (b) frozen 선형 대신 무학습 사영.
- [근거] 매트릭스에서 모든 앵커가 동일하게 태스크 조건화를 받아야 공정.
- [틀렸을 수 있는 지점] **DINOv2 폐루프에 CLIP엔 없는 학습형 어댑터가 추가**된다 →
  G2 비교가 오히려 DINOv2에 유리하게 기울 수도(반대 방향 불공정). 즉 G2 +7.8pp는
  "어댑터 있는 DINOv2"와 "네이티브 CLIP" 비교라 완전 대칭이 아님.
- [확인법] CLIP도 lang_proj 경유로 통일하거나, DINOv2 lang-off 대조 1런.

### 7B.2 "공정한 DINOv2 구성"의 재구성 = no-crop + clsmp ⚠️
- [결정] v2 = center-crop 제거 + CLS⊕patch-mean(2048d).
- [대안] DINO-WM은 **전체 patch grid**(요약 아님)를 씀; Theia도 spatial token.
- [근거] DINOv2 논문 프로빙 강구성이 CLS⊕mean; crop은 렌더 테두리 삭제.
- [틀렸을 수 있는 지점] clsmp도 **여전히 공간 정보를 요약 압축** → "공정"의 상한이
  아닐 수 있음. 즉 DINOv2를 문헌 관행대로 완전히 살리면 79.2보다 더 오를 여지 → G2
  결론(CLIP 우위)이 약해질 수 있음. + register 부재 아티팩트(우리 128²에선 검출 0이나).
- [확인법] 전체 patch-grid 관측(DINO-WM식) DINOv2 폐루프 1런.

### 7B.3 sigmoid 손실을 "균형 pos/neg 평균"으로 정의 ⚠️
- [결정] `-(mean_pos logσ + mean_neg logσ(-))`, b₀=−5.5.
- [대안] 정통 SigLIP = `-1/B Σ_i Σ_j logσ(label·(t·sim+b))` (sum-j, mean-i).
- [근거] 정통형은 B-결합 스케일이라 hybrid 가중합에서 튜닝 난해 → 균형형이 스케일 안정.
- [틀렸을 수 있는 지점] **cowork의 SAIL 비교는 정통 sigmoid를 전제** → 내 균형형이
  "sigmoid"를 충실히 대표 못하면 §6.4의 "sigmoid 우위 반증"이 무효. 즉 sigmoid가
  진다는 결론이 내 구현 아티팩트일 수 있음.
- [확인법] 정통 sum-j-mean-i + contrast_w를 1/B 스케일로 재실행 대조.

### 7B.4 D4 다봉성 추정기 (중심 발견의 토대) ⚠️
- [결정] z-공간 kNN(k=10, 동일 태스크·타 에피소드) 조건부 액션 분산 → 상한 =
  1 − E[Var(A|z)]/Var(A). MAX_G=4000 서브샘플.
- [대안] 밀도추정·가우시안혼합·조건부 흐름으로 Var(A|z) 추정.
- [근거] 무학습·구현 단순.
- [틀렸을 수 있는 지점] k·이웃 정의·서브샘플이 상한 절대값을 좌우. "3배"는 트랙 간
  비교로만 유효(문서화함)하나, **"다봉성이 R² 붕괴의 지배 요인"이라는 인과 주장**은
  추정기 설계에 의존 — 상관을 인과로 과대 서술했을 위험.
- [확인법] k 스윕(5/10/20)·다른 추정기로 상한 재계산해 순위 불변 확인.

### 7B.5 P6 지름길 프로브 프로토콜 + 적신호 기준
- [결정] 후보 토큰 단독 2층 MLP(512) 30ep val MAE, 시각 단독 이하 = 적신호.
- [틀렸을 수 있는 지점] 이미 **필요조건일 뿐**임이 드러남(gripper 2D 비적신호인데
  −19.4pp). MLP 용량·에폭·MAE 지표(그리퍼 제외)가 판정을 좌우. 적신호 임계가 자의적.
- [확인법] 프로브 용량 스윕 + 비적신호 후보도 소규모 폐루프 선행(이미 규칙 채택).

### 7B.6 P7 이탈도 지표 + 앵커별 캘리브레이션
- [결정] 롤아웃 프레임의 데모뱅크 kNN(10) cos 거리, 앵커별 held-out 데모로 정규화.
- [틀렸을 수 있는 지점] 캘리브레이션을 raw 수치가 교차공간 비교불가라서 **사후에 추가**함
  → 확증편향 소지. 영상 표본이 성공/실패 불균형(DINOv2는 실패 위주).
- [확인법] z-트레이스 직접 로깅(영상 재인코딩 아님)으로 전 에피소드 재측정.

### 7B.7 레시피 확정 로직 = "비정규화" ⚠️
- [결정] 오프라인 retrieval + zero-shot 안정성 근거로 비정규화 확정.
- [틀렸을 수 있는 지점] **폐루프 비교(81.0 vs 정규화 87.0)는 시드·레시피 교란 상태**
  였는데 확정을 진행함. 비정규화의 폐루프 비용을 배제 안 하고 대표 레시피로 굳혔을 위험.
- [확인법] 비정규화@seed2 폐루프 절제 1런 (미실행 — §9).

### 7B.8 실패 모드 분류기 임계 / 모션 어휘 스킴 / comp 손실 근사
- 실패 분류기: 변위 2cm·리프트 5cm·타물체 5cm 임계는 내가 고른 상수 — 오분류 가능.
- 모션 어휘: 카테고리 스킴(지배축×방향×2-bin), bin 경계=중앙값, v3 dual-score 선별
  규칙(양 앵커 마진>0) 전부 내 설계 — C8·언어 결과 전반에 영향.
- comp 손실: `z_mid ≈ z_t + g_a` 근사는 span=n_chunk=16일 때만 타당(리뷰 확인). 그 외엔
  half-chunk가 외삽.
- cache_key: 초기 {id/proj/norm}이 **전처리 판(crop 여부)을 안 담아** DINOv2 오염 캐시
  혼합 위험이 있었음 → id 접미사로 사후 보강(설계 초기 불완전).

**요지**: §7(버그)은 닫혔다고 보지만, **위 재설계 판단들은 열려 있다**. 특히 7B.1(어댑터
비대칭)·7B.2(공정 DINOv2 상한)·7B.3(sigmoid 충실성)·7B.4(다봉성 인과)·7B.7(레시피 확정)은
**논문 주장에 직접 영향**하므로 우선 검토 바람. 내 판단이 틀렸다면 G2·SigLIP2·다봉성
서사가 흔들릴 수 있다.

---

## 8. 방법론 규율

- **게이트** (사전 등록, RESEARCH_PLAN §7): G0(R²≥0.85, 진단으로 해소), G2(언어정렬 SR ≥
  DINOv2 −10pp, **통과 +7.8pp**), G3/G5/G-floor 등. `docs/upgrade_ledger.md` 게이트표.
- **예측 장부** (upgrade_ledger): 사전 등록 vs 실측 9건. 폐루프 예측 3연속 실패(proprio
  gripper·DINOv2·SigLIP2) 후 조건부 등록으로 개정 → #8(DINOv2-v2) 첫 적중, #9(sigmoid) 반증.
- **평가 프로토콜** (eval_protocol.md): fp16 비결정성 태스크별 ±30pp → **suite 평균(n≥500)만
  공식**, 태스크별은 참고치. <3pp 차이는 반복 규칙. 시드 스크리닝(10eps/task) 후 승자만 전량.
- **검증 이중화** (verification_log.md): 구현 적대적 리뷰 + 앵커 적응 조사 + 이론 파트너
  (cowork). 사이클마다 발견→조치→영향 판정 기록.

---

## 9. 미해결·주의·다음 단계

**미해결/주의**:
- 대표 수치 81.0(비정규화 s1) vs 87.0(정규화 s2): 레시피×시드 교란 — 절제 1런(비정규화
  @seed2) 미실행. **비정규화의 폐루프 비용 가능성 배제 안 됨.**
- 3시드 완주 미실시 → 전부 "시드2/시드1 조건부".
- C7 융합 셀: P6 적신호(dinov2-mp 0.0987)로 스크리닝 필수 — 폐루프 미실행.
- P7-v2 이탈도(잔존 −7.8pp 기제 판정)·CI-MSE 정독·SigLIP2 sigmoid disanalogy: cowork 큐.
- 문서 4종(upgrade_report/journal 등 병렬 캠페인 원본) 미도달 — 캠페인 수치 프로토콜 미확정.

**다음 단계**:
1. S2 (계층 하이브리드): 상위층 긴 창(1.6–3.2s) Δz ← 태스크 지시문 정렬 / 하위층 0.8s +
   모션 문장 (3.2s 파일럿 교훈). suite 확장(Object/Goal 캐시 준비됨) + G3 혼동 행렬.
2. 대표 수치 확정 (교란 절제 → 3시드).
3. Phase 3 언어 정량화 (L2–L4 어댑터).

---

## 10. 파일 색인 (감사용)

| 무엇 | 어디 |
|---|---|
| 계획서 (사전 등록) | `RESEARCH_PLAN_delta_anchor_v1.1.md` |
| 130편 서베이 | `최종보고서_v2.md` |
| 게이트·예측장부·판정이력 | `docs/upgrade_ledger.md` |
| 평가 프로토콜 | `docs/eval_protocol.md` |
| 검증 사이클 기록 | `docs/verification_log.md` |
| 경쟁 논문 차별화 | `docs/related_competitors.md` |
| 발표자료 정정 | `docs/talk_errata.md` |
| cowork(이론 파트너) | `docs/COWORK_*.md`, `verification_reply_20260707_onboarding.md` |
| 판정 문서 | `outputs/report/{g2_final_verdict,arm_ae_verdict,proprio_final_verdict,matrix_final_table,phase1_5_diagnosis,...}.md` |
| 전 런 §8 수치 | `outputs/report/*.json` |
| 그림·숫자카드 | `outputs/presentation/` (`NUMBER_CARD.md`, `INDEX.md`, FIG1–10) |
| 코어 코드 | `src/core/anchor.py`, `src/models/{networks,policy}.py`, `src/data/{libero,motion_lang}.py` |
| 학습·평가 | `src/training/train_phase{1,2}.py`, `src/eval_libero/`, `src/diagnosis/` |

**검토 요청 우선순위**: **§7B(재설계 판단 — 최우선)** → §7(버그·수정) → §6(해석
타당성) → §3(구현 정확성) → §8(규율). §7B는 내가 자가 검증할 수 없는, 재량으로 내린
설계 판단이라 연구자의 전문 판단이 가장 필요하다. 특히 7B.1(lang_proj 어댑터 비대칭),
7B.2(공정 DINOv2 상한), 7B.3(sigmoid 충실성), 7B.4(다봉성 인과 주장), 7B.7(레시피 확정)이
틀렸다면 G2·SigLIP2·다봉성 등 핵심 서사가 흔들린다 — 여기부터 봐주면 좋겠다.
