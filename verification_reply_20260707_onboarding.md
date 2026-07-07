# 검증 파트너 회신 #1 — 온보딩 확인 + 선제 보고 + 조사 과제 3건 (2026-07-07)

> 작성: 이론·문헌 검증 파트너 (Cowork). 실행자가 repo `docs/`에 그대로 커밋 가능한 형태.
> 검증 수준 표기: [1차 확인] = 원문/공식 코드 직접 대조, [에이전트 검증] = 조사 에이전트가
> 1차 출처 대조(본인이 표본 재확인 2건 통과), UNVERIFIED = 1차 출처 미확인.
> 조사 수행: 병렬 조사 에이전트 3개 + 감사(본인). 신규 인용 표본 재검증: PosA-VLA
> 2512.03724, SAIL 2412.04616 — 둘 다 arXiv abs 페이지 [1차 확인] 통과.

---

## §0. 선제 보고 (스스로 발견한 위험 — 응답 규약에 따라 최우선 기재)

### 0.1 [차단급] repo main에 정본 문서 다수 누락 — 수치 인용 규율이 현재 이행 불가

jsDelivr로 main 브랜치 전체 파일 트리를 열람한 결과(2026-07-07), 문서들이 참조하는
다음 파일이 **push되어 있지 않다**:

- `outputs/presentation/NUMBER_CARD.md` — **내 수치 인용 규율의 경유지 자체가 없음**
- `docs/verification_log.md` — H2-fair v2 규칙("verification_log에 기록 후에만 비교표 승격")의 기록처
- `docs/related_competitors.md`, `docs/talk_errata.md`
- `outputs/report/*.json` (§8 스키마 전 런) — 현재 git에는 `arm_ae_verdict.md` 1건만 존재
- `data/motion_lang_v3.json` (v3 dual-score 어휘 — v1/v2만 존재)

**영향**: 이 회신의 수치는 부득이 repo 내 존재 문서(README_EXPERIMENTS.md,
upgrade_ledger.md, eval_protocol.md) 기준으로 인용했다. **조치 요청(실행자)**: 위 파일들
push. push 전까지 내 쪽 수치 인용은 "README_EXPERIMENTS 경유"로 표기함.

### 0.2 anchor.py 사용법 감사 결과 (v2 보정 구현 — 통과, 주의 3건)

`src/core/anchor.py` [1차 확인]:

- **DINOv2 v2 보정 정확**: `do_center_crop=False` + 224 직접 resize (DINO-WM 등 로봇
  관행), `pooled="clsmp"`(CLS⊕mean-patch, 2048d = DINOv2 논문 프로빙 프로토콜의 강한
  구성), 캐시 키 분리(`dinov2-large-nc-clsmp`)로 구 crop판 캐시와 혼합 방지. HF
  `Dinov2Model.pooler_output` = layernorm 후 `last_hidden[:,0]`이 맞으므로 주석도 정확.
- **주의 1 — 고노름 아티팩트 토큰**: `facebook/dinov2-large`는 **register 없는 판**이라
  저정보 배경 패치에 고노름 아티팩트 토큰이 발생함 (*Vision Transformers Need
  Registers*, 2309.16588 [에이전트 검증]). mean-patch 풀링이 이를 그대로 평균에 편입
  → **P7-v2에 패치 노름 분포(p95/p99) 모니터링 추가** 권고. v2 결과가 기대 이하면
  `facebook/dinov2-with-registers-large` 교차 확인이 1순위 절제.
- **주의 2 — SigLIP2 토크나이저**: `padding="max_length"`는 SigLIP 학습 관행과 일치
  (올바름), `get_text_features`에 attention_mask 미전달도 공식 사용법과 일치. 단
  `max_length` 명시가 없으므로 `tokenizer.model_max_length == 64` 1줄 확인 권장
  (SigLIP2 텍스트 타워는 64 토큰 학습).
- **주의 3 — 문서 정정 필요 (V-JEPA 2-AC)**: 우리 문서들이 전제해 온 "V-JEPA2-AC는 raw
  특징에 L1"은 **부분 부정확**. 공식 config(`loss_exp: 1.0`, `normalize_reps: true`)와
  코드상, 타깃·예측 **양쪽에 parameter-free LayerNorm 표준화 후 L1**이다
  (facebookresearch/vjepa2 `app/vjepa_droid/train.py` [에이전트 검증]). → §4의 완화책
  예비안과 upgrade_ledger 관련 행 갱신 제안 참조.

### 0.3 [기회] G2 서사를 보강하는 신규 독립 증거 발견

**PosA-VLA (2512.03724, §9)** [에이전트 검증, abs 페이지 1차 확인]: LIBERO 시뮬 적응에서
*"we use the CLIP image encoder as the visual backbone instead of DINOv2, since we
observed that DINOv2 features underperform on Libero's synthetic renderings"* — 동일
벤치마크·flow matching 계열에서 **DINOv2→CLIP 교체를 독립 보고**. G2 서사(및 서베이
"최대 리스크" 방어)의 외적 타당성 증거로 `related_competitors.md`/논문 related work에
인용 확보 권고. 단 정량 격차는 미보고(관측 진술만)이므로 보조 증거로만.

---

## §1. 온보딩 확인 답변

### (a) 신규성 주장 · 최대 리스크 · 방어 실험

우리의 신규성은 raw 액션청크(7-DoF×16)를 **frozen CLIP vision-language 공동 공간의
변위 Δz = z_{t+16}−z_t에 forward 회귀로 접지**한다는 조합 자체다 — 액션 임베딩이 미래
latent를 예측하게 하는 것(DynE 1908.09357), frozen 시각 latent의 시간차가 액션으로
예측 가능하다는 것(DINO-WM 2411.04983, V-JEPA2-AC 2506.09985), 새 모달리티를 frozen
CLIP 공간에 바인딩하는 것(MotionCLIP 2203.08063, ImageBind 2305.05665)은 각각 검증돼
있으나, 감독 신호가 자기 재구성이 아닌 **외부 고정 VL 공동 공간**인 사례는 130편 서베이
기준 미점유다. 최대 리스크는 그 공간 선택 자체다: 2024–25 문헌은 dynamics 기질로 CLIP을
일관되게 회피하고(MAE/VQGAN/SigLIP2 — GR-1, Seer, FLARE), CLIP-blind pairs(2401.06209)·
temporal entanglement(2502.03270)·공간 정밀도 부족(2312.12444, 2407.20179)이 근거로
제시되므로 "잘못된 공간 위에 지었다"는 비판이 표준 공격선이 된다. 방어는 이중이다:
① **G2 앵커 대조** — 동일 파이프라인에서 DINOv2 폐루프 65.2 vs CLIP 87.0 (+21.8pp,
README_EXPERIMENTS 경유; 단 crop·CLS 오염 발견으로 조건부 격하, v2 재실험 진행 중),
② **ARM-AE 대조군** — align=0이면 오프라인 동급인데 폐루프 −7.4pp CI 분리 → CLIP-Δz
접지 *자체*의 기여 실증. 여기에 이번 조사로 PosA-VLA(§0.3)와 "오프라인↔폐루프 해리"
문헌군(§2)이 외곽 방어선으로 추가된다.

### (b) DINOv2-v2 재실험의 필요성 (사고 경위) · 사전 등록 3구간 판정 규칙

사고 경위: G2 1차 런에서 DINOv2 임베딩이 HF 기본 프로세서 경로(짧은 변 256 resize →
224 center-crop, **시뮬 렌더 테두리 ~12.5% 삭제**)와 **CLS 단독 풀링**으로 생성됐다.
둘 다 dynamics/제어 용도의 문헌 관행이 아니다(DINO-WM은 patch 임베딩, DINOv2 표준
강프로브는 CLS⊕mean-patch). CLIP은 정사각 입력에서 224 resize→224 crop이 무연산이라
**우연히** 무손상이어서 비교가 비대칭으로 오염됐다(eval_protocol.md의 각주 의무 사항).
더 뼈아픈 것은 절차 실패다: 예측 #4([87,93])가 65.2로 대실패했을 때 측정 계보 감사보다
표현 속성 해석("의미 불변성")을 먼저 전개했고, 이후 감사가 오염을 발견했다 — 그 교훈이
"역전급 신호는 감사가 해석에 선행"이라는 순서 규칙으로 upgrade_ledger에 등재됐다.
v2 재실험 = no-crop(224 직접 resize) + clsmp 풀링 + 신규 캐시 키. 사전 등록 판정
(예측 장부 #8): 폐루프 SR **≥83.8**(=CLIP-DZ 87.0의 Wilson 하한)이면 역전=아티팩트로
헤드라인 #2 철회, **70–83.8**이면 혼합(오염이 일부 설명, 잔존 격차 주장은 약화 유지,
기제 판정은 P7-v2로 이관), **<70**이면 속성 확정. 부대 예측: 오프라인 dec ≥0.75·검색
상승·wrong_object 감소하되 비소멸, 기대 구간 [72,84]. 문헌 대조(§2 상세): DINO-WM
Table 2가 CLS 풀링 단독으로 폐루프 SR 반토막(PushT 0.90→0.44)을 실증하므로 v1 65.2의
상당 부분이 풀링 오염일 개연성이 높고, 동시에 Theia Table 12는 MuJoCo 폐루프에서
DINOv2-L≥CLIP-L이므로 **v2가 세 구간 어디에 떨어져도 문헌과 모순되지 않는다** — 즉
판정은 백본 본질론이 아니라 파이프라인 성분 분해로 서술해야 한다.

### (c) 폐루프 예측 반복 실패의 원인 가설 · 개선의 문헌 근거

실패 3건(#2 그리퍼2D 통과 예측→−19.4pp, #4 DINOv2 [87,93]→65.2, #5 SigLIP2 언어
우세→열세)의 공통 구조는 **"오프라인 정보량/적합도에서 폐루프 성능을 외삽"**한 것이다.
내 가설은 3부: **(1) 범주 오류** — 오프라인 지표는 데모 분포(i.i.d.)에서의 코딩 품질을
재지만, 폐루프 SR은 자기 유발 분포이동 하의 오류 누적(O(T²), Ross & Bagnell 1011.0686)과
지름길 취약성(causal confusion, de Haan 1905.11979: "access to more information can
yield worse performance" — proprio −28pp의 교과서 사례)이 지배한다. robomimic
(2108.03298)은 val loss 최적 정책이 최고 정책 대비 50–100% 나쁨을 실증 — 해리는 이
분야의 구조적 현상이지 우리 파이프라인의 결함이 아니다. **(2) 조절 변수 누락** — 예측이
백본/토큰 수준의 명성("DINOv2는 dynamics에 좋다", "proprio는 정보 추가")으로 발행됐고
풀링·전처리·어휘·손실 계열·렌더 도메인 같은 파이프라인 조건을 조건화하지 않았다.
문헌상 정확히 이 변수들이 결과를 뒤집는다(DINO-WM CLS 0.44 vs patch 0.90; PosA-VLA의
LIBERO 렌더 관측; SAIL의 손실 계열 효과 §3). **(3) 구간 보정 실패** — fp16 비결정성
(태스크별 ±30pp) 레짐에서 [87,93] 폭의 구간은 과신이다. 개선안(문헌 근거): 이미 등재된
"코딩 축 + 강건성 축(P7) 분리" 개정을 지지하며 — P7이 재는 것이 문헌이 성공 예측자로
지목한 바로 그 축이다(Burns 2312.12444: emergent segmentation이 OOD 성공의 **순위**
예측자; Temporal Trap 2502.03270: SR ∝ task-progression 코딩) — 추가로 ① LAOM
(2502.00379)의 "프로브 8× 개선 → 다운스트림 ~2×"를 **오프라인 이득의 할인 계수
휴리스틱**으로 채택, ② 예측을 무조건 구간이 아닌 **조건부 등록**(전처리·풀링·어휘·손실
명시, H2-fair 감사 통과 전 구간 발행 금지)으로, ③ 근소 차이 레짐에선 점추정 대신
서수 예측(A>B만 등록)으로 전환할 것을 제안한다.

---

## §2. 조사 1 — Frozen 백본 잠재공간의 폐루프 실패 모드 (과제 6)

### 2.1 요약 결론

1. **"오프라인 우세 ↔ 폐루프 붕괴" 해리 자체는 반복 보고된 현상**: robomimic(2108.03298,
   val loss 기준 선택 시 50–100% 손해), LAOM(2502.00379, 프로브 8×↑ ≠ 제어↑),
   Temporal Trap(2502.03270), Burns(2312.12444) — 독립 4계열 + 이론 앵커 2개(DAgger
   1011.0686, causal confusion 1905.11979). 우리 DINOv2 역전은 그 자체로 이상 관측이
   아니다.
2. **CLS 풀링의 폐루프 붕괴는 직접 정량 증거 존재**: DINO-WM(2411.04983) Table 2 —
   PushT SR patch 0.90 vs **DINO CLS 0.44**, Wall 0.96 vs 0.58 (오프라인 LPIPS에서는
   CLS가 중위권 = 해리 재현). 원문: "world models that encode observations as a single
   latent vector show a significant drop in performance … losing crucial spatial
   details necessary for manipulation tasks."
3. **단, "DINOv2가 폐루프에서 원천 열위"는 문헌이 지지하지 않음**: Theia(2407.20179)
   Table 12의 MuJoCo류 폐루프에서 DINOv2-L ≥ CLIP-L (Assembly 93.3 vs 69.3 등).
   DINO-WM 성공 자체가 DINOv2 patch 기반.
4. **LIBERO 한정 "DINOv2 < CLIP" 독립 관측 존재**: PosA-VLA(2512.03724 §9) — 합성
   렌더 도메인 갭이 유력 조절 변수.
5. V-JEPA2-AC(2506.09985) §4.3 자체 보고 실패 모드: 카메라 좌표축 추론 민감성("we
   manually tried different camera positions"), 잠재 롤아웃 누적 오류("the accuracy of
   the representation-space predictions decreases with longer autoregressive
   rollouts"), 정밀 파지 실패(grasp cup 65%/box 25%) — 모두 오프라인 1-step 지표에
   안 잡히고 폐루프에서만 드러나는 유형.

### 2.2 논문별 검증 표

| 논문 (arXiv ID) | 설정 | 핵심 근거 (원문 인용) | 우리 관측과의 대응 |
|---|---|---|---|
| DINO-WM (2411.04983) | frozen DINOv2 **patch** + latent MPC, 폐루프 SR | CLS 절제: PushT 0.44 vs patch 0.90; "patch-based representations better capture spatial information, in contrast to … DINO CLS" | **지지(강)** — v1 오염(CLS)이 65.2의 주요 성분일 개연성. 단 CLIP 비교는 없음 |
| V-JEPA 2-AC (2506.09985) | frozen video enc + action-cond predictor, 실물 Franka MPC | §4.3: 카메라 민감성·autoregressive error accumulation·정밀 파지 실패 | **지지(중)** — frozen-표현 제어의 대표 실패 모드 자가 문서화 |
| robomimic (2108.03298) | 시각 BC, 폐루프 | "the best validation loss does not correspond to the best performing policy … 50 to 100% worse" | **지지(강)** — 오프라인 지표 기반 선택의 손해 정량화 |
| Burns et al. (2312.12444) | 15 frozen PVR + BC, 분포이동 폐루프 | "emergent segmentation ability is a strong predictor of out-of-distribution generalization … more predictive than ImageNet accuracy, in-domain accuracy, or shape-bias" | **부분 지지+긴장** — 표준 지표 무력은 지지; 단 이 예측자는 대체로 DINOv2에 유리 → 우리 역전은 도메인/파이프라인 요인으로 서술해야 |
| Temporal Trap (2502.03270) | frozen PVR 정책, 폐루프 | "strong correlation between a policy's success rate and the ability of its latent space to capture task-progression cues" | **지지(중)** — Δz 설계가 겨냥한 축과 동일; P7-v2의 이론 근거 |
| Theia (2407.20179) | distilled/frozen 인코더, CortexBench 폐루프 | spatial token 채택 근거 + Table 12: DINOv2-L ≥ CLIP-L (MuJoCo) | **반박 데이터점** — "CLIP이 이겨야 정상"도 아님. 벤치마크 조건부 |
| PosA-VLA (2512.03724 §9) | VLA(flow matching), LIBERO | "DINOv2 features underperform on Libero's synthetic renderings" → CLIP 교체 | **직접 지지(가장 근접 선례)** — 정량 미보고, 보조 증거 |
| LAOM (2502.00379) | LAM+BC, Distracting Suite 폐루프 | 프로브 8× 개선 → "the resulting performance is only slightly better than simple Behavioral Cloning" | **지지(강)** — 프로브↑ ≠ SR↑의 LAM 선례 + action-correlated distractor 개념 |
| What Do LAMs Learn? (2506.15691) | 이론 | Δ프레임 잠재는 "controllable changes as well as exogenous noise"를 흡수 | **지지(이론)** — Δz의 위험 요인 규명 |
| DAgger (1011.0686) / Causal Confusion (1905.11979) | 이론 앵커 | 누적 오류 O(T²); "access to more information can yield worse performance" | proprio −28pp의 직접 앵커 |
| JEPA-WMs (2512.24497, TMLR 2026) | frozen enc + predictor 설계 공간 연구 | multi-step rollout loss를 표준 설계 축으로 취급, DINO-WM·V-JEPA2-AC 상회 | **정황 지지** — 잠재 롤아웃 누적 오류가 공인 병목 |
| PVM-in-MBRL (2509.12531) | PVM+MBRL 폐루프 | "existing work has found PVMs to be ineffective in MBRL … partial fine-tuning can maintain the highest average task performance" | **맥락 지지** (초록만 1차 확인) |

### 2.3 v2 결과 해석 지침 (사전 등록 보완)

1. **[72,84] 착지 + CLIP 미달** → PosA-VLA 정합: "오염 제거로 대부분 회복, 잔존 격차는
   합성 렌더 도메인 갭" 서사. 문헌 무모순.
2. **<70** → 풀링을 1순위 용의자로 재감사: clsmp(concat)은 DINO-WM/Theia의 '전체
   spatial token' 사용과 여전히 다름(공간 정보 압축). + register 아티팩트(§0.2 주의 1).
3. **>84 (사전 등록 위반 방향)** → Theia·DINO-WM 방향과는 정합이므로 그 자체는 이상
   아님. 단 v1 대비 성능 성분 분해(crop 제거분 vs 풀링 교체분) 의무 — crop만/풀링만
   교차 셀 2런이 최소 설계.
4. **백본 선택을 오프라인 지표로 하지 말 것** (robomimic·LAOM·Temporal Trap 3중 근거)
   — 보고는 하되 선택 기준은 항상 폐루프 SR.
5. 백본 간 비교는 동일 horizon에서만 (V-JEPA2 error accumulation — 16-step 청크 고정
   유지).

---

## §3. 조사 2 — "시그모이드 사전학습 + InfoNCE 재정렬" 비정합 (과제 7)

### 3.1 요약 판정

| 질문 | 판정 |
|---|---|
| sigmoid 공간과 InfoNCE 공간의 기하가 다른가 (이론) | **예 — 정리 수준 확립** (2509.18552, 2005.10242) |
| "시그모이드 공간 = 더 조밀한 텍스트 콘"이 일반 예측인가 | **아니오 — 반대 방향 실측 존재** (2603.17246) |
| frozen SigLIP 공간에 InfoNCE 정렬 실패의 직접 보고 | **직접 증거 부재** (시도 기록 자체가 없음) |
| frozen 타워 정렬에서 sigmoid > InfoNCE 통제 비교 | **있음 — SAIL (2412.04616, CVPR 2025 Highlight)**: 동일 조건 CC3M에서 InfoNCE→sigmoid 교체로 IN-1K 45.4→50.7, COCO T2I R@1 16.1→25.4 |
| frozen SigLIP2에 sigmoid(+learnable t,b) 정렬 선례 | **있음** (2606.24080, 오디오 인코더) |
| **시그모이드 손실 교체 권고 강도** | **중~강 (조건부)** |

### 3.2 이론 핵심 (우리 판정의 뒷받침 + 재해석)

- **InfoNCE**: Wang & Isola(2005.10242) — 점근적으로 alignment + **uniformity**(초구
  균등 분포)를 최적화. 반발항(log-sum-exp)이 배치 결합.
- **Sigmoid**: 쌍별 독립, 배치 정규화 항 없음. **Global Minimizers of Sigmoid
  Contrastive Loss(2509.18552)**: 전역 최적해는 내적 분리 조건만 만족하는
  constellation — **uniformity 요구가 전혀 없음**. 임의로 조밀한 콘도 마진만 확보하면
  완전 최적 + retrieval 완벽. 또한 학습형 (t, b)가 locked 인코더 위 "암묵적 선형
  어댑터"로 기능(Observation 1) — **frozen SigLIP2 + sigmoid 재정렬의 이론적 정당화**.
- SigLIP 실측 공간(2509.18552 Table 2): 교차모달 양성 cos 0.095–0.138, 마진 ≈0.06–0.07
  — 유사도가 **b_rel 근방 좁은 밴드에 마진 기반으로 캘리브레이션**돼 있어, 스프레드에서
  그래디언트를 얻는 InfoNCE 헤드와 작동 방식이 다름.

### 3.3 우리 판정에 대한 함의 — 지지하되 재해석 필요

- **"목적함수 비정합" 판정의 뼈대는 유지 가능**하나, 문헌이 실제로 입증하는 형태는
  "사전학습 손실 계열 매칭"이 아니라 **"frozen 타워 + 경량 헤드 정렬에는 sigmoid가
  일반적으로 우수"**다(SAIL은 타워가 sigmoid 사전학습이 아닌데도 sigmoid 승리; LiT
  2111.07991·SigLiT는 교차 계열도 성공).
- **판별 실험 제안(사전 등록 권고)**: C8 sigmoid 교체를 SigLIP2 트랙에만 걸지 말고
  **CLIP 트랙에도 거는 2×2 (backbone × loss)**. sigmoid가 CLIP 트랙에서도 오르면
  결론은 "sigmoid의 일반 우위"로, SigLIP2에서만 오르면 "계열 매칭"으로 확정 — 예측
  장부 신규 행으로 등록할 것.
- **우리 측정(쌍별 cos 0.891 vs 0.863)은 "이론상 허용, 일반 예측은 아님"**:
  2603.17246(Table 1, Mean Resultant Length)은 자연 데이터 4종 전부에서 **CLIP 텍스트
  콘이 SigLIP보다 조밀**(반대 방향). 우리 관측은 sigmoid 일반 시그니처가 아니라
  SigLIP2 고유 레시피/모션 어휘 OOD 효과일 가능성 — 지표를 (Mean Resultant Length,
  앵커별 negative 스프레드 IQR, centroid 갭)으로 확장 재측정 권고. t2a 격차의 더
  그럴듯한 기하 원인은 콘 절대 밀도보다 **앵커별 negative 유사도 스프레드**다(InfoNCE
  그래디언트 구조상 — [추론] 표기).
- **SigLIP2 비순수성 주의**: sigmoid 외에 캡셔닝 디코더·자기증류·마스크 예측·다국어가
  결합된 다중 목적 학습(2502.14786) — sigmoid 교체가 격차를 전부 닫지 못할 수 있음
  (어휘 교정이 ~45%만 설명했던 것과 같은 잔여 구조 가능).

### 3.4 시그모이드 교체 구현 시 함정 (원문 근거)

1. **초기화**: b₀=−10, t′₀=log 10 (SigLIP 2303.15343 원문 — b=0이면 초기 대규모 편향
   교정 스텝으로 저하). 단 −10은 배치 ~수만 기준이라는 해석이 있고, open_clip
   discussion #687에서 Beyer가 온도-편향 직접 관계를 시사 — **소배치인 우리 설정에선
   b₀ ∈ {−10, −log(B−1)} 스윕** 권장.
2. **per-anchor 학습 온도 폐지 → 전역 (t, b) 각 1개** (SigLIP 표준; 앵커별 온도 하
   sigmoid는 선례 없음 = UNVERIFIED 영역).
3. 참고 수렴값: t≈117.8, b≈−12.9 (b_rel≈−0.11; 2509.18552 §3.4 — 모델 버전 귀속
   UNVERIFIED). **우리 체크포인트에서 `model.logit_scale.exp(), model.logit_bias`
   직접 확인 1줄 추가** 권장.
4. sigmoid는 소배치에서 softmax 대비 유리(2303.15343 abstract) — 우리 배치 규모에
   불리하지 않음.

---

## §4. 조사 3 — 비정규화 Δz: 노름 정보 보존 vs 폐기 (과제 8)

### 4.1 판정: **비정규화 유지의 문헌 방어 강도 = 강함** (정렬≠디코딩 구분 조건부)

핵심 논리: **"정렬(alignment)" 문헌과 "미미크리/디코딩" 문헌은 준거집단이 다르다.**
코사인(방향 전용) 진영의 성공 사례(MotionCLIP 2203.08063, REPA 2410.06940, FLARE
2505.15659, BYOL 2006.07733)는 전부 *타깃을 복원할 필요가 없는 보조 정렬*이다. 우리
h는 Δz-접지 매니폴드에서 **액션을 복원**해야 하므로 미미크리 진영(CLIP-KD 2307.12732:
"simple feature mimicry with MSE works surprisingly well")과 로봇 latent-거리 실무가
준거다. 과거 서베이의 "MotionCLIP처럼 L2 정규화" 권고는 이 목적 혼동으로 정리 가능.

### 4.2 증거 요약

**노름 = 정보 (보존 지지):**

- **CLIP 자체 증거**: *Double-Ellipsoid Geometry of CLIP* (2411.14517, ICML 2025) —
  raw 임베딩은 원점 비중심 타원체 셸; 단위구 사영은 정보 손실; 노름이 불확실성 인코딩.
  *CLIP-like Model as Density Ratio Estimator* (2506.22881, v1 제목 기준 공분산 가중
  노름이 정보 이득을 R² 0.98–1.00으로 설명).
- **워드 임베딩 선례**: 노름² = 정보 이득 (2212.09663, EMNLP 2023; softmax 출력층
  일반화 논증 — CLIP 이식은 외삽임을 명시).
- **로봇 실무**: VIP(2210.00030) 공식 코드 = **비정규화 −L2 거리**("value function
  implicitly defined via the embedding distance" — **한 스텝 변위의 크기 = 진행량**);
  R3M(2203.12601) 기본 비정규화 L2 + 노름 패널티 별도; DINO-WM = raw 패치 latent에
  순수 MSE. → ‖Δz‖를 버리면 진행량/속도 정보를 디코더에서 제거하는 셈.
- KD에서 노름 항 추가로 성능 상승 (ND loss 2305.17007 — "large-norm features to be
  more significant").

**반대 진영 (정직 기재):** 타깃 표준화가 안정성 표준 처방인 계열 — data2vec(2202.03555,
붕괴 방지 + 고노름 지배 방지), V-JEPA2-AC(**raw 아님** — LayerNorm 표준화 후 L1,
§0.2 주의 3), L2-softmax(1703.09507, 고노름 쉬운 샘플의 손실 지배 논리).

**contrast_head 제거 (강한 지지):** SimCLR(2002.05709) head 앞 표현이 >10% 우수(대조
손실의 불변화가 head에 흡수); Gupta et al. **2212.11491** (head = InfoNCE 목적함수의
파라메트릭 구성요소); Xue et al.(2403.11391, 층별 특징 가중 이론); Guillotine
Regularization(2206.13378, head 제거로 30%p+ 이득). 핵심 구분: SimCLR에서 head가
이로운 것은 downstream이 *별도 과제*이기 때문 — **우리는 대조 정렬 자체가 downstream**
이므로 head를 두면 접지가 head로 흡수된다. norm-split t2a 9.3% 붕괴는 이 문헌군이
예측하는 실패 모드와 정확히 일치.

**norm-split 기각:** 행동/제어 latent에서 방향·크기 분리 회귀의 직접 선례 없음(인접:
Weight Norm 1602.07868, PoLAR 2506.03133 — 모두 파라미터 공간). 기각은 "선례 없는 추가
구조의 실험적 배제"로 타당.

### 4.3 남은 리스크 + 권장 모니터링 (P7-v2/정렬 리포트에 편입 제안)

1. **노름 아웃라이어의 MSE 지배**: 배치별 ‖Δz‖ 분위수(p50/p95/p99) 히스토그램 드리프트
   + 상위 1% 노름 샘플의 손실 점유율 + grad norm 스파이크 동시 로깅.
2. **작동점 드리프트**: ‖g‖/‖Δz‖(현 0.56)을 학습·태스크별 추적 — 급락 = 방향 정보
   붕괴 신호, 1 초과 = 과적합 신호.
3. **분리 진단**: 손실은 joint 유지하되 cos(g,Δz)와 |‖g‖−‖Δz‖|를 별도 로깅 —
   norm-split을 *구조*가 아닌 *메트릭*으로만 유지.
4. **불안정 시 예비안**: L2 정규화(방향 전용)가 아니라 **V-JEPA2-AC식 parameter-free
   LayerNorm 표준화(양쪽 적용)**가 1순위 — 상대 구조·크기 순서를 보존하며 아웃라이어
   완충 (공식 레시피 [에이전트 검증: config `normalize_reps: true` + `loss_exp: 1.0`]).
5. DINOv2 트랙 한정: mean-patch에 고노름 아티팩트 토큰 유입 가능(2309.16588) — §0.2.

---

## §5. upgrade_ledger 갱신 제안 (실행자 커밋용 구체 수정안)

1. **[정정] V-JEPA2-AC 인용 서술**: "raw 특징 L1" → "parameter-free LayerNorm 표준화
   후 L1 (`normalize_reps: true`, `loss_exp: 1.0` — facebookresearch/vjepa2 공식
   config)". 해당 서술이 등장하는 모든 문서 일괄 정정.
2. **[신규 행] Δz 안정화 예비안**: 현행 "비정규화 Δz 유지" / 후보 "양쪽 LayerNorm 표준화
   (V-JEPA2-AC식)" / 발동 조건 "노름 아웃라이어 손실 점유율 급증 또는 ‖g‖/‖Δz‖ 급락"
   / 근거 2202.03555, 2506.09985.
3. **[갱신] C8 시그모이드 교체 행**: 실행 세부에 "전역 (t,b) 1쌍(앵커별 온도 폐지),
   b₀∈{−10, −log(B−1)} 스윕, **CLIP 트랙 동시 적용 2×2 설계**" 추가. 근거 문헌에
   SAIL 2412.04616, 2509.18552 추가.
4. **[신규 예측 등록 제안]** #9: "C8-sigmoid는 SigLIP2 t2a를 개선한다. CLIP 트랙에도
   +면 '일반 우위', SigLIP2 단독 +면 '계열 매칭'" — 두 시나리오 사전 구분.
5. **[신규 행] DINOv2 register 교차 확인**: 발동 조건 "v2 재실험 <70 또는 P7-v2에서
   패치 노름 아티팩트 검출" / 후보 `dinov2-with-registers-large` / 근거 2309.16588.
6. **[인용 확보] related_competitors.md**: PosA-VLA 2512.03724 §9 (LIBERO에서
   DINOv2→CLIP 교체 독립 보고) — G2 서사 보조 증거.
7. **[지표 확장] H2-fair v3 후속**: 텍스트 공간 통계를 쌍별 cos 중앙값 단독 →
   (Mean Resultant Length, 앵커별 negative 스프레드 IQR, centroid 갭 ‖Δ‖) 3종 병기.

## §6. UNVERIFIED 통합 목록 (후속 확인 대상)

- V-JEPA2-AC 카메라 민감성의 정량 분석(Appendix B.4) — 본문 §4.3만 1차 확인.
- Temporal Trap(2502.03270)의 평가 백본 목록·벤치마크 구성 세부.
- Theia CortexBench 교사별 평가의 풀링/프로토콜 동일성 (CLIP-L vs DINOv2-L 공정성).
- PVM-in-MBRL(2509.12531) 본문, JEPA-WMs(2512.24497) 인코더 절제 세부.
- CI-MSE "Toward Reliable Offline Validation for Robot Manipulation Policies"
  (2606.29898) — 제목만 확인. **오프라인 검증 지표의 2026 최신 시도로 정독 가치 높음**
  (예측 방법론 개정과 직결) — 다음 조사 사이클 후보로 등록 요청.
- 2509.18552의 수렴 (t,b) 값의 SigLIP2 버전 귀속; CAIP(2606.17256)의 대조 손실 계열.
- 2603.17246 (의료 VLE 콘 효과) — 2026-03 프리프린트, 피어리뷰 상태 미확인.
- data2vec 타깃 정규화 문구의 축자 인용 (취지는 다수 출처 일치 — 패러프레이즈로만 인용).
- PosA-VLA §9 인용문 — 에이전트가 v2 PDF에서 축자 추출(본인은 abs 페이지만 재확인).
  실행자 인용 시 PDF §9 직접 확인 1회 권장.

*조사 원본(에이전트 3종 전문)은 요청 시 별도 파일로 제공 가능. 본 문서는 통합·감사판.*

