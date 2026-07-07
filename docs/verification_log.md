# 검증 이중화 로그 (verification log)

신설: 2026-07-08 (사용자 지시: 구현·평가 이중 검증 상설화). 사이클마다 검증 에이전트
발견 → 조치 → 영향 런 판정을 기록한다.

## 사이클 1 (2026-07-08) — 절제 배치·C7 투입 직후

### 발견 및 조치

| # | 발견 (출처) | 심각도 | 조치 | 영향 런 판정 |
|---|---|---|---|---|
| 1 | **train↔rollout 토큰 순서 불일치**: 학습 [..., proprio, obs2] vs 롤아웃 [..., obs2, proprio] (자체 교차점검, 리뷰 에이전트 병행 의뢰 중) | CRITICAL (잠재) | rollout_sim 순서 수정 | 기존 런 무영향 (proprio+obs2 동시 사용 런 없음) |
| 2 | **DINOv2 center-crop**: 기본 processor가 256 resize→224 crop → 시뮬 렌더 테두리 12.5% 삭제. 로봇 문헌 관행 = 224 직접 resize (앵커 조사 에이전트, DINO-WM 등 인용) | HIGH | Dinov2Anchor 기본 no-crop + 캐시 키 분리(dinov2-large-nc) | **기존 DINOv2 런 전부 crop판** — m_dinov2_unnorm_p1(매트릭스 1차), DINOv2-DZ 폐루프 65.2, P7 이탈도, D5. 결과표에 "crop판" 명기, v2 재실험 발진 |
| 3 | **DINOv2 CLS는 dynamics에 부적합**: DINO-WM 절제에서 CLS 단독은 patch 대비 유의 저하 (Chamfer 0.26→0.79). 우리 DZ는 CLS(pooler) 사용 (앵커 조사) | **HIGH — H2 판정 위협** | pooled=clsmp(CLS⊕patch-mean, 2048d) 옵션 추가, **h2f_dinov2_v2_clsmp_p1 재실험 발진** | **G2(+21.8pp) 판정은 "CLS·crop판 DINOv2 조건부"로 격하** — v2 결과 도착까지 논문 주장 보류 |
| 4 | SigLIP2 InfoNCE 정규화 (앵커 조사 권고) | — | **점검 결과 이상 없음**: DeltaAE.info_nce가 양쪽 F.normalize 수행, 신규 온도 학습(0.07 init) = 권고와 일치, logit_scale/bias 미이식 = 정답 | 무영향 |
| 5 | SigLIP2 소문자화 (앵커 조사) | — | 점검: 문장이 항상 Siglip2 processor 경유 (백엔드 소문자화) — 이상 없음 | 무영향 |
| 6 | dinov2-large(무레지스터) 고노름 아티팩트 토큰이 patch-mean 오염 가능 (registers 논문) | MED | ledger 등재: with-registers 체크포인트 후보 (슬라이스 [:, 5:] 주의) — v2 결과 후 검토 | mp 캐시에 잠재 노이즈 (진행) |
| 7 | HF pooler_output = last_hidden[:,0] (trained pooler 아님) — 우리 두 접근 경로 일치 확인 | — | 확인만 (버그 아님) | 무영향 |

### 신규 실험 (검증발 재실험)

- `h2f_dinov2_v2_clsmp_p1`: DINOv2 no-crop + CLS⊕patch-mean — **H2 공정성 v2**.
  오프라인에서 crop·CLS판 대비 개선 폭 측정 → 폐루프 재진출 여부는 분석자 판정.
- C7 mp 캐시 no-crop 재생성 → P6 → 융합 phase2 (GPU9 체인 재발진).

### 방법론 노트

crop·CLS 이슈는 "CLIP-first 코드베이스의 앵커 편향" 우려(사용자)가 코드 수준에서
실재했음을 보여줌 — H2-fair(어휘 편향)에 이어 두 번째 실증. **모든 앵커 비교 주장은
해당 앵커의 문헌 관행 구성으로 재확인 후에만 승격**을 표준 규칙으로 채택.

### 리뷰 에이전트 확정 발견 (사이클 1 계속)

| # | 발견 | 심각도 | 조치 | 영향 런 |
|---|---|---|---|---|
| 8 | **strided 인코더가 마지막 3스텝에 blind** (stride-2에 잘못된 causal pad — grad=0 실증) | CRITICAL | pad (k−s) 수정 + 전 스텝 grad 검증 | **abl_enc_strided 무효 → 수정판 재실행 필요** |
| 9 | load_models n_tok에 obs2 누락 → C7 폐루프가 로드 시점 크래시 (size mismatch 실증) | HIGH | n_tok에 obs2 포함 | 크래시 전 수정 (silent 아님) |
| 10 | comp × transformer/mlp 조합 크래시 (pos embed 길이 불일치 실증) | HIGH | 가드 추가 (명시 에러) | 미발동 조합 — 그리드 확장 시 예방 |
| 11 | obs2 비표준화 → 공유 LayerNorm 지배 (per-elem std 0.75 vs 0.036 = ~20×) → C7 "무익" 오판 위험. P6 프로브는 표준화했으므로 프로브-정책 불일치 | MED | o_mean/o_std 표준화 + ckpt 저장 + 롤아웃 적용 | GPU9 phase2 시작 전 수정 동기화 완료 |
| 12 | quantile 축퇴 가드 부재 (현 데이터는 무사 — "data-luck") / comp·vel × DCT 의미론 무의미 (휴면) | MED | 문서화 (발동 시 가드) | 무영향 |
| 13 | 토큰 순서·obs2 정렬·전처리 일관성·p6 정렬·comp 리샘플 시맨틱: **이상 없음 확정** (리뷰 검증) | — | — | — |

## 사이클 2 (2026-07-07~08) — Cowork(이론 검증 파트너) 온보딩 회신 반영

| # | 발견 (cowork) | 조치 |
|---|---|---|
| 14 | [차단급] repo main에 정본 문서 미푸시 (NUMBER_CARD·verification_log·§8 JSON·motion_lang_v3 등) — gitignore가 docs/·outputs/ 전체 차단 | .gitignore 예외 패턴(/dir/* + !재포함) 도입, 79파일 스테이징·푸시 (영상 제외) |
| 15 | V-JEPA2-AC "raw L1" 서술 부정확 — 공식 config는 **LayerNorm 표준화 후 L1** (`normalize_reps: true`) | ledger Δz 안정화 행에 정정 반영, 예비안 1순위를 "양쪽 LayerNorm 표준화"로 교체 |
| 16 | 시그모이드 교체는 "계열 매칭"이 아니라 "일반 우위"일 가능성 (SAIL: InfoNCE→sigmoid가 교차 계열에서도 승리) | **2×2 설계(backbone × loss) 사전 등록** — 예측 #9 등록, 분석자 승인 대기 |
| 17 | dinov2-large 무레지스터 → mean-patch 고노름 아티팩트 유입 가능 | P7-v2에 패치 노름 p95/p99 모니터링 추가 예정, register 교차 확인 ledger 행 신설 |
| 18 | PosA-VLA 2512.03724 §9: LIBERO에서 DINOv2→CLIP 교체 독립 보고 | related_competitors 추가 (보조 증거, PDF 재확인 1회 필요 표기) |
| 19 | 비정규화 Δz 레시피: 문헌 방어 강함 (VIP/R3M/DINO-WM 정합; norm-split 기각도 SimCLR 문헌군 정합) | 결정 유지·방어 근거 확보. 모니터링 3종(노름 분위수 드리프트, ‖g‖/‖Δz‖ 추적, cos/노름차 분리 로깅)을 정렬 리포트에 편입 예정 |
| 20 | 폐루프 예측 실패 3건의 구조 진단 (범주 오류·조절변수 누락·구간 과신) + 개선안 (LAOM 할인계수, 조건부 등록, 서수 예측) | 예측 방법론에 반영 제안 — 분석자 판단 대상 |

### 사이클 2 후속 — H2-fair 해상도 통제 마감 (2026-07-08)

SigLIP2-so400m **224** (patch14, v3 어휘, HY03): dec 0.690 / t2a **39.9%** @5 56.6
vs **384판**: dec 0.690 / t2a 40.8 — **사실상 동일**. 해상도(128→384 업샘플 3배)는
SigLIP2 언어 축 격차의 교란 변수가 **아님**을 확정. 잔여 격차(vs CLIP 52.7)는
목적함수 축 (cowork 재해석: "sigmoid 일반 우위" 가설 포함 — 2×2로 분리 예정).
H2-fair §3 항목 마감.
