# 마일스톤 1 보고 — Phase 0 완료 + Phase 1.5 진단(D1–D4) + 50롤아웃 재평가

실행: Claude Code (실행자 세션) / 날짜: 2026-07-06 / 계획서: RESEARCH_PLAN_delta_anchor_v1.1.md
환경: conda `clip_libero` (mujoco 3.3.2 고정), MUJOCO_GL=egl, RTX 5070 Ti
모든 런 JSON: `outputs/report/` (§8 스키마) / 그리드 지표: `outputs/grid/`

---

## 1. Phase 0 잔여 — 전부 완료

| 항목 | 상태 | 산출물 |
|---|---|---|
| (a) 앵커 추상화 | ✅ | `src/core/anchor.py` (Clip/SigLIP2/DINOv2 + get_anchor), policy.py LATENT 제거→anchor.dim, 캐시 키 `{anchor_id}/{proj}/{norm}` 분리. 기본 앵커 출력 = 기존 ClipWrapper와 완전 동일(수치 검증), 양 트랙 스모크 통과, ALOHA GT 평가 MAE 0.28° 재현 |
| (b) 50롤아웃 승급 | ✅ | rollout_sim.py: paired 공식 init_states, wait 10스텝(OpenVLA 관행), Wilson CI, 실패태깅(reach/grasp 자동, wrong_* 수동 대상), 실패영상 2편/태스크, §8 JSON (data_variant·max_steps 필수) |
| (c) k-NN 바닥선 | ✅ | `src/eval_libero/knn_baseline.py` (태스크별 데모 DB, cos k=5 가중 청크 재생) |
| (d) 데이터 판본 2종 | ✅ | raw(보유) + openvla_modified 재현 생성(`d2_rerender.py --mode openvla_replay`, 공식 규칙: 재생 성공만+no-op ε=1e-4+256²) → `data/libero_openvla_mod/` |
| 확인 필요 3종 | ✅ | `verification_notes.md` — 수정판=RLDS만 공식(HDF5는 재생성 스크립트), no-op 규칙 원문, spatial max_steps=220+wait10/50트라이얼, V-JEPA2 `facebook/vjepa2-vitl-fpc64-256` 등(프레임 반복=이미지 모드 공식) |

## 2. 폐루프 수치 (신 프로토콜: 50롤아웃/task, paired, wait10, max_steps 300)

| 런 | suite SR | Wilson CI | 비고 |
|---|---|---|---|
| **현행 정책 재평가** `p0_reeval_libero_spatial_raw_s0` | **35.6%** (178/500) | 31.5–39.9% | 구프로토콜 36%와 일치 → 베이스라인 고정 |
| **k-NN 바닥선** `p0_knn5_libero_spatial_raw` | **18.2%** (91/500) | 15.1–21.8% | 표현만의 성능. 정책(f+DeltaAE)의 순기여 ≈ +17%p |

- 태스크별(정책): [40,14,36,86,18,30,24,8,66,34]% / (kNN): [8,18,12,52,10,6,14,26,26,10]%
- 실패 모드: 정책 실패 322건 전부 'grasp' 태깅(그리퍼 닫기 명령은 항상 발생) — **도달은 되나 파지·배치 정밀도가 병목**. wrong_object/goal 구분은 실패 영상(태스크당 2편 저장됨) 수동 태깅 필요.
- 공표 비교 시 명기: 우리 max_steps=300(고정) vs OpenVLA 관행 220+wait10. data_variant=raw.

## 3. Phase 1.5 진단 — 핵심 결론: "R² 붕괴의 주범은 인간 데모 다봉성" (D4)

R² 레짐: ALOHA 0.988 vs LIBERO 0.682 (동일 구조). 가설별 절제 결과:

| 진단 | 조작 | dec R² | 판정 |
|---|---|---|---|
| 기준선 | raw, 128², chunk 0.8s | 0.6821 | — |
| **D1** no-op 필터 | ε=1e-4 / 0.05 | 0.6811 / 0.6828 | **기각** (raw 사본에 no-op 0.12%뿐, 히스토그램 단봉·중앙값 0.82) |
| **D2** 해상도 | 256² 상태주입 재렌더 | 0.6935 | 경미(+0.011) |
| **D3** 청크 스케일 | 0.4s / 1.6s | 0.6933 / 0.6197 | 트레이드오프: 복원↔맵핑 역방향 (긴 청크=align 0.73·retr 33.7%↑, 복원↓) |
| D2×D3 | 256²+1.6s | 0.6263 | 시너지 없음 |
| G0확장: pre-projection Δz | 1024d | **0.7015** | 최고치 — projection 정보손실 존재 |
| G0확장: 비정규화 Δz | norm off | 0.6841 | R² 동급, **retrieval 최고 35.1%** — Δz 크기 정보 유효 |
| 수정판 재현 | 실패12.4%제거+no-op+256² | 0.6683 | 수정판 가설 종결 (개선 없음) |

**D4 다봉성 정량화** (`d4_multimodality.py`): z-이웃(k=10, 타 에피소드) 조건부 액션 분산
- LIBERO **E[Var(A|z)]=0.410** vs ALOHA 0.135 → **3.0배**
- z_t 조건 결정론 상한 추정: LIBERO ≈0.59 vs ALOHA ≈0.87 (보수적 하계 — 실측 R²가 높은 건 디코더가 실제 Δz(미래 정보 포함)를 조건으로 받기 때문)

### 게이트 G0 판정: **FAIL** (최고 0.70 < 0.85)

D1–D3 전 조합 + 계획서 지시 확장(Δz 재정의 2종)까지 소진. D4 수치와 결합한 실행자 소견:
- 0.85 미달은 데이터·표현 조작의 문제가 아니라 **결정론 디코더 h의 구조적 상한** (인간 데모 = 다봉).
- 계획서 §5 4.3(flow/CVAE 디코더, 이미 "보류 해제"로 명시됨)이 다음의 구조적 해법 — ACT 절제에서 CVAE 제거 시 35.3%→2% 붕괴 사례와 정합.
- **분석자 검토 요청**: G0을 "결정론 R² 절대 0.85" 대신 (i) D4 상한 대비 달성률, 또는 (ii) 폐루프 SR 게이트로 재정의할 것을 제안. 현행 기준으로는 Phase 2 매트릭스 진입이 영구 봉쇄되나, 다봉성은 앵커 선택과 독립인 데이터 속성이므로 "동일 디코더 조건 하 앵커 비교"는 여전히 유효한 실험임.

## 4. 부수 발견 (분석자 참고)

1. **공식 데모 12.4%가 재생 실패** (62/500, 태스크별 1~13개) — env 재생 비결정성 또는 원 데모 품질. OpenVLA 수정판과의 비교표 작성 시 주의.
2. 우리 raw 사본(pypi `libero` 재배포판)은 no-op가 거의 없음 — OpenVLA 논문의 no-op 필터 서사가 우리 데이터엔 비적용. 필터 효과는 그들의 원본 사본 기준일 가능성.
3. 청크 시간창↑ → Δz-액션 정렬↑ (P2 SNR 가설 선행 증거). Phase 1 프로브 P2 설계에 반영 권장.
4. k-NN 18.2%는 태스크3(52%)에서 특히 높음 — 해당 태스크는 표현만으로도 절반 해결되는 "쉬운" 분포.

## 5. 프로토콜·재현 정보

- 모든 phase1 비교는 seed 0, 동일 val split(에피소드 순서 불변) = paired.
- 체크포인트: `checkpoints/grid/<tag>.pt` (본 체크포인트 미오염). 지표: `outputs/grid/<tag>.json`.
- 실패·기각 런 포함 전 결과 보고(본 문서 + JSON). 실패 영상: `outputs/eval/videos/`.
- blocker 없음. 범위 외 리팩토링 없음 (공용층 변경 = 계획 명시 항목만, 양 트랙 스모크 통과).
- 코드 변경 미커밋 상태 — 커밋 정책 지시 대기.

## 6. 다음 단계 제안 (분석자 결정 대기)

1. **G0 게이트 재정의 여부** (§3 소견) — 결정 전 Phase 2 매트릭스 착수 금지 준수 중.
2. flow matching 디코더 최소구현(계획 §5 4.3, VITA FLD 참조)을 Phase 1.5 후속으로 앞당길지.
3. Phase 1 프로브 5종(P1–P5)은 게이트와 독립적으로 진행 가능 — 지시 시 즉시 착수.
