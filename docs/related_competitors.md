# Related Competitors: LARA, JALA, VLM-LAM, A2A (요약판 — 전문은 에이전트 조사 원문)

검증일 2026-07-08. **4개 arXiv ID 전부 실존 확인** (fictional 아님).

## 차별화 핵심 (분석자 반박 프레임 + 실측 근거)

| | 우리 | LARA 2606.07100 | JALA 2602.21736 (CVPR26) | VLM-LAM 2601.22714 | A2A 2602.07322 |
|---|---|---|---|---|---|
| 액션 표현 공간 | **frozen CLIP Δz (감독 접지)** | 학습 LAM(IDM+VQ128)+VLA 공동학습 | 학습 latent(Perceiver+GRVQ4096) | 무감독 LAM (VLM 임베딩=FDM 타깃) | past-action 임베딩 = flow **source** |
| 양자화 | 없음 (연속) | VQ | GRVQ | 명시적 생략 | 없음 |
| 언어 정렬 액션공간 | **있음** (InfoNCE 모션문장) | 간접 | 간접 | promptable VLM 경유 | 없음 |

- **LARA 반박 성립**: LARA의 비판 대상은 "정확한 액션 궤적에 미접지된 frozen LAM 의사라벨" — 우리 g는 실제 액션청크로 **완전 감독 접지**되므로 비판 범위 밖. C8 절제·ARM-AE 대조군·DINOv2 역전이 실증.
- **VLM-LAM의 추가 무기**: 그들은 CLIP/DINOv2를 FDM 타깃 최악으로 보고(오프라인 LAM 품질 기준) — 우리 폐루프 +21.8pp CLIP 승과 결합하면 **"오프라인↔폐루프 지표 역전" 논증의 독립 보강 증거**.
- **JALA와의 분리**: 앵커가 학습 중 움직임(우리는 frozen) + 언어 정렬 변수 미분리 (우리 매트릭스가 그 변수를 분리).
- **A2A 실존 확정**: "Action-to-Action Flow Matching" Jia et al., arXiv **2602.07322** — flow_source=past의 정식 인용 확정 (코드 주석 갱신 완료). "ACT2ACT"는 오기 (2019 HRI 논문과 무관). A2A는 source 측 수정, 우리는 target/code 측 — 직교·조합 가능.

주의: 세부 수치(LARA 5–15%, JALA 96.9% 등)는 HTML 자동 요약 경유 — 논문 인용 전 PDF 재확인 (UNVERIFIED at figure level). A2A 벤치마크 구성 미확인.

## 추가 (cowork 조사, 2026-07-07): PosA-VLA 2512.03724 — G2 보조 증거

§9에서 LIBERO 시뮬 적응 중 *"DINOv2 features underperform on Libero's synthetic
renderings"*라며 CLIP으로 교체했다는 **독립 보고** (동일 벤치마크·flow matching 계열).
G2 서사의 외적 타당성 보강. 단 정량 격차 미보고(관측 진술만) → 보조 증거로만 인용.
[주의: 인용문은 에이전트가 v2 PDF에서 추출 — 논문 인용 전 PDF §9 직접 확인 1회 필요]
