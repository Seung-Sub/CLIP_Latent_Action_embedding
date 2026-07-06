# 업그레이드 원장 (upgrade ledger) — 컴포넌트별 현행/후보/발동 조건

신설: 2026-07-07 (분석자 S2 예비 지시 §운영). 관리 규칙: 컴포넌트 교체·승격 시 이 표를
갱신하고 발동 조건 충족 근거를 비고에 남긴다. "현행(테스트급)"은 의도적 최소 구현임을
뜻하며 결함이 아니다.

| 컴포넌트 | 현행 (테스트급) | 업그레이드 후보 | 근거 문헌 | 발동 조건 | 상태 |
|---|---|---|---|---|---|
| C8 정렬 손실 | SupCon식 InfoNCE (in-batch, 학습 온도) | SigLIP식 시그모이드 pairwise 손실 | CAIP 선례, SigLIP (Zhai '23) | 배치 확대 필요 시 또는 매트릭스 승자 확정 시 | 대기 |
| 모션 문장 어휘 | 템플릿+패러프레이즈 고정 vocab (motion_lang.json, v2=F2.5 증강) | RT-H식 언어 모션 계층 + 인간 교정 인터페이스 | RT-H (Belkhale '24) | S2 정식 설계 착수 | 대기 |
| flow ctx 인코더 | 결합 MLP (ResidualBlock 스택) | 패치 토큰 attention 구조 (관측 토큰화) | ACT/OpenVLA-OFT 관측 처리 | grasp 정밀 병목 재확인 시. 비고: proprio 인과혼동 재안전화 가설 — 패치 attention이면 proprio 지름길 억제 가능성 | 대기 |
| 디코더 h | 결정론 MLP (상태조건) | 생성 디코드 (조건부 flow/CVAE) | VITA FLD, ACT CVAE 절제(35.3→2%) | 2층 상한(pooled Δz many-to-one)이 계층화(S2) 후에도 병목일 때 | 대기 |
| F2 텍스트→ζ 매퍼 | 2층 MLP prior (5시드) | mini-flow prior (unCLIP prior 완전판) | DALL·E 2 prior (Ramesh '22) | F2.5 후 방향정확도 포화 시 | 대기 |
| 텍스트 인코더 | 앵커 종속 (CLIP text) | 앵커 교체 시 해당 앵커 텍스트 타워 (SigLIP2 등) | — | 앵커 교체와 동시 (분리 금지 — joint 공간 정합) | 대기 |

## 이력

- 2026-07-07: 원장 신설, 6항목 초기 등재.
- (참고) 이미 승격 완료된 컴포넌트: 정책 헤드 mlp→flow matching (캠페인, upgrade_report.md),
  phase1 align dz→hybrid λ0.3 (C8 G5 판정), 평가 20→50롤아웃 paired+Wilson CI+분류기 (S1.v2).
