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
| 텍스트 인코더 | 앵커 종속 (CLIP text) + 교차 앵커 lang_proj 어댑터(무텍스트 앵커용, L3-lite) | 앵커 교체 시 해당 앵커 텍스트 타워 (SigLIP2 등) | — | 앵커 교체와 동시 (분리 금지 — joint 공간 정합) | lang_proj 구현됨 (2026-07-07) |
| 앵커 백본 (신규 후보) | CLIP/SigLIP2/DINOv2 3종 매트릭스 | **PE 계열** — PE-core(제로샷 SigLIP2 상회)/PE-spatial(dense)/PE-AV(오디오-비디오-텍스트 단일 공간, 시그모이드 contrastive) | Meta Perception Encoder ('25) | 오디오 모달 단계 개시 or SigLIP2-HY03 폐루프 실망. HF/repo 통합 비용은 구현 시점 검증 | 대기 |
| 앵커 백본 (융합 경로) | 단일 백본 선택 | **RADIO** — CLIP+DINOv2+SAM 다중 교사 증류 단일 백본 (텍스트 접지 보존) = "앵커 수준 융합"의 구조 정합 경로 | NVIDIA RADIO (Ranzinger '24) | H2가 "코딩=DINO, 언어=SigLIP2" 교착일 때 (현재 매트릭스 1차가 정확히 이 구도 — 폐루프 결과 대기) | 대기 |
| 폐루프 평가 | fp16 인코딩 (비결정, 태스크별 ±30pp) | **결정론 평가 모드** — fp32 인코딩 or torch deterministic 플래그 | 캠페인 journal 실측 | 근소 차이 판정(<3pp 반복 규칙)이 반복되어 비용 정당화될 때 | 미적용 |
| 디코드 스케일 | ‖g‖/‖Δz‖ ≈ 0.56 수축 매니폴드 (자기일관 동작) | **디코드 노름 재보정** — 스케일 캘리브레이션 | 정렬 리포트 실측 (bridge 0.56) | 폐루프가 크기 과소 예측을 병목으로 지목할 때 | 미적용 |
| 손목 모달리티 | 손목캠 z_t 토큰 (정책만) | **손목 Δz 제2 스트림 + 그리퍼 채널 분리 + (미래) 손목축 tactile 바인딩** 통합 트랙 — 손목캠 자기운동 변위 = EE 운동의 시각 프록시, 그리퍼 준이산 이벤트 분리, 캠페인 잔여 레버("phase1 g/h 손목캠 조건")와 합류 | 캠페인 journal | S2/패치 토큰 후 grasp 잔존 시 | 대기 |
| S2 상위층 분절 | 고정창 (1.6–3.2s 서브골 Δz ← 태스크 지시문 정렬, S2 승인 구조) | 가변 길이 의미 분절 (그리퍼 이벤트·속도 기반) + 분절 라벨링 | — | S2 정식 스펙 작성 시 대안 트랙으로 검토 | 대기 |

## 정렬의 의미론 (분석자 표준 노트, 2026-07-07)

phase1의 align 항은 **접지 정규화(grounding regularizer)**다 — per-sample 완전 일치가
목적이 아니다. 완전 정렬(g ≡ Δz)은 recon과 상충한다(VITA 절제 근거): Δz에는 액션과
무관한 렌더·조명 성분이 있고, g가 그걸 모두 좇으면 액션 복원이 무너진다.
따라서 **정렬 상태의 공식 뷰는 "정렬 리포트" 그림**(per-sample cos 히스토그램 + 노름비
+ retrieval)이며, 필드 표준 지표도 상대(retrieval) 정렬이다. PCA 산점의 시각 인상
(화살표 불일치 등)으로 정렬 실패를 진단하지 말 것 — 소노름 Δz는 전역 분산에 묻힌다
(→ latent_mapping 접선 투영 뷰).
**g-매니폴드 의미론**: 정책의 잠재 타깃은 Δz가 아니라 g(A_fut, z)이고 h는 cycle 손실로
g 분포에서 디코딩을 학습한다 — 시스템은 Δz에 느슨히 접지된 **수축 매니폴드(‖g‖≈0.56‖Δz‖)
에서 자기일관적으로 동작**한다. 노름비 0.56은 결함이 아니라 동작점이다.

## 이력

- 2026-07-07: 원장 신설, 6항목 초기 등재.
- 2026-07-07: PE 계열·RADIO·S2 가변 분절 등재, lang_proj 어댑터 구현 기록, 정렬 의미론 노트.
- (참고) 이미 승격 완료된 컴포넌트: 정책 헤드 mlp→flow matching (캠페인, upgrade_report.md),
  phase1 align dz→hybrid λ0.3 (C8 G5 판정), 평가 20→50롤아웃 paired+Wilson CI+분류기 (S1.v2).
