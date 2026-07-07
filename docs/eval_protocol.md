# 폐루프 평가 프로토콜 (공식) — v1.1 (2026-07-08 개정)

기준: 계획서 v1.1 §0.2 + S1.v2 §1 + 분석자 프로토콜 개정 (07-08).

## 표준 절차

- LIBERO: 태스크당 50롤아웃, 공식 `get_task_init_states` 순서 고정(=paired), wait 10스텝
  (dummy `[0]*6+[-1]`, max_steps 불포함), max_steps 300 (공표 비교 시 SR@220 병기),
  Wilson 95% CI, 상태 기반 실패 분류(reach/grasp/wrong_object/wrong_goal),
  §8 JSON (data_variant·max_steps·seed 필수), 실패 영상 2편/task.
- 시드/모델 선택 (journal §4 반영, 07-08 개정): val 손실 아닌 **소량 폐루프 스크리닝
  (10 eps/task)**으로 시드 선별 → 승자만 전량(500) 평가.

## 비결정성 주의 (07-08 신설)

- 폐루프는 fp16 CLIP 인코딩 등 비결정성으로 **태스크별 수치가 반복 런 간 ±30pp까지 변동**
  (캠페인 journal 실측). **태스크별 수치는 참고치이며, 공식 수치는 suite 평균(n≥500)만**.
- **근소 차이 판정 규칙**: 조건 간 suite SR 차 **< 3pp**이면 반복 런 1회 추가 후 평균으로 판정.
- 기왕 보고 정정: "DZ = HY03 87.0 동률"(c8_g5_verdict.md)은 **"반복 잡음 내 동률"**로 해석.
  G5 판정(HY03 승격)은 폐루프 무손실 + 언어 축 이득이 근거이므로 영향 없음.
- 결정론 평가 모드(fp32/deterministic 플래그)는 ledger 등재 — 근소 차이 판정이 반복될 때 발동.

## 정렬 수치 표기 규약 (07-08 신설)

모든 정렬 수치에 (i) Δz 정의: 정규화 여부·projection(joint/pre), (ii) 집계: 평균/중앙값을
병기한다. 선례 각주: 학습 로그의 "align_cos 0.65~0.67"은 **정규화-Δz 공간의 평균** cos,
정렬 리포트의 "median 0.316"은 **비정규화-Δz 공간의 중앙값** — 정의가 달라 직접 비교 불가.

## H2-fair v2 (07-08 비준 병합)

**앵커 비교는 문헌 관행 구성으로만 승격**: 각 앵커의 전처리(resize/crop)·풀링(CLS/patch)·
정규화·온도가 해당 앵커의 공식/문헌 관행과 일치함을 verification_log에 기록한 뒤에만
비교표 승격. 위반 사례 실증 2건: 모션 어휘 CLIP 편향(→v3 dual-score), DINOv2 crop·CLS
(→v2 no-crop+clsmp 재실험). 각주 의무: CLIP이 무손상이었던 이유 = 정사각 입력에서
224 resize→224 crop이 무연산이라 crop 이슈가 발현되지 않음 (우연이지 설계 아님).
