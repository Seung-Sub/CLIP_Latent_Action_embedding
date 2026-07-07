# 숫자 위생 — 폐루프 수치 단일 대조표 (S1.v2 §6)

작성: 2026-07-07. 모든 수치 LIBERO-Spatial. **프로토콜 열 없이 수치만 인용 금지.**

| 런 | SR | SR@220 | n/task | init_states | wait | max_steps | seed | data | 정책 | phase1 | 출처 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 초기 20롤아웃 (구 프로토콜) | 36.0% | — | 20 | 공식, ep%n 순환 | 5 | 300 | 0 | raw | mlp d1024+lang | dz(정규화) | outputs/eval/rollout_libero_spatial.txt (2026-07-06 재현 런) |
| 기준선 재평가 (paired 승급) | **35.6%** [31.5–39.9] | 34.2% | 50 | 공식, 순서 고정 | 10 | 300 | 0 | raw | mlp d1024+lang | dz | §8 p0_reeval_libero_spatial_raw_s0 |
| k-NN 바닥선 (VINN식) | 18.2% [15.1–21.8] | 18.0% | 50 | 공식, 순서 고정 | 10 | 300 | — | raw | 무학습 k=5 | (CLIP z만) | §8 p0_knn5_libero_spatial_raw |
| 캠페인 mlp 기준 (병렬 연구자) | 36.5% | — | 미상* | 미상* | 미상* | 미상* | 3시드 평균 | raw | mlp | dz | README (docs/upgrade_report.md 이 머신 부재*) |
| 캠페인 flow SOTA (병렬 연구자) | 80.0% (Obj 85.7 / Goal 80.0) | — | 미상* | 미상* | 미상* | 미상* | 3시드 평균 | raw | flow+wrist d1536+lang | dz | README* |
| **C8-DZ 폐루프 (본 세션)** | **87.0%** [83.8–89.7] | **79.8%** | 50 | 공식, 순서 고정 | 10 | 300 | **2 (승자 시드)** | raw | flow+wrist d1536+lang | c8_arm_dz(dz) | §8 c8_closedloop_dz_spatial_raw_s0 |
| (참고) 공표 외부 기준 | DP 78.3 / OpenVLA 84.7 / OFT 97.6 / π0 96.8–98.0 | (그들 horizon=220+wait10) | 50×3시드 | 공식 | 10 | **220** | 3시드 | **openvla_modified** (no-op·실패 필터+256²) | 각 논문 | verification_notes.md §2 |

\* docs/는 .gitignore라 이 머신에 없음 — 캠페인 수치는 README·config 주석 기재값만 확보.
§8 JSON 소급 등재는 프로토콜 필드를 "unknown"으로 명기한 스텁으로만 가능해 보류
(불명 필드를 채워 넣는 것이 오히려 위생 위반). **분석자/병렬 연구자에게
upgrade_report.md 공유 요청** — 수신 시 즉시 §8 소급.

핵심 주의 3건:
1. 우리 SR(max 300) vs 공표(220+wait10) — **SR@220 열이 공정 비교치**.
2. 공표 OpenVLA 계열은 **수정판 데이터** 기준 (우리 raw와 데이터 판본 상이). 우리
   수정판 재현 실험은 R² 기준 무이득 확인(phase1_5_diagnosis.json) — 폐루프 판본
   비교는 미실시.
3. 시드: C8 폐루프는 시드2(승자) 단일 — 3시드 완주는 최종 동결 구성에만 (규율 §3).
