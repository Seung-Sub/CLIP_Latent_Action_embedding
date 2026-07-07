# 숫자 카드 — 발표 인용 수치 전량 (오인용 방지)  [자동 생성]

| 수치 | 값 | CI | 프로토콜 | run_id/출처 |
|---|---|---|---|---|
| C8 HY03 폐루프 (seed2) | **87.0%** (SR@220 86.4) | [83.8, 89.7] | 50/task paired wait10 max300 | c8_closedloop_hy03_spatial_raw_s0 |
| C8 DZ 폐루프 (seed2) | **87.0%** (SR@220 86.2) | [83.8, 89.7] | 50/task paired wait10 max300 | c8_closedloop_dz_spatial_raw_s0 |
| 대표 수치: s1 500롤 (확정 레시피) | **81.0%** (SR@220 80.4) | [77.3, 84.2] | 50/task paired wait10 max300 | final_hy03_unnorm_spatial_raw_s1 |
| DINOv2-DZ 폐루프 | **65.2%** (SR@220 63.4) | [60.9, 69.2] | 50/task paired wait10 max300 | s2p_dinov2_dz_spatial_raw_s2 |
| k-NN 바닥선 | **18.2%** (SR@220 18.0) | [15.1, 21.8] | 50/task paired wait10 max300 | p0_knn5_libero_spatial_raw |
| mlp 기준선 재평가 | **35.6%** (SR@220 34.2) | [31.5, 39.9] | 50/task paired wait10 max300 | p0_reeval_libero_spatial_raw_s0 |
| proprio 9D (기각) | **59.0%** (SR@220 56.8) | [54.6, 63.2] | 50/task paired wait10 max300 | s1v2_dz_proprio_spatial_raw_s2 |
| Phase1 R² 레짐 | ALOHA 0.988 vs LIBERO 0.682 | — | seed0 오프라인 | grid/*.json |
| D4 상한 사다리 | 0.590 → 0.698 → 0.746 | — | kNN k=10 | d4_refined_conditions.json |
| F2.5 zero-shot (HY03) | 58.3%±5.3 | 5시드 | holdout 20문장 | c8_gapfix_f25.json |
| G2 마진 | +21.8pp (CLIP vs DINOv2 폐루프) | CI 비겹침 | 동일 프로토콜 | matrix_closedloop_report.md |

주의: 태스크별 수치는 fp16 비결정성으로 ±30pp 반복 변동 — suite 평균만 인용 (docs/eval_protocol.md).