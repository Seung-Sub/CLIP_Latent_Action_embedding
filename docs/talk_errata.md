# 발표자료 정정표 (talk errata) — Action_Latent_ver2 대비 실측 갱신 항목

생성: 2026-07-08 (분석자 §17). 팀미팅 자료의 계획·예상 항목 중 실측으로 대체된 것.

| 자료 내 항목 | 실측 갱신 | 근거 run_id |
|---|---|---|
| G4 예상: "DINOv2가 dynamics에 유리" | **역전 실측**: DINOv2-DZ 폐루프 65.2 vs CLIP 87.0 (+21.8pp CLIP), wrong_object 실패 신규 등장 (17건) | s2p_dinov2_dz_spatial_raw_s2 |
| L0(언어 축) 미측정 | t2a 62.1%(HY03, v2) / 52.7%(v3 중립) / zero-shot 58.3±5.3 (F2.5) | c8_arm_hy03, h2f_clip_hy03_v3, c8_gapfix_f25 |
| proprio 토큰 계획 | **기각**: full −28.0 / dropout −13.2 / gripper-2D −19.4pp (인과 혼동, P6 실증) | s1v2_dz_proprio, s2p_proprio_a1/a2 |
| DCT 청크 표현 후보 | **보류 판정** (LIBERO 무기여 — 캠페인 실측, ledger) | docs/upgrade_ledger.md |
| Δz 정규화 권고 | **비정규화 채택** (검색 +7.5pp, zero-shot 분산 1/6; 노름분리 변형은 기각 t2a 9.3) | bridge_hy03_unnorm_p1, recipe_normsplit_p1 |
| 앵커 비교 = 오프라인 지표 기준 | 오프라인↔폐루프 **역전 2례** (proprio, DINOv2) — 예측은 코딩+강건성 축 분리로 개정 | matrix_final_table.md |
| 모션 어휘 = 단일(CLIP) 기준 | **CLIP 편향 실측** (변별력 2배) → dual-score v3 채택, SigLIP2 격차의 ~45% 설명 | h2fair_checks.json, h2f_*_v3 |
