# 발표 자산 인덱스 (INDEX) — 그림별 데이터 출처·캡션 초안

생성: 2026-07-08 / 생성 스크립트: src/diagnosis/make_presentation.py (FIG1~6,8) + 인라인 (FIG7,9,10)
수치 인용은 반드시 NUMBER_CARD.md 경유 (CI·프로토콜·run_id 포함).

| 자산 | 데이터 출처 | 캡션 초안 |
|---|---|---|
| FIG1_timeline.png | README 캠페인 기재값 + §8 JSON (c8_closedloop_hy03, final_hy03_s1) | 정책 헤드·관측·정렬 개선의 누적 궤적. 좌측 3점은 캠페인 프로토콜(20롤 비paired), 우측 2점은 paired 50롤 — 주석 없이 직접 비교 금지 |
| FIG2_g5_arms.png | c8_closedloop_{dz,da,hy01,hy03}_*.json + c8_gapfix_f2.json | C8 정렬 절제 4팔. 교차 패턴(폐루프는 Δz-접지, 언어 축은 직접 정렬 우세)이 HY03 승격의 근거 |
| FIG3_inversion.png | outputs/grid/*.json + 폐루프 §8 JSON 8건 | 오프라인↔폐루프 역전 2례(DINOv2, proprio). 오프라인 코딩 지표는 폐루프 예측 불충분 → P7 강건성 축 분리 근거 |
| FIG4_p2_sweep.png | grid/{p2,d3,d1}_libero_chunk*.json | P2 시간창 스윕 5점. 맵핑·검색 단조↑ vs 복원 단조↓, 0.8s = 제어 동작점. S2 계층 설계 근거 |
| FIG5_d4_ladder.png | d4_refined_conditions.json + 폐루프 실측 | 조건부 정보 사다리와 폐루프 효과의 분리 — "정보량 ≠ 인과적 유용성" (proprio 기각 서사) |
| FIG6_proprio.png | c8_closedloop_dz + s1v2_dz_proprio + s2p_proprio_{a1,a2} JSON | proprio 4조건 판정 + 실패 모드 스택. 상태 기반 분류기 데이터 (wrong_goal = 관성 진행) |
| FIG7_alignment_panels.png | alignment_{m_clip,bridge_hy03}_unnorm_p1.png | 정렬 리포트 공식 뷰 (비정규화 Δz·중앙값 표기). DZ vs HY03 — per-sample 정렬↔언어 접지 트레이드 |
| FIG8_matrix.png | grid/m_*.json, m2_*.json + 폐루프 JSON | 앵커 매트릭스 (CLIP-pre 각주 = 공정 비교 의무). DINOv2 오프라인 압도 vs 폐루프 역전 |
| FIG9_baselines.png | baseline_comparison.md | 프로토콜 열 포함 단일 대조표 (캠페인 열은 journal 도달 시 확정) |
| FIG10_pipeline.png | 구조 도식 (데이터 무관) | frozen 앵커 잠재공간 위 Δz-접지 flow 정책, 토큰 5개, receding 8 |
| NUMBER_CARD.md | §8 JSON 자동 집계 | 발표 인용 수치 전량 + CI + run_id (s1 500롤 완료 시 재생성으로 자동 갱신) |
| videos/a_HY03_success_* | s1 확정 레시피 런 성공 3편 | 대표 성공 시연 |
| videos/b_DINOv2_fail_* vs b_CLIPDZ_success_* | mtime 귀속 아카이브 (videos_p7/) | 동일 벤치 DINOv2 실패 vs CLIP 성공 대비 |
| videos/c_proprio_wronggoal_example.mp4 | a1/a2 시절 아카이브 | 파지 후 시각 검증 없는 관성 진행 (wrong_goal) 예시 |

주의사항:
- 영상 파일명 run_id 접두사는 07-08 이후 런부터 (이전 영상은 mtime 귀속 — videos_p7/ 폴더 주석 참조).
- FIG1·FIG9의 캠페인 수치 프로토콜은 journal/upgrade_report 문서 도달 시 확정 기입.
- 재생성: `python src/diagnosis/make_presentation.py` (s1 완료 후 실행하면 FIG1·NUMBER_CARD 자동 갱신).
