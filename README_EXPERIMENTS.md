# 실험 실행 브랜치 (exp/executor-log) — 작업 현황·결과 요약

> 실행자(Claude Code) 세션의 작업 브랜치. `RESEARCH_PLAN_delta_anchor_v1.1.md`(계획서)와
> 분석자 스프린트 지시(S1.v2 → S2-prelim → H2-fair → 서베이 병합)를 실행한 코드·결과 전량.
> **모든 수치의 정본은 `outputs/report/*.json` (§8 스키마)** — 인용은 `outputs/presentation/NUMBER_CARD.md` 경유.
> 최종 갱신: 2026-07-08 (지속 갱신됨)

---

## 연구 한 줄 요약

frozen CLIP ViT-L/14 잠재공간의 **변위 Δz = z_{t+16} − z_t를 액션 코드로 쓰는** 정책의
검증 — 액션청크를 Δz에 접지(phase1 g/h)하고 그 위에서 flow matching 정책(phase2)을 학습,
LIBERO-Spatial 폐루프로 평가. 130편 서베이(`최종보고서_v2.md`) 기준 **신규성 성립**.

## 핵심 결과 (LIBERO-Spatial, 50롤/task paired, Wilson CI)

| 구성 | 폐루프 SR | 의미 |
|---|---|---|
| **CLIP-HY03 (정규화, seed2)** | **87.0** [83.8–89.7] | C8 승자 (G5: hybrid 승격) |
| CLIP-HY03 확정 레시피 (비정규화, seed1) | 81.0 [77.3–84.2] | 대표 수치 후보 (시드×레시피 절제 대기) |
| SigLIP2-HY03 | 80.2 [76.5–83.5] | 공정화 전 수치 (vocab v3 재판정 진행) |
| **ARM-AE (정렬 제거 대조군)** | **73.6** [69.4–77.1] | **Δz 접지 자체의 기여 +7.4pp 실증** |
| DINOv2-DZ | 65.2 [60.9–69.2] | 오프라인 전지표 1위인데 폐루프 최하 — **역전** |
| +proprio (기각) | 59.0~73.8 | 인과 혼동 실증 (P6 프로브가 사전 예측) |
| k-NN 바닥선 | 18.2 | 표현만의 성능 |

## 주요 발견 5가지

1. **G2 통과 — 서베이 최대 리스크의 정면 반증**: 문헌은 dynamics에 CLIP을 회피하고
   DINOv2를 선호하지만, 폐루프에서는 **CLIP이 DINOv2를 +21.8pp 압도** (오프라인 코딩
   지표는 DINOv2 압승 — 지표 역전). VLM-LAM의 "CLIP 오프라인 최악" 보고와 결합하면
   오프라인↔폐루프 역전 논증의 독립 보강 증거.
2. **정렬 절제 (C8) 교차 패턴**: 폐루프는 Δz-접지(DZ·HY03) 우세, 언어 축(text→action
   검색·zero-shot)은 직접 정렬(DA) 우세 → **hybrid(λ_c=0.3)가 폐루프 무손실로 언어 축
   획득** — 승격 확정. ARM-AE 대조군으로 접지 자체의 기여(+7.4pp)도 분리 실증.
3. **R² 붕괴 원인 규명 (Phase 1.5)**: ALOHA 0.99 vs LIBERO 0.68의 원인은 no-op·해상도·
   청크길이가 아니라 **인간 데모 다봉성** (조건부 액션 분산 3배) — flow 디코더 채택 근거.
4. **proprio 기각 + P6 표준**: 정보를 더해도(+0.048 상한) 폐루프는 −28pp — 인과 혼동.
   "후보 토큰 단독 정책 MAE" 프로브(P6)가 이를 사전 예측함을 소급 검증, 신규 토큰 도입
   관문으로 표준화. (한계: 필요조건 스크린 — 그리퍼 2D 사례)
5. **공정 비교 프로토콜 (H2-fair)**: 모션 어휘가 CLIP 편향(변별력 2배)임을 실측 →
   dual-score v3 어휘로 재판정. SigLIP2 격차의 ~45%가 어휘 편향, 잔여는 목적함수
   비정합(시그모이드 vs InfoNCE).

## 재현 진입점

```bash
# 환경: environment_libero.yml (conda clip_libero) — mujoco==3.3.2 고정
# phase1 (확정 레시피: joint+비정규화+hybrid λ0.3):
python src/training/train_phase1.py --config configs/phase1_libero.yaml \
  --set anchor.normalize=false --set model.align_mode=hybrid --set loss.contrast=0.3
# phase2 (flow+손목캠 d1536) + 폐루프 50/task:
python src/training/train_phase2.py --config configs/phase2_libero.yaml
MUJOCO_GL=egl python src/eval_libero/rollout_sim.py --episodes 50
```

## 디렉터리 안내 (이 브랜치에서 추가된 것)

| 경로 | 내용 |
|---|---|
| `src/core/anchor.py` | 앵커 추상화 (CLIP/SigLIP2/DINOv2, 캐시 키 분리) |
| `src/data/motion_lang.py` + `data/motion_lang*.json` | C8 모션 문장 어휘 v1/v2/v3(중립) |
| `src/diagnosis/` | D1~D4 진단, P6 지름길 프로브, P7 강건성, 정렬 리포트, 발표 자산 생성 |
| `src/eval_libero/{knn_baseline,c8_zeroshot,c8_gapfix,c8_f2_prior}.py` | 바닥선·zero-shot·갭 보정 |
| `docs/upgrade_ledger.md` | 컴포넌트 업그레이드 원장 + 예측 장부 + 판정 이력 |
| `docs/eval_protocol.md` | 폐루프 공식 프로토콜 (비결정성 규칙 포함) |
| `docs/related_competitors.md` | LARA/JALA/VLM-LAM 차별화 + A2A 인용 확정 |
| `docs/talk_errata.md` | 발표자료 대비 실측 갱신 목록 |
| `outputs/report/` | §8 JSON 전 런 + 판정 문서 (git 포함) |
| `outputs/presentation/` | FIG1~10 + NUMBER_CARD + 영상 큐레이션 |
| `scripts/remote_ablation_batch.sh` | 원격(A6000) 절제 배치 |

## 진행 중 / 다음

- 원격 A6000 (GPU 8·9) 병렬화: 절제 배치 #2~7 + C7 융합 셀(CLIP-HY03 + DINOv2 mean-patch)
- H2-fair v3 재판정 → SigLIP2 최종 비교
- 분석자 대기: G2 종합, S2 정식 설계(계층 하이브리드), 대표 수치 절제 승인

## 데이터·재현 주의

- 폐루프 수치는 fp16 비결정성으로 태스크별 ±30pp 변동 — **suite 평균(n≥500)만 공식** (`docs/eval_protocol.md`)
- 대용량(데이터·체크포인트·캐시)은 gitignore — 재생성 방법은 각 README/스크립트 참조
