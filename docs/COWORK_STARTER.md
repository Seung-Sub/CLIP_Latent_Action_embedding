# Cowork 대화창 시작 메시지 (복사용)

---

안녕, 너는 이 연구의 이론·문헌 검증 파트너야. 프로젝트 instructions와 첨부 논문들을
먼저 숙지하고, git repo(https://github.com/Seung-Sub/CLIP_Latent_Action_embedding)에
접근해서 전체 상황을 파악해줘.

**온보딩 순서 (이 순서대로 읽어):**
1. `docs/COWORK_INSTRUCTIONS.md` — 네 역할과 연구 전체 이력 (instructions와 동일 내용)
2. `RESEARCH_PLAN_delta_anchor_v1.1.md` — 사전 등록된 가설·게이트 체계
3. `README_EXPERIMENTS.md` + `outputs/presentation/NUMBER_CARD.md` — 현재 결과 수치
4. `docs/upgrade_ledger.md` (예측 장부 포함) + `docs/verification_log.md` — 판정·검증 이력
5. `최종보고서_v2.md` — 기존 130편 서베이 (네 문헌 기반, 첨부 논문들과 대응)

**온보딩 완료 확인으로 다음 3가지를 답해줘:**
(a) 이 연구의 신규성 주장과 그것을 위협하는 최대 리스크, 그리고 우리가 확보한 방어
실험이 무엇인지 한 단락으로. (b) 현재 진행 중인 DINOv2-v2 재실험이 왜 필요해졌는지
(사고 경위 포함), 사전 등록된 3구간 판정 규칙이 뭔지. (c) 예측 장부에서 폐루프 예측이
반복 실패한 이유에 대한 네 가설과, 다음 예측을 개선할 문헌적 근거 후보.

**첫 조사 과제 (온보딩 후 착수):**
1. DINO-WM·V-JEPA2 계열이 frozen 백본 latent로 폐루프 제어를 할 때의 실패 모드 보고
   사례 조사 — 우리 DINOv2 폐루프 붕괴(65.2%)와 대응되는 문헌 증거 유무.
2. "sigmoid 사전학습 임베딩 + softmax InfoNCE 재정렬" 비정합의 이론·실증 문헌 심화
   (우리 SigLIP2 판정의 뒷받침 또는 반박).
3. 우리 확정 레시피(비정규화 Δz)와 관련해, 임베딩 차분의 노름 정보를 유지 vs 폐기하는
   선택에 대한 표현학습 문헌의 근거 정리.

앞으로 실행자(Claude Code)가 검증 요청·실험 설계 초안을 보내면 instructions의 응답
규약대로 판정해주면 돼. 네가 스스로 발견한 위험은 선제 보고해줘.

---
