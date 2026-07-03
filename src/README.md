# src/
| 패키지 | 파일 | 역할 |
|---|---|---|
| core/ | clip_wrapper.py | frozen CLIP ViT-L/14: pooled 768 임베딩 + 패치토큰 |
|       | config.py | configs/config.yaml 로드 |
| data/ | act_sim.py | HDF5 로더: 임베딩 캐시, (z_t, z_{t+16}, 청크) 쌍, 정책용 삼중쌍(경계 포함) |
| models/ | networks.py | Phase1 DeltaAE (ChunkEncoder g / ChunkDecoder h / g2dec) |
|         | policy.py | Phase2 f 모듈 3종(mlp/cls/pma) + 3항 손실 |
| training/ | train_phase1.py, train_phase2.py | 학습 (--smoke 점검, --set 오버라이드, wandb) |
| eval/ | eval_gt_trace.py | GT 에피소드 전체 시계열 추론 → 14차원 그래프 + MAE |
|       | rollout_sim.py | 시뮬 폐루프 실시간 추론 → 성공률·reward·영상 |
