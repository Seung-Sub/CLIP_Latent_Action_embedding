#!/bin/bash
# 절제 배치 #2~7 — 원격 GPU8 전용 (서베이 병합 지시)
# 실행: CUDA_VISIBLE_DEVICES=8 bash scripts/remote_ablation_batch.sh
# 전제: /workspace/clip_ws, 캐시 동기화 완료, 시스템 python
set -u
cd /workspace/clip_ws
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-8}
PY=python3
BASE="--config configs/phase1_libero.yaml --set wandb.enabled=false --set anchor.normalize=false"
HY="--set model.align_mode=hybrid --set loss.contrast=0.3"

run() { echo "=== $1"; shift; $PY src/training/train_phase1.py $BASE "$@" 2>&1 | grep -E "디코더|cycle|align cos|top-1|저장|Traceback" | head -6; }

# #2 L_comp (HY03 위)
run abl_comp01 $HY --set loss.comp=0.1 --tag abl_comp01
# #3 quantile 정규화 (HY03 위)
run abl_quantile $HY --set data.norm_scheme=quantile --tag abl_quantile
# #4 QueST 조건화 절제 (DZ 모드에서 — 조건화 효과의 순수 측정)
run abl_quest_a --set model.g_state_cond=false --tag abl_quest_a
run abl_quest_b --set model.g_state_cond=false --set model.h_state_cond=true --tag abl_quest_b
# #5 velocity L2
run abl_vel025 $HY --set loss.vel=0.25 --tag abl_vel025
# #6 인코더 4종 (기준 cnn = bridge_hy03_unnorm_p1 재사용, 나머지 3종)
run abl_enc_strided $HY --set model.encoder_kind=strided --tag abl_enc_strided
run abl_enc_transformer $HY --set model.encoder_kind=transformer --tag abl_enc_transformer
run abl_enc_mlp $HY --set model.encoder_kind=mlp --tag abl_enc_mlp
touch outputs/abl_batch.DONE
echo ALL-DONE
