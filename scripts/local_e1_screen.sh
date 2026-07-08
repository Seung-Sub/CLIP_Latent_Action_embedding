#!/bin/bash
# E1 폐루프 스크리닝 (로컬, 10eps/task × 12 config) — paired, 앵커별 native 승자 선정
source ~/miniconda3/etc/profile.d/conda.sh && conda activate clip_libero
cd ~/clip_ws
for anc in clip siglip2; do for nm in true false; do for md in 0.1 0.3 dz; do
  tag=e1_${anc}_norm${nm}_${md}
  cfg=outputs/c8_eval_cfgs/${tag}.yaml
  [ -f checkpoints/grid/p2_${tag}.pt ] || { echo "SKIP $tag (no ckpt)"; continue; }
  MUJOCO_GL=egl python src/eval_libero/rollout_sim.py --config $cfg --episodes 10 \
    --phase e1_screen --data-variant raw --run-id e1screen_${anc}_norm${nm}_${md} \
    --notes "E1 screening 10eps/task paired: anchor=$anc norm=$nm align=$md" 2>&1 | grep '평균 성공률'
done; done; done
touch outputs/e1_screen.DONE; echo E1-SCREEN-DONE
