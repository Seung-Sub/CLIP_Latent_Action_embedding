# configs/
- `config.yaml`  : CLIP 백본 경로·정밀도 (core/clip_wrapper.py가 읽음)
- `phase1.yaml`  : delta-AE 학습 — 그리드서치 확정 레시피 (state_cond, patchgrid, 16청크, 0.5/0.5/0.25)
- `phase2.yaml`  : 정책 f 학습 — 46런 그리드 확정 (MLP-concat d1024 L4, act1.0+lat0.5, past_noise 0.05)

모든 값에 근거가 주석으로 달려 있음. 실험 시 파일 수정 대신
`python src/training/train_phase*.py --set key=value --tag 이름` 오버라이드 권장.
