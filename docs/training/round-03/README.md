# 2차 학습 — 서기는 하는데 안 걸음

2차(2026-08-15, 1500 iter, 4096 env, 평지)는 1차처럼 0.28초에 넘어지지 않음.  
대신 **20초 동안 제자리**. 걷는 보상 `track_lin_vel_xy_exp` 0.62는 걸었다는 뜻이 아님. 명령이 ±0.5 m/s라서 가만히 있어도 그 점수가 나옴.

체크포인트·tfevents는 `logs/` (gitignore). 여기엔 글·그래프·캡처만 둠.

| 파일 | 내용 |
|---|---|
| [01-2차-서기만함.md](01-2차-서기만함.md) | 본문. 보상 표가 맨 위, 그다음 캡처·그래프 |
| [figures/](figures/) | TensorBoard PNG 8장 |
| [screenshots/](screenshots/) | PLAY 제자리 프레임 |
| [data/round2b_metrics.json](data/round2b_metrics.json) | 시작/끝 숫자 |
| [export_tensorboard_figures.py](export_tensorboard_figures.py) | 그래프 다시 뽑기 |

런: `logs/rsl_rl/rough_spot_with_arm/2026-08-15_16-36-59/`  
1차 분석은 [../round-02/](../round-02/) 에 그대로 둠.
