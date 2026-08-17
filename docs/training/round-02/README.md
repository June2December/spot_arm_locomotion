# 2차 학습 준비 — 1차가 넘어진 이유

1차(2026-08-15, 1500 iter)는 걷는 법을 배운 게 아님.  
평균 0.28초에 옆으로 넘어짐. 화면의 “오른쪽 다리 안 닿고 바로 쓰러짐”과 같음.

체크포인트·tfevents는 `logs/` (gitignore). 여기엔 글·그래프·캡처·영상만 둠.

| 파일 | 내용 |
|---|---|
| [01-1차-실패분석.md](01-1차-실패분석.md) | 본문. 보상 표가 맨 위, 그다음 캡처·그래프 |
| [figures/](figures/) | TensorBoard PNG 8장 |
| [screenshots/](screenshots/) | PLAY 넘어지는 프레임 5장 |
| [videos/rl-video-step-0.mp4](videos/rl-video-step-0.mp4) | 카메라 2.4초 |
| [data/round1_metrics.json](data/round1_metrics.json) | 시작/끝 숫자 |
| [export_tensorboard_figures.py](export_tensorboard_figures.py) | 그래프 다시 뽑기 |

런: `logs/rsl_rl/rough_spot_with_arm/2026-08-15_13-54-47/`

2차(서서만 있음) 분석은 [../round-03/](../round-03/) .
