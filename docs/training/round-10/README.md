# 10차 — 험지 커리큘럼

9차 평지 걸음 체크포인트에서 이어서, 지형만 켬. 레벨 0부터.

결과: 레벨 6까지 오르고 800 iter 정체. `feet_air_time` 0.011. 원인은 `foot_clearance` 가
발 높이를 세계좌표 상수 위에서 잰 것 → [round-11](../round-11/).

| 파일 | 내용 |
|---|---|
| [01-험지커리큘럼.md](01-험지커리큘럼.md) | 본문 + 결과 |
| [export_tensorboard_figures.py](export_tensorboard_figures.py) | 그림·수치 추출 |

테스트 PLAY: `Isaac-Velocity-Rough-Spot-Arm-RoughTest-v0` (10×20 격자)
