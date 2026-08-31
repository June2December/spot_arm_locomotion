# 11차 — 클리어런스 기준

`foot_clearance` 가 발 높이를 세계좌표 상수 위에서 재고 있었다. 험지에서는 그게 지형 높이 보상이다.
그 발 아래 지형 높이 위에서 재도록 바꿈. 10차 `model_4150` 에서 이어서.

| 파일 | 내용 |
|---|---|
| [01-클리어런스기준.md](01-클리어런스기준.md) | 본문 |
| [02-롤아웃-자세결함.md](02-롤아웃-자세결함.md) | 롤아웃에서 본 자세 결함 4개 실측 |
| [check_foot_clearance_reference.py](check_foot_clearance_reference.py) | 행동 0 으로 세계 z 기준 vs 지형 상대 실측 |
| [diagnose_gait.py](diagnose_gait.py) | 관절 가동폭·발 위치·접지·직진 추종 (부분) — 통합: [`../../scripts/evaluate_rollout.py`](../../scripts/evaluate_rollout.py) |
| [watch_gates.py](watch_gates.py) | 학습 중 게이트 수치 출력 |
| [export_tensorboard_figures.py](export_tensorboard_figures.py) | 그림·수치 추출 |
