# 15차 — 무릎을 기본 자세로

앞 무릎 mean −1.90 → **−1.75**, 몸통 높이 0.43 → **0.47 m**. 걸음·std 유지.
체크포인트: `2026-08-27_10-20-22/model_10146.pt`.

| 파일 | 내용 |
|---|---|
| [01-무릎-기본자세-당김.md](01-무릎-기본자세-당김.md) | 본문 + 학습·PLAY 결과 |
| [figures/01-reward-terms.png](figures/01-reward-terms.png) | 보상 항 곡선 |
| [figures/03-ppo-health.png](figures/03-ppo-health.png) | mean_std·entropy |
| [data/round15_rollout.json](data/round15_rollout.json) | `evaluate_rollout` 숫자 |
| [videos/round-15-play.mp4](videos/round-15-play.mp4) | PLAY 캡처 |
| [watch_gates.py](watch_gates.py) | 학습 중 TensorBoard 스냅샷 |
| [export_tensorboard_figures.py](export_tensorboard_figures.py) | 14차와 겹쳐 그림 (구버전) |

그래프 재생성: `python docs/training/export_training_figures.py logs/rsl_rl/rough_spot_with_arm/2026-08-27_10-20-22 --out docs/training/round-15 --baseline logs/rsl_rl/rough_spot_with_arm/2026-08-26_12-52-45`
