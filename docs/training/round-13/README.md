# 13차 — 걸음이 물리적으로 말이 되게

12차는 스탠스와 웅크림을 고쳤다. GUI에서 남은 것: 왼앞 무릎이 한 스텝에 점프하고,
나머지는 두둥실하게 떠 보인다. 원인은 보상 하나가 아니라 **한도가 없어서** 그 자세가 싸다는 것.

노브 셋, 문제 하나(불가능한 스윙). 과속 std / `cmd·v` 는 이번에도 안 건드린다.

결과: PLAY에서 왼앞 무릎 50→5.9 rad/s, 한 발 접지 19%→0%, 대각선 트로트 접지율 78%.
학습 로그의 `mean_std` 가 203으로 폭주. **resume 금지.** 다음엔 12차 `model_7148`에서 std 상한 후 클립.

| 파일 | 내용 |
|---|---|
| [01-걸음물리.md](01-걸음물리.md) | 본문 + 학습·PLAY 결과 |
| [figures/01-reward-terms.png](figures/01-reward-terms.png) | 보상 항 곡선 |
| [figures/03-ppo-health.png](figures/03-ppo-health.png) | **mean_std 203** 폭주 |
| [check_joint_vel_limits.py](check_joint_vel_limits.py) | 시뮬이 들고 있는 관절 속도·토크 한도 |
| [check_knee_chatter.py](check_knee_chatter.py) | 스윙 vs 고주파 진동 |
| [watch_gates.py](watch_gates.py) | 학습 중 게이트 숫자 |
| [export_tensorboard_figures.py](export_tensorboard_figures.py) | 12차와 겹쳐 그림 |

테스트 PLAY: `Isaac-Velocity-Rough-Spot-Arm-RoughTest-v0` · `model_8647.pt` (추론은 평균만, std 폭주는 PLAY에 안 보임)

