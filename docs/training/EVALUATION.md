# 롤아웃 평가 — GUI 말고 숫자로 보기

GUI는 “넘어지나, 걸음이 그럴듯한가”만 빠르게 본다.  
라운드 간 비교·회귀 잡기·다음 노브 결정은 **아래 지표**로 한다.

## 한 번에 돌리기

```bash
source scripts/isaaclab_env.sh
python scripts/evaluate_rollout.py \
  --checkpoint logs/rsl_rl/rough_spot_with_arm/2026-08-27_10-20-22/model_10146.pt \
  --json docs/training/round-15/data/round15_rollout.json
```

기본 태스크: `Isaac-Velocity-Rough-Spot-Arm-RoughTest-v0` (vx=1.0, 험지 80 env).  
평지만 보려면 `--task Isaac-Velocity-Rough-Spot-Arm-Play-v0`.

학습 중 TensorBoard 게이트는 각 라운드 `watch_gates.py` (에피소드 길이, `mean_std`, 지형 레벨 등).

---

## 지표 — 무엇을 왜 보나

### A. 속도 추종 (`velocity_tracking`)

| 지표 | 의미 | 나쁜 신호 |
|---|---|---|
| `vx_tracking_ratio` | 실제 vx / 명령 vx | 1.5처럼 **과속** (내리막·느슨한 추종) |
| `error_vy_mean`, crab angle | 옆으로 끌림·요 흔들림 | 직진 명령인데 vy·각도 큼 |
| `error_wz_mean` | 요 레이트 | 명령 0인데 몸이 돔 |

눈으로는 “빠르게/느리게 간다” 정도만 보인다. 숫자로 **과속·게걸음**을 잡는다 (14·15차 미해결 항).

### B. 접지 패턴 (`contact_pattern`)

| 지표 | 의미 | 나쁜 신호 |
|---|---|---|
| `feet_down_1_pct` | 한 발만 접지인 프레임 비율 | 높음 → 두둥실·삼족보행 (12차 19%) |
| `diagonal_trot_contact_pct` | FL+RR / FR+RL 중 **정확히 한 대각선만** 접지 | 낮음 → 트로트 아님 (목표 >70%) |
| `feet_down_2_pct` | 두 발 접지 | 트로트면 대략 60–80% |

`gait` 보상이 먹었는지, PLAY에서도 대각선이 유지되는지 확인.

### C. 관절 자세 (`knees`, `leg_joints`)

| 지표 | 의미 | 나쁜 신호 |
|---|---|---|
| 무릎 `mean` vs `default` (−1.5) | 웅크림 오프셋 | −1.9 이하로 깊게 상주 (14·15차 이슈) |
| `range_p95_p5` (가동폭) | 스윙 시 실제 움직임 | <0.3 rad → 잠김 (13차 클립 1.0) |

평균만 보면 “굽었다” 정도; 가동폭으로 **잠김 vs 스윙**을 구분.

### D. 스탠스 (`stance`)

| 지표 | 의미 | 나쁜 신호 |
|---|---|---|
| `front_stance_pct_of_nominal` | 앞발 좌우 간격 / URDF 기준 | 19% (11차) → 발이 배 밑 |

### E. 보폭 타이밍 (`gait_timing`)

| 지표 | 의미 | 나쁜 신호 |
|---|---|---|
| `swing_s_mean` | 공중에 있는 시간 | 0.08 s (12차 왼앞) — 실기 0.20–0.35 s |
| `stride_hz` | 보폭 주파수 | 다리마다 30% 이상 벌어지면 비대칭 |
| `peak_foot_speed_m_s` | 발 끝 속도 | 한 다리만 튀면 순간이동 의심 |

### F. 몸통 (`body`)

| 지표 | 의미 | 나쁜 신호 |
|---|---|---|
| `height_above_terrain_mean_m` | 지형 위 몸 높이 | 0.43 m (14차) vs 서기 0.54 m |
| `height_above_terrain_std_m` | 상하 흔들림 | 크면 통통 튐 (떠 보이는 느낌) |

### G. 관절 속도 (`joint_velocity`)

| 지표 | 의미 | 나쁜 신호 |
|---|---|---|
| 무릎 `p99` vs URDF 12 rad/s | 관절이 얼마나 빨리 도는지 | 50 rad/s (12차) → 물리적으로 말 안 됨 |

### H. 액션 (`actions`)

| 지표 | 의미 | 나쁜 신호 |
|---|---|---|
| `saturated_pct` | `|a| > 0.95×clip_actions` 비율 | 높음 → 클립에 박힘, 보폭/자세 한계 |
| `max_abs` | 최대 액션 | clip 넘으면 환경은 잘리지만 정책은 포화 |

13차 `clip=1.0`, 14·15차 `clip=2.5` 비교에 쓴다.

### I. 종료 (`terminations`)

| 지표 | 의미 | 나쁜 신호 |
|---|---|---|
| `episode_done_frames_pct` | 측정 구간 중 reset 비율 | 높음 → 자주 넘어짐 |

---

## 학습 중 vs 롤아웃 후

| 시점 | 도구 | 보는 것 |
|---|---|---|
| 학습 중 | `watch_gates.py` | 에피소드 길이, `mean_std` 폭주, 지형 레벨, `error_vel_xy` |
| 체크포인트 후 | `evaluate_rollout.py` | 위 A–I 전부, JSON 저장 |
| 예전 스크립트 | `round-11/diagnose_gait.py`, `round-12/diagnose_gait_timing.py` | 부분 집합 (통합 스크립트로 대체 가능) |

라운드 끝날 때마다 `--json docs/training/round-XX/data/roundXX_rollout.json` 으로 남기면 diff가 쉽다.
