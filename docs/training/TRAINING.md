# 학습 중 모니터링 — RL 튜닝 가이드

GUI·롤아웃 숫자는 **학습이 끝난 뒤** 정책을 검증할 때 쓴다.  
**학습 도중**에는 TensorBoard 스칼라를 보며 “이 노브를 돌려도 되는지 / 지금 끊어야 하는지”를 판단한다.

이 문서는 그때 보는 것, 우리 보상 항, 라운드마다 실제로 본 신호를 정리한다.  
각 라운드 `01-*.md`의 표·게이트는 이 가이드를 **그 라운드에 적용한 기록**이다.

---

## 1. 학습 중 확인 순서

```
TensorBoard 켜기
    ↓
① 에피소드가 살아 있는가? (길이, termination)
    ↓
② PPO가 망가지지 않았는가? (mean_std, entropy)
    ↓
③ 커리큘럼이 오르는가? (terrain level)
    ↓
④ 보상 항이 의도대로 움직이는가? (Episode_Reward/*)
    ↓
⑤ 추종·걸음 지표 (error_vel_xy, gait, …)
    ↓
학습 끝 → export 그래프 + evaluate_rollout.py
```

터미널에서 빠르게 보려면:

```bash
python docs/training/watch_gates.py logs/rsl_rl/rough_spot_with_arm/<run_dir>
```

학습 끝나면 그래프·JSON:

```bash
python docs/training/export_training_figures.py \
  logs/rsl_rl/rough_spot_with_arm/<run_dir> \
  --out docs/training/round-XX \
  --baseline logs/rsl_rl/rough_spot_with_arm/<prev_run> \
  --title "Round XX"
```

---

## 2. TensorBoard 카테고리

### 2.1 에피소드 생존 (가장 먼저)

| 태그 | 뜻 | 정상 | 끊을 신호 |
|---|---|---|---|
| `Train/mean_episode_length` | 평균 에피소드 길이 (스텝) | timeout 근처 (~987 step ≈ **19.7 s**) | 5 s 아래로 급락 (13차 10.6 s) |
| `Episode_Termination/time_out` | 시간 만료로 끝난 비율 | **>0.9** | 급감 + `base_contact`/`bad_orientation` 급증 |
| `Episode_Termination/base_contact` | 몸통·팔이 땅에 닿아 종료 | **<0.05** | **>0.3** (13차 0.45) |
| `Episode_Termination/bad_orientation` | 기울어져 종료 | **<0.05** | 급증 |

**왜:** 보상이 아무리 좋아 보여도 에피소드가 짧으면 “걷다 죽는” 정책이다. 1차는 여기서 0.28 s에 끝났다.

### 2.2 PPO / 정책 건강 (두 번째 — 13차 교훈)

| 태그 | 뜻 | 정상 | 끊을 신호 |
|---|---|---|---|
| `Policy/mean_std` | 가우시안 행동 표준편차 | **0.4–0.7** | **>2** 지속, 13차 **203** |
| `Loss/entropy` | 엔트로피 보너스 | 완만 | 13차 12→76 과 함께 std 폭주 |
| `Loss/value` | 가치망 손실 | 완만 상승 | std 폭주와 같이 튀면 학습 불안 |
| `Loss/learning_rate` | adaptive KL lr | | std 폭주 시 비정상적으로 커질 수 있음 |

**왜:** `clip_actions`가 있으면 환경에 들어가는 행동은 잘리지만, **엔트로피는 잘리기 전 분포**에 대해 계산된다. std를 키우면 엔트로피만 오르고 실제 행동은 안 바뀐다 → PPO가 std를 무한히 키우는 게 유리 (13차).  
**PLAY는 평균만 쓰므로** std 폭주가 GUI에 안 보일 수 있다. **resume 전에 반드시 확인.**

### 2.3 커리큘럼

| 태그 | 뜻 | 정상 | 나쁜 신호 |
|---|---|---|---|
| `Curriculum/terrain_levels` | 평균 지형 난이도 (0–9) | resume 후 다시 올라 **~5–6** | 3 아래에서 정체 (클립이 보폭 죽임) |

10차부터 험지. 레벨이 안 오르면 “안 넘어지지만 앞으로 못 감”.

### 2.4 추종 (학습 로그)

| 태그 | 뜻 | 참고 |
|---|---|---|
| `Metrics/base_velocity/error_vel_xy` | xy 속도 오차 | 15차 끝 **~0.43**. PLAY 과속과 별개로 느슨한 추종 신호 |
| `Metrics/base_velocity/error_vel_yaw` | 요 오차 | |
| `Metrics/success_rate` | Lab success metric | 낮아도 episode 길이가 길면 일단 통과한 적 있음 |

16차 후보: `track_lin_vel_xy_exp` std, `track_lin_vel_xy_dot` 조정.

### 2.5 `Train/mean_reward` — 라운드 간 비교 금지

보상 **항을 추가·가중치를 바꾸면** 총 reward 스케일이 바뀐다.

| 라운드 | mean_reward 대략 | 왜 비교하면 안 되나 |
|---|---|---|
| 12차 끝 | ~13 | 기준 |
| 13차 끝 | ~32 | `gait` +5.0 신규 |
| 14차 끝 | ~115 | gait가 1.88→4.48로 커짐 |
| 15차 끝 | ~117 | leg_joint_deviation만 변경 |

**항별** `Episode_Reward/*` 와 생존 지표로 본다.

---

## 3. 보상 항 전체 (현재 env)

가중치는 `spot_arm_env_cfg.py` / `spot_arm_rough_cfg.py`.  
TensorBoard에는 **가중치 곱한 뒤 에피소드 평균**이 찍힌다.

### 추종

| 항 | weight | 학습 중 보면 |
|---|---:|---|
| `track_lin_vel_xy_exp` | 1.0 | 전진 추종. std 넓으면 과속 허용 |
| `track_lin_vel_xy_dot` | 1.5 (rough) | 명령 방향으로 속도 내적. 과속에도 점수 |
| `track_ang_vel_z_exp` | 0.5 | 요 추종 |

### 걸음

| 항 | weight | 학습 중 보면 |
|---|---:|---|
| `gait` | 5.0 | **0에서 안 오르면** 발 이름/게이트 문제 (13차 게이트). 4+ 이면 트로트 학습 중 |
| `feet_air_time` | 1.5 | 발 들림. 명령 있을 때만 |
| `foot_clearance` | 2.0 | 지형 위 발 높이. 11차: 세계 z 기준이면 험지에서 깨짐 |

### 자세·안정

| 항 | weight | 학습 중 보면 |
|---|---:|---|
| `leg_joint_deviation` | −0.2 | hip_y+knee default 당김. **더 음수** = 더 웅크림 벌점 (15차 −0.53) |
| `hip_x_deviation` | −1.0 | 스탠스 폭. 12차 핵심 |
| `dof_pos_limits` | −10.0 | 관절 한계 (다리만) |
| `arm_joint_deviation` | −1.0 | 팔 접힘 유지 |
| `flat_orientation_l2` | −1.0 | 몸 수평 |
| `lin_vel_z_l2`, `ang_vel_xy_l2` | −2.0 | 통통 튐 억제 |

### 효율·안전

| 항 | weight | 학습 중 보면 |
|---|---:|---|
| `dof_torques_l2`, `dof_acc_l2` | 작은 음수 | 토크·가속 페널티 |
| `action_rate_l2` | −0.005 | 행동 변화율 |
| `undesired_contacts` | −1.0 | 허벅지 등 원치 않는 접촉 |
| `termination_penalty` | −200 | 넘어짐·비정상 종료 |

그래프: `figures/01-reward-terms.png` (export 스크립트).

---

## 4. 증상 → 노브 (튜닝 표)

| 학습/PLAY에서 본 것 | TensorBoard 신호 | 다음 노브 |
|---|---|---|
| 금방 넘어짐 | `base_contact`↑, episode 짧음 | PD, reset, termination_penalty (1–2차) |
| 서만 있음 | `feet_air_time`≈0, episode 짧지 않음 | 발 들림 보상 (7차) |
| 평지만 걸음 | terrain level 0 | 험지 커리큘럼 (10차) |
| 발 높이 보상 이상 | foot_clearance만 높음, 험지 실패 | clearance 기준 지형 상대 (11차) |
| 무릎 깊게·발 모임 | leg_dev 낮음, PLAY stance 19% | leg/hip_x deviation (12차) |
| 한 다리 순간이동 | PLAY 무릎 p99 50 rad/s | `clip_actions`, `gait` (13차) |
| std 203, episode 10 s | `mean_std` 폭주 | **resume 금지**, entropy↓ clip 조정 (14차) |
| 무릎 ㄱ 고정 | PLAY 가동폭 0.4 | clip 2.5 (14차) |
| 무릎 mean −1.9 | leg_dev hip_y만 | knee를 leg_dev에 복원 (15차) |
| PLAY 과속 1.5 m/s | `error_vel_xy`~0.43 | tracking std / dot weight (16차 예정) |

**원칙:** 한 라운드에 **한 문제·노브 묶음**. 자세와 속도 추종을 동시에 바꾸면 원인 분리 불가 (12→13, 14→15에서 지킴).

---

## 5. 라운드별 — 실제로 본 신호 → 결정

| Round | 학습 중 본 것 | 노브 | 롤아웃/GUI로 확인 |
|---|---|---|---|
| 1–2 | episode 0.28 s, bad_orientation | PD, reset, penalty | 넘어짐 영상 |
| 7–9 | feet_air_time, 평지 walking | 발 보상, PD 80 | 평지 PLAY |
| 10 | terrain level 6 정체 | (커리큘럼 유지) | 험지 |
| 11 | foot_clearance 험지 오류 | clearance 기준 | diagnose_gait 자세 |
| 12 | leg_dev 없음, stance 19% | leg/hip_x dev, dof_limits | diagnose_gait |
| 13 | gait 0→1.88, **mean_std→203** | clip 1.0, gait 5.0 | timing: 트로트 78%, **resume 금지** |
| 14 | mean_std 0.54 유지 | clip 2.5, entropy 0.005, 12차 resume | 무릎 mean −1.9, 가동폭 0.7 |
| 15 | leg_dev −0.53, episode 유지 | knee를 leg_dev 복원 | 무릎 −1.75, 높이 0.47 |

각 라운드 `watch_gates.py`는 그때 **실제로 찍어본 태그 subset**이다.

---

## 6. 라운드 끝 체크리스트

- [ ] `export_training_figures.py` → `figures/00`–`04`, `data/*_training_metrics.json`
- [ ] `00-dashboard`: episode, terrain, mean_std
- [ ] `01-reward-terms`: 새로 넣은 항이 곡선으로 움직였는지
- [ ] `03-ppo-health`: mean_std 폭주 없음
- [ ] `evaluate_rollout.py --json` → PLAY 숫자 (EVALUATION.md)
- [ ] GUI는 위 통과 후

---

## 7. 관련 문서

| 문서 | 내용 |
|---|---|
| [EVALUATION.md](EVALUATION.md) | 학습 **후** 롤아웃 숫자 (접지, 무릎, 과속) |
| [README.md](README.md) | 라운드 폴더 인덱스 |
| `scripts/evaluate_rollout.py` | 롤아웃 평가 |
| `export_training_figures.py` | 학습 그래프 export |
| `watch_gates.py` | 학습 중 터미널 게이트 |
