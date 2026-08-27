# 12차 — 기본 자세

11차는 걷지만 무릎을 0.7 rad 더 접고 네 발을 배 밑에 모은다. 다리를 기본 자세로 당기는 항이 없었다.
`leg_joint_deviation` · `hip_x_deviation` 신규 + `dof_pos_limits` 를 다리로 스코프해 −10.0 복원.

결과: 앞 스탠스 폭 0.063 → **0.395 m**, 무릎 편차 0.7 → 0.3 rad, `hip_x` 부호 복귀.
지형 레벨 5.9 → 5.6 으로 걸음은 유지. 과속은 14차. 순간이동·두둥실은 [round-13](../round-13/).

| 파일 | 내용 |
|---|---|
| [01-기본자세.md](01-기본자세.md) | 본문 + 결과 |
| [check_joint_limit_penalty.py](check_joint_limit_penalty.py) | `dof_pos_limits` 의 다리/팔 분해 |
| [export_tensorboard_figures.py](export_tensorboard_figures.py) | 11차와 겹쳐 그림·수치 추출 |

자세 측정은 [`../round-11/diagnose_gait.py`](../round-11/diagnose_gait.py) 를 그대로 씀.
