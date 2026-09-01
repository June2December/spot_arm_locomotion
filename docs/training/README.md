# Training log

One folder per training round. Checkpoints and raw TensorBoard runs stay in gitignored `logs/`.

| Folder | Summary |
|---|---|
| [round-02](round-02/) | Round 1. Fell over in ~0.28 s |
| [round-03](round-03/) | Round 2. Stood but did not walk |
| [round-04](round-04/) | Round 4. Balanced in place |
| [round-05](round-05/) | Round 5. Body moved, feet did not lift |
| [round-07](round-07/) | Round 7. Foot-clearance reward only |
| [round-08](round-08/) | Round 8. Fall/slide penalties → short episodes |
| [round-09](round-09/) | Round 9. PD 80. Walking on flat terrain |
| [round-10](round-10/) | Round 10. Rough curriculum from round 9. Stuck at terrain level 6 |
| [round-11](round-11/) | Round 11. Foot height measured above terrain, not world z |
| [round-12](round-12/) | Round 12. Default-pose leg pull. Stance width 19% → 119% |
| [round-13](round-13/) | Round 13. Clip + trot. Better PLAY gait; `mean_std` blew up to 203 |
| [round-14](round-14/) | Round 14. `clip_actions=2.5`. No std blow-up; knees stayed crouched |
| [round-15](round-15/) | Round 15. Knee default pull. Front knee −1.90→−1.75, height 0.43→0.47 m · [PLAY](round-15/videos/round-15-play.mp4) |
| [round-16](round-16/) | Round 16. Lab velocity commands, dot off, std 0.5 · vx ratio 1.44→0.52 |
| [round-17](round-17/) | Round 17. Narrow commands · vx ratio 0.52→**0.92** · terrain 0.38→3.62 |

학습 그래프 재생성: `docs/training/export_training_figures.py` · PLAY 숫자: `scripts/evaluate_rollout.py`

