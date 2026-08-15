---
name: remote-control
description: >-
  Pushes the current spot_arm_locomotion work to origin/main. Use when the user
  runs /remote-control, says remote-control, or asks to main 에 올려.
disable-model-invocation: true
---

# remote-control

main 에 올려

## Goal

Put the current `spot_arm_locomotion` tree on GitHub `main`. Do not force-push. Do not skip hooks.

Repo: `/home/june/isaac_projects/spot_arm_locomotion`  
Remote: `git@github.com:June2December/spot_arm_locomotion.git`

## Steps

1. `cd` to the repo. Confirm `git status`, `git diff`, `git log -8 --oneline`, and `git remote -v`.
2. Ignore `logs/`, USD caches, `*.pt`, `*.onnx`, `.egg-info`, `__pycache__`. Keep `assets/spot_with_arm` as a symlink.
3. If the GitHub repo is missing, tell the user to create an empty `June2December/spot_arm_locomotion` (no README) and stop.
4. If there are uncommitted changes, stage the relevant files and commit with a 1–2 sentence message via HEREDOC. Never `git config`. If identity is missing, use a one-shot `git -c user.name=... -c user.email=...` for that commit only.
5. If HEAD is not `main`, checkout `main` and merge (or cherry-pick) the work into `main`. Do not rewrite published history.
6. `git push -u origin main`.
7. Reply with the commit hash and the GitHub URL.

## Do not

- `push --force` to `main`
- `--no-verify` / `--no-gpg-sign`
- Commit secrets (`.env`, credentials)
- Recreate Cursor's built-in phone Remote Control handoff; this skill is git `main` only
