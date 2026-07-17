# AGENTS.md

## Project purpose

This repository is used for reinforcement-learning research based on
bigger-better-faster-SAC.

The current research goal is to improve the algorithm while preserving
reproducibility, clear Git history, and fair Atari 100K evaluation.

## Git workflow

- Work only on the current `exp/*` experiment branch.
- Never modify `research-main` or `bbf-starting-point-claude` directly.
- Before editing, run `git branch --show-current` and report the current branch.
- Do not switch branches unless explicitly requested.
- Do not run `git commit` or `git push` unless explicitly requested.
- Do not run `git reset`, `git clean`, `git rebase`, force push, or history-rewriting commands.
- Do not delete existing branches, tags, commits, checkpoints, logs, or experiment results.
- After editing, report every modified, added, and deleted file.
- Keep each task focused on one clearly defined algorithmic change.

## Runtime environment

- The required Conda environment is `bbf-sac-dreamerv3`.
- Do not use the Conda `base` environment for project tests.
- Prefer explicit commands such as:

  `conda run -n bbf-sac-dreamerv3 python ...`

- For Python modules, prefer:

  `conda run -n bbf-sac-dreamerv3 python -m ...`

- Before running project code, verify the environment with:

  `conda run -n bbf-sac-dreamerv3 python --version`

- Verify PyTorch and CUDA with:

  `conda run -n bbf-sac-dreamerv3 python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"`

- Do not install, remove, or upgrade Python, Conda, CUDA, PyTorch, JAX,
  Atari, or other dependencies without explicit permission.
- Do not modify the Conda environment merely to make a test pass.
- If an environment dependency is missing, report it instead of silently installing it.

## Code-change rules

- Analyze the existing implementation before editing.
- Explain the planned changes before modifying files.
- Make the smallest change required for the current task.
- Do not modify unrelated files.
- Preserve the original default behavior unless the task explicitly requests otherwise.
- Experimental behavior must be controlled through configuration rather than hard-coded.
- Reuse the existing configuration system and coding conventions.
- Avoid unnecessary renaming, formatting, file movement, or large-scale refactoring.
- Do not remove existing compatibility paths unless explicitly requested.
- Clearly explain all behavioral changes and configuration defaults.

## Reinforcement-learning requirements

- Do not silently change random seeds.
- Do not silently change the Atari game.
- Do not silently change the training frame budget.
- Do not silently change replay ratio, batch size, update frequency,
  evaluation episodes, or evaluation protocol.
- Do not silently change reward processing, action repeat, sticky actions,
  terminal handling, or observation preprocessing.
- Clearly distinguish:
  - online networks;
  - target networks;
  - behavior policies;
  - evaluation policies.
- Check tensor shapes and broadcasting explicitly.
- Check gradient flow and stop-gradient boundaries explicitly.
- Check device placement and dtype consistency.
- Check numerical stability, including logarithms, probabilities, entropy,
  distributional supports, and clipping.
- Do not claim an algorithmic improvement based on a single game or seed.
- Compare experiments only when frame budgets and evaluation protocols match.

## Exact discrete SAC requirements

When working on the exact discrete SAC experiment, explicitly verify:

- whether the actor objective uses sampled actions or an exact expectation
  over all discrete actions;
- whether the critic target is hard or entropy-regularized;
- whether actor and critic objectives are mathematically consistent;
- whether `_log_alpha` receives the intended gradients;
- whether the alpha-loss sign is correct;
- whether target entropy is defined appropriately for a discrete action space;
- whether learned alpha and fixed entropy scheduling can be selected by configuration;
- whether target-network outputs are detached from gradient computation;
- whether entropy shifts interact safely with the C51 support;
- whether distributional projection loses mass at support boundaries;
- whether the default configuration reproduces the original implementation.

## Testing rules

- Inspect the repository before choosing test commands.
- Use existing tests when they are available.
- Do not invent a nonexistent test suite or claim that tests passed when they did not run.
- Run relevant import checks, configuration checks, unit tests, and minimal smoke tests.
- Short GPU smoke tests are allowed.
- Before using a GPU, report the selected device and relevant command.
- Do not start a complete Atari 100K training run unless explicitly requested.
- Do not start long-running or multi-seed experiments without explicit permission.
- Do not occupy a different GPU from the one explicitly assigned.
- Report every executed command and its exit status.
- Clearly identify tests that failed, were skipped, or could not be run.
- Never describe an unexecuted test as successful.

## Experiment reproducibility

Before a full experiment, report and record:

- Git branch;
- full Git commit hash;
- Conda environment;
- Python version;
- framework and CUDA versions;
- GPU model;
- complete training command;
- configuration files and overrides;
- Atari game;
- seed;
- frame budget;
- evaluation protocol.

Use:

`git rev-parse HEAD`

to identify the exact code version used for an experiment.

Do not begin a full experiment if relevant source-code changes remain uncommitted,
unless the user explicitly approves running from a dirty working tree.

## Generated and large files

Do not add to Git or delete without explicit permission:

- checkpoints;
- model weights;
- replay buffers;
- datasets;
- ROM files;
- full training logs;
- TensorBoard or WandB run directories;
- core dumps;
- cache directories;
- temporary files;
- environment credentials;
- proxy credentials;
- API keys;
- SSH keys;
- Codex authentication files.

Examples include:

- `*.ckpt`
- `*.pt`
- `*.pth`
- `*.pkl`
- `*.npz`
- `core.*`
- `__pycache__/`
- `.pytest_cache/`
- `wandb/`
- `runs/`
- `logs/`
- `checkpoints/`
- `replay_buffers/`

Small experiment summaries, configuration files, CSV/JSON metrics,
analysis scripts, and Markdown reports may be committed after review.

## Shell-script safety

- Inspect `run-cuda0.sh` and `run-cuda1.sh` before changing or running them.
- Do not overwrite existing experiment output directories.
- Do not terminate unrelated training processes.
- Do not delete another experiment's logs or checkpoints.
- Do not run destructive shell commands.
- Ask before changing GPU assignment or output paths.

## Completion report

At the end of every coding task, provide:

1. current Git branch;
2. concise summary of the implementation;
3. list of changed files;
4. important mathematical or behavioral decisions;
5. configuration options and defaults;
6. commands executed;
7. test and smoke-test results;
8. unresolved risks or assumptions;
9. recommended server experiment;
10. confirmation that no commit or push was performed unless explicitly requested.
