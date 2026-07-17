# AGENTS.md

## Project purpose

This repository is used for reinforcement-learning research based on
bigger-better-faster-SAC.

The goal is to discover, implement, and evaluate algorithmic improvements
while preserving reproducibility, fair comparison, and clear Git history.

## Git workflow

- Work only on the current `exp/*` experiment branch.
- Never modify `research-main` or `bbf-starting-point-claude` directly.
- Before editing, run `git branch --show-current` and report the current branch.
- Do not switch branches unless explicitly requested.
- Do not run `git commit` or `git push` unless explicitly requested.
- Do not run `git reset`, `git clean`, `git rebase`, force push, or other
  history-rewriting commands.
- Do not delete existing branches, tags, commits, checkpoints, logs, or
  experiment results.
- Keep each experiment focused on one clearly defined improvement.
- At the end of a task, report every modified, added, and deleted file.

## Runtime environment

- The required Conda environment is `bbf-sac-dreamerv3`.
- Do not use the Conda `base` environment for project testing.
- Prefer explicit commands such as:

  `conda run -n bbf-sac-dreamerv3 python ...`

- For Python modules and tests, prefer:

  `conda run -n bbf-sac-dreamerv3 python -m ...`

- Before running project code, verify the environment with:

  `conda run -n bbf-sac-dreamerv3 python --version`

- Verify the main framework and CUDA availability before GPU testing.
- Do not install, remove, or upgrade Python, Conda, CUDA, PyTorch, JAX,
  Atari, or other dependencies without explicit permission.
- If a dependency is missing or incompatible, report the issue rather than
  silently changing the environment.

## Selecting an optimization direction

When asked to find and implement an optimization without a specified direction:

1. Inspect the current implementation, configuration, Git history, existing
   documentation, and available experiment records.
2. Identify at least three technically distinct candidate improvements.
3. For each candidate, briefly evaluate:
   - theoretical motivation;
   - affected files and modules;
   - implementation complexity;
   - expected benefit;
   - major risks;
   - required tests and experiments;
   - whether it overlaps with an earlier attempt.
4. Rank the candidates using:
   - technical soundness;
   - compatibility with the existing codebase;
   - expected research value;
   - feasibility under the available compute budget;
   - ability to isolate the change in an ablation study.
5. Select one candidate for the current experiment.
6. Explain the selected direction and the planned files before editing.
7. Implement only the selected direction during the current task.
8. Do not combine multiple major algorithmic changes in one experiment.
9. Do not repeat a previously unsuccessful direction without explaining what
   has changed and why another attempt is justified.
10. Do not claim novelty without checking relevant literature or being
    explicitly asked to perform a literature search.

If the user requests analysis only, stop after proposing and ranking the
candidate directions without modifying files.

## Code-change rules

- Analyze the existing implementation before editing.
- Explain the planned changes before modifying files.
- Make the smallest coherent change required for the selected direction.
- Do not modify unrelated files.
- Avoid unnecessary renaming, formatting, file movement, or broad refactoring.
- Preserve the original behavior by default whenever practical.
- New experimental behavior should be configurable rather than hard-coded.
- Reuse the existing configuration system and project conventions.
- Preserve an accessible baseline path for ablation experiments.
- Clearly explain changes in behavior, mathematical objectives, and defaults.
- Do not remove existing compatibility paths unless explicitly requested.

## Reinforcement-learning requirements

- Do not silently change random seeds.
- Do not silently change the Atari game or environment.
- Do not silently change the training frame budget.
- Do not silently change replay ratio, batch size, update frequency,
  evaluation episodes, or evaluation protocol.
- Do not silently change reward processing, action repeat, sticky actions,
  terminal handling, or observation preprocessing.
- Clearly distinguish:
  - online networks;
  - target networks;
  - behavior policies;
  - training policies;
  - evaluation policies.
- Check tensor shapes and broadcasting explicitly.
- Check gradient flow and stop-gradient boundaries explicitly.
- Check device placement and dtype consistency.
- Check numerical stability, probability normalization, clipping, and
  distributional projections where applicable.
- Ensure that new objectives are mathematically consistent with the update
  rules they interact with.
- Keep the baseline configuration available for controlled comparison.
- Do not claim an improvement based on one game, one seed, or one training run.
- Compare results only when frame budgets and evaluation protocols match.

## Evaluation and ablation requirements

For each selected optimization:

- Define a baseline configuration and an improved configuration.
- Identify the smallest ablation that isolates the proposed change.
- State which metrics should reveal whether the change behaves as intended.
- Record both performance metrics and relevant diagnostic metrics.
- Consider runtime, memory use, training stability, and implementation cost.
- Distinguish code correctness from empirical performance.
- A successful smoke test does not demonstrate an algorithmic improvement.
- A higher score from one run is preliminary evidence, not a final conclusion.
- Recommend multi-seed evaluation before making a strong performance claim.

For aggregate Atari evaluation, preserve the same normalization, game set,
seed count, frame budget, and confidence-interval procedure across methods.

## Testing rules

- Inspect the repository before selecting test commands.
- Use existing tests when available.
- Do not invent a nonexistent test suite.
- Never claim that an unexecuted test passed.
- Run relevant:
  - syntax and import checks;
  - configuration parsing checks;
  - unit tests;
  - tensor-shape checks;
  - gradient-flow checks;
  - model initialization checks;
  - short CPU or GPU smoke tests.
- Before using a GPU, report the selected device and command.
- Do not start a complete Atari 100K run unless explicitly requested.
- Do not start long-running or multi-seed experiments without explicit
  permission.
- Do not use a GPU different from the one explicitly assigned.
- Report every executed command and its exit status.
- Clearly identify tests that failed, were skipped, timed out, or could not run.

## Experiment reproducibility

Before a full experiment, report and record:

- Git branch;
- full Git commit hash;
- whether the working tree is clean;
- Conda environment;
- Python version;
- framework and CUDA versions;
- GPU model;
- complete training command;
- configuration files and command-line overrides;
- Atari game;
- seed;
- frame budget;
- evaluation protocol;
- output directory.

Use:

`git rev-parse HEAD`

to identify the exact code version.

Use:

`git status --short`

to check whether uncommitted changes exist.

Do not begin a full experiment from a dirty working tree unless the user
explicitly approves it and the exact diff is archived.

## Generated and large files

Do not add to Git or delete without explicit permission:

- checkpoints;
- model weights;
- replay buffers;
- datasets;
- ROM files;
- full training logs;
- TensorBoard or WandB directories;
- core dumps;
- caches;
- temporary files;
- API keys;
- proxy credentials;
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

Small experiment summaries, configuration files, CSV or JSON metrics,
analysis scripts, plots, and Markdown reports may be committed after review.

## Shell and server safety

- Inspect shell scripts before modifying or running them.
- Do not overwrite an existing experiment output directory.
- Do not terminate unrelated training processes.
- Do not delete another experiment's logs, checkpoints, or replay data.
- Do not run destructive shell commands.
- Ask before changing GPU assignments or output locations.
- Do not launch background jobs without reporting how to inspect and stop them.
- Before launching a training process, check for existing GPU workloads when
  practical.

## Completion report

At the end of every analysis or coding task, provide:

1. current Git branch;
2. selected optimization direction and why it was selected;
3. alternative directions considered;
4. concise implementation summary;
5. list of changed files;
6. important mathematical and behavioral decisions;
7. configuration options and defaults;
8. commands executed;
9. test and smoke-test results;
10. unresolved risks and assumptions;
11. recommended ablation and server experiment;
12. confirmation of whether any commit, push, dependency change, or long
    training run was performed.
