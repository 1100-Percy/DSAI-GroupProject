# Global rules for Codex

## Operating principles
- Prefer small, reviewable diffs. Avoid sweeping refactors unless explicitly requested.
- Before editing, identify the files to change and state the plan in 3–6 bullets.
- Never invent APIs, configs, or file paths. If unsure, search the repo first.
- Keep changes consistent with existing style and architecture.

## Safety and secrets
- Never paste secrets, tokens, private keys, .env values, or credentials into code or logs.
- If a task requires secrets, ask the user to provide them via environment variables.
- Do not add analytics, telemetry, or network calls unless asked. The task's explicit dependency and pretrained resource downloads are within its stated scope.

## Code quality
- Add or update tests for behavior changes when the project has tests.
- Prefer type safety and explicit error handling.
- Add comments only when the intent is non-obvious.

## Build and run
- Before running commands, propose the exact command and explain why.
- When changes may break the build, run the fastest relevant check first.

## Output
- For code changes, include a short summary and list of changed files.
- For debugging, include hypotheses, experiments run, and the minimal fix.
- Default explanation language: Chinese.

## User's explanation preferences
- Start with the overall framework, background, and ultimate goal before details.
- Organize explanations as a clear trunk–branch–leaf knowledge tree.
- Prefer compact tables, outlines, visual scenarios, and step-by-step reasoning over dense prose.
- Explain underlying mechanisms and causes with rigorous logical connections.

## Stage 1 repository scope
- Read `STAGE1_TASK.md` before implementing the experiment.
- Treat it as the requested experimental specification; preserve the original file.
- `DATA_DIR` is `data/DSAI4205_unified_cleaning_seed42_v3_stemming` relative to the repository root.
- Treat all dataset files as immutable. Use `Phrase` exactly as stored.
- Do not read test data or audit files, including copies inside the archived ZIP.
- Use only the allowed training/fold inputs; use the original `normalize_input` function only for GloVe vocabulary alignment.
- Do not regenerate the dataset, introduce external experimental data, fine-tune BERTweet, or start Stage 2.
- Record unspecified decisions and protocol conflicts in `reports/stage1/OPEN_QUESTIONS.md`.
- Do not claim environment validation, execution, scores, or reproducibility without actual evidence.
