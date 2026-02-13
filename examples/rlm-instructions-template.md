# RLM Refactoring Instructions

## Goal

<!-- Describe the refactoring task clearly. What should change and why? -->

Rename all instances of `OldClassName` to `NewClassName` across the codebase,
updating imports, type annotations, docstrings, and tests.

## Repositories

<!-- List the repositories available in /repos and what each contains. -->

- `my-service/` — Main application service (Python, FastAPI)
- `my-client-lib/` — Client library that depends on my-service's types

## Constraints

<!-- Any rules the agent must follow. -->

- Do not modify files in `vendor/` or `node_modules/`
- Preserve all existing test coverage — tests must still pass
- Do not change any public API signatures beyond the rename
- Keep backwards-compatible type aliases where the old name was part of a public API

## Commit Strategy

<!-- How should changes be committed? -->

- One commit per repository
- Commit message format: `refactor: rename OldClassName to NewClassName`
- Do not push — changes will be reviewed manually

## Verification

<!-- How should the agent verify its work? -->

- Run `python -m pytest` in each repo after changes
- Run `grep -rn OldClassName /repos` to confirm no remaining references
- Check `git diff --stat` to review scope of changes
