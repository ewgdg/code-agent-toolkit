# Repository Guidelines

## Project Structure & Module Organization

- `src/claude_router/`: FastAPI router, adapters, config, entrypoint.
- `config/`: runtime configuration (e.g., `router.yaml`).
- `tests/`: unit/integration tests (PyTest).
- `tasks/` and `shared/`: shared utilities and task helpers.
- `docs/`: contributor docs (this file). Keep new docs here.

## Build, Test, and Development Commands

- Install deps: `uv sync` (creates/updates the environment from `pyproject.toml` + `uv.lock`).
- Run server: `uv run claude-router --config config/router.yaml` or `uv run python -m claude_router.server`.
- Run locally with auto‑reload (via config hot‑reload): edit `config/router.yaml`; server restarts automatically.
- Tests: `uv run pytest -q` or with coverage `uv run pytest --cov=src --cov-report=term-missing`.
- Lint: `uv run ruff check src tests`.
- Format: `uv run ruff format`.
- Types: `uv run mypy src` and `uv run pyright`.

## Coding Style & Naming Conventions

- Python `3.13`, line length `88`, double quotes; enforced by Ruff (see `pyproject.toml`).
- Modules: lowercase with underscores; classes `PascalCase`; functions/vars `snake_case`.
- Keep public APIs typed; prefer `pydantic` models for request/response schemas.

## Testing Guidelines

- Framework: PyTest with `pytest-asyncio`; place tests under `tests/` mirroring package paths.
- Name tests `test_*.py`, functions `test_*`.
- Provide coverage for new logic; prefer fast, isolated tests. Use `pytest.mark.asyncio` for async.

## Commit & Pull Request Guidelines

- Follow Conventional Commits: `feat:`, `fix:`, `refactor:`, `perf:`, `chore:`, `doc:` (matches repo history).
- PRs must include: clear summary, linked issues, test coverage notes, and before/after evidence for behavior changes (logs or screenshots).

## Security & Configuration Tips

- Secrets via `.env` (loaded at startup); never commit secrets.

## Agent‑Specific Notes

- Use latest library versions (managed by `uv`).
- When adding dependencies, update `pyproject.toml` and run `uv sync`.
- Consult official docs via Context7 before introducing new frameworks or APIs.
