# Contributing to NeurInferno

Thank you for improving NeurInferno. Focused bug fixes, tests, documentation,
performance improvements, and well-scoped features are welcome.

## Development setup

```bash
git clone https://github.com/Sachithx/NeurInferno.git
cd NeurInferno
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,demo]"
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

Run the checks used by CI:

```bash
ruff check .
ruff format --check .
pytest
python -m build
twine check dist/*
```

Use `ruff format .` to apply the project's formatting rules.

## Pull requests

- Open an issue before starting a large or behavior-changing feature.
- Keep each pull request focused on one concern.
- Add regression tests for bug fixes and tests for new behavior.
- Update public documentation when an API or command changes.
- Avoid unrelated formatting or generated-file changes.
- Do not commit model checkpoints, credentials, private network traffic, or
  full generated datasets.

Small sample messages used in tests must be synthetic or already public and
must not contain identifying or sensitive information.

## Compatibility

NeurInferno supports Python 3.10 and newer. Public API changes should preserve
backward compatibility where practical. If a breaking change is necessary,
describe it clearly in the pull request and add it to `CHANGELOG.md`.

## Reporting bugs

Use the structured bug report template and include a minimal reproduction.
Sanitize all message bytes, file paths, environment variables, and logs before
posting them publicly.

By participating, you agree to follow the project's `CODE_OF_CONDUCT.md`.
