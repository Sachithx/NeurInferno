# Changelog

All notable user-facing changes are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Automated package, lint, and test checks.
- Structured contribution, conduct, security, and issue-reporting guidance.

## [0.2.0] - 2026-08-21

### Added

- Responsive Hugging Face interface with structured segments, confidence
  views, validation feedback, and JSON output.
- JSON output and version reporting in the command-line interface.
- Explicit truncation metadata in inference results.
- Python 3.10–3.13 test coverage and automated package publishing support.

### Changed

- Checkpoints now use restricted PyTorch deserialization.
- Invalid hexadecimal characters and oversized batches are rejected instead
  of being silently discarded or truncated.
- The Hugging Face Space installs the versioned package from PyPI rather than
  bundling a duplicate source tree.

## [0.1.0] - 2026-08-20

- Initial package, command-line interface, model loading, training utilities,
  evaluation utilities, and CPU demo.

[Unreleased]: https://github.com/Sachithx/NeurInferno/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Sachithx/NeurInferno/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Sachithx/NeurInferno/releases/tag/v0.1.0
