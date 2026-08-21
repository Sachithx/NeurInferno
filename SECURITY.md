# Security policy

## Supported versions

Security fixes are applied to the latest released minor version.

| Version | Supported |
|---|---|
| 0.2.x | Yes |
| 0.1.x | No |

## Report a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private
security advisory form:

<https://github.com/Sachithx/NeurInferno/security/advisories/new>

Include the affected version, a minimal reproduction, potential impact, and
any suggested mitigation. Remove credentials and sensitive packet contents.

## Model and input safety

- Download model artifacts only from the repository linked in `README.md` or
  from a source you trust.
- NeurInferno uses restricted PyTorch deserialization for checkpoints.
- Treat protocol messages as potentially sensitive. Do not submit private
  traffic to a hosted demo.
- Review inferred boundaries before using them in security-sensitive tooling.

Inference results are predictions and must not be treated as validated protocol
specifications.
