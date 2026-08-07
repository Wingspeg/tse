# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-06

### Added
- Initial public release.
- `pef` Python package with `src/pef/` layout.
- Unified Typer-based CLI: `pef train`, `pef ablation`, `pef extras`, `pef viz`,
  `pef viz-multi`, `pef pipeline`.
- Federated simulation (FedAvg + Dirichlet non-IID partitioning) on
  CIFAR-10, MNIST, Fashion-MNIST.
- Trusted representation generator (lattice-based commitment + RS-encoded ID).
- Shapley-guided adaptive noise allocation on three resource axes
  (algorithm, compute, data).
- Collaborative folding protocol for O(1) per-aggregation proof verification.
- Supplementary experiments: scalability, privacy-utility sweep, linkage
  attack resistance, baseline comparison.
- Publication-style figure generation (PDF + 300 DPI PNG).
- GitHub Actions CI: ruff lint + smoke test on Python 3.11/3.12/3.13.
- English and Chinese READMEs.
- MIT license.
