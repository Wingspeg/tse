# Privacy-Enhanced Federated Learning Framework

A research framework that combines **Shapley-guided adaptive differential privacy**
with **lattice-based collaborative folding proofs** for verifiable federated
training. The framework is built around three privacy-sensitive resource
axes — *algorithm*, *compute*, and *data* — and ships a full experiment suite
covering accuracy, privacy–utility tradeoff, scalability, and linkage attack
resistance.

> **Status:** active research code. Cryptographic primitives use simplified
> lattice constructions suitable for empirical comparison; the framework is
> **not** intended for production security without further hardening. See
> [Caveats & Honest Notes](#caveats--honest-notes) for details.

---

## Highlights

- **Shapley-guided noise allocation.** Three-way Rényi mutual information
  decomposition on the (algorithm, compute, data) representations yields a
  Shapley allocation that drives per-resource Gaussian noise calibration.
  Higher-Shapley axes get tighter privacy budgets (smaller ε, larger σ).
- **Collaborative folding proof.** Each participant commits to its enhanced
  representation via a lattice (SIS) commitment; the aggregator folds them
  into a single 32-byte proof that verifies in O(1) — empirically **~30×**
  faster and **~10×** smaller than naive per-party verification.
- **Federated simulation out of the box.** FedAvg with Dirichlet non-IID
  partitioning on CIFAR-10, MNIST, and Fashion-MNIST, end-to-end runnable
  in a few minutes on a single GPU.
- **Reproducible experiments + figures.** One CLI produces the full set of
  tables (JSON) and publication-style figures (PDF + 300 DPI PNG).

## Results snapshot

| Dataset        | Final Acc | Final MI | Verified | Avg Round |
|----------------|-----------|----------|----------|-----------|
| CIFAR-10       | 75.4%     | 0.47     | ✓        | 32 s      |
| MNIST          | 99.4%     | 0.57     | ✓        | 24 s      |
| Fashion-MNIST  | 92.1%     | 0.43     | ✓        | 24 s      |

*Ablation: noise savings 1.9% over uniform; folding 30× faster verification.*

---

## Installation

### Recommended (uv)

```bash
git clone https://github.com/your-org/privacy_enhanced_framework
cd privacy_enhanced_framework
uv sync --extra dev
source .venv/bin/activate
```

### Plain pip + venv

```bash
git clone https://github.com/your-org/privacy_enhanced_framework
cd privacy_enhanced_framework
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### CPU-only (no CUDA)

```bash
pip install -e ".[dev,cpu]"
# or, after `uv sync`:
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

> Tested on Python 3.11 / 3.12 / 3.13. macOS (Apple Silicon), Linux
> (CUDA 12.x) verified.

---

## Quickstart

```bash
# 1. Run a single federated training experiment (downloads CIFAR-10 on
#    first run).
pef train --dataset cifar10

# 2. Run all three datasets and print a summary.
pef train --all

# 3. Run the supplementary experiments (scalability, privacy–utility,
#    linkage, baselines).
pef extras

# 4. Generate all publication figures.
pef viz single
pef viz multi    # equivalent: pef viz-multi

# 5. Full pipeline (training + extras + figures) — what a CI run does.
pef pipeline
```

> Tip: the flat alias `pef viz-multi` is also accepted as a shortcut for
> `pef viz multi`.

All commands accept `--results-dir results` (default) and write JSON
metadata plus PDF/PNG figures under `results/`.

## CLI reference

| Command                       | What it does                                            |
|-------------------------------|---------------------------------------------------------|
| `pef train --dataset NAME`    | Federated training on one dataset                       |
| `pef train --all`             | Train on CIFAR-10, MNIST, Fashion-MNIST                 |
| `pef ablation`                | 4-config ablation: full / −Shapley / −Folding / −Both   |
| `pef extras`                  | Scalability, privacy-utility, linkage, baselines        |
| `pef viz single`              | Single-dataset figures                                  |
| `pef viz multi` / `pef viz-multi` | Cross-dataset figures + summary                    |
| `pef pipeline`                | `train --all` + `extras` + `viz single` + `viz multi`   |
| `pef version`                 | Print the installed version                             |
| `pef --help`                  | Discoverable help (Typer auto-docs)                     |

Each subcommand has `--help`:

```bash
pef train --help
pef ablation --help
```

## Project layout

```
privacy_enhanced_framework/
├── pyproject.toml          # build + deps + ruff + pytest + mypy
├── README.md / README.zh.md
├── LICENSE                 # MIT
├── CHANGELOG.md
├── .github/workflows/ci.yml
├── src/pef/                # the actual package
│   ├── cli.py              # Typer entry point
│   ├── config.py           # lattice / DP / FL hyper-params
│   ├── lattice_utils.py    # SIS hash, Merkle tree, sampler
│   ├── representation.py   # trusted 3-axis representation
│   ├── renyi_mutual_info.py# pairwise / three-way Rényi MI estimator
│   ├── shapley_enhancement.py  # adaptive noise calibration
│   ├── collaborative_folding.py# folding proof protocol
│   ├── federated_simulation.py # FedAvg + Dirichlet + main loop
│   ├── experiments_extra.py    # scalability / privacy / linkage
│   ├── ablation.py             # 4-config component ablation
│   ├── visualization.py        # single-dataset figures
│   └── visualization_multi.py  # cross-dataset figures + summary
└── tests/                  # smoke + unit tests (CPU-only)
```

## How the three mechanisms fit together

```
            ┌──────────────────────────────────────────────┐
            │ Per-round federated training (FedAvg)         │
            └────────────────────┬─────────────────────────┘
                                 │
            per client, per round: (φ_a, φ_c, φ_d)
            ALGO  =  algorithm representation  (model Δ)
            COMP  =  compute   representation  (flops/mem/bw)
            DATA  =  data     representation  (sample stats)
                                 │
                                 ▼
            ┌──────────────────────────────────────────────┐
            │ Shapley-guided adaptive noise (3-way Rényi MI)│
            │   sv_a, sv_c, sv_d  →  ε_r  →  σ_r            │
            │ iterate until I_α(φ̃_a; φ̃_c; φ̃_d) ≤ θ         │
            └────────────────────┬─────────────────────────┘
                                 │  enhanced (φ̃_a, φ̃_c, φ̃_d)
                                 ▼
            ┌──────────────────────────────────────────────┐
            │ Collaborative folding proof                   │
            │   SIS-commit each φ̃  →  fold →  32-byte proof│
            │   O(1) verification, ~30× faster than naive    │
            └────────────────────┬─────────────────────────┘
                                 │
                                 ▼
                       global model + verified proof
```

## Configuration

All hyper-parameters live in `src/pef/config.py`. The most commonly
tuned ones:

| Symbol            | Default | Meaning                                    |
|-------------------|---------|--------------------------------------------|
| `Q`               | 12289   | Lattice modulus                            |
| `N_LAT`           | 256     | Lattice dimension                          |
| `ALPHA_RENYI`     | 2.0     | Rényi order                                |
| `THETA`           | 0.5     | MI convergence threshold                   |
| `EPSILON_0`       | 1.0     | Base privacy budget                        |
| `LAMBDA_PARAM`    | 2.0     | Shapley weight in noise calibration        |
| `NUM_PARTICIPANTS`| 10      | Clients                                    |
| `NUM_ROUNDS`      | 20      | Rounds                                     |
| `DIRICHLET_ALPHA` | 0.5     | Non-IID concentration                      |
| `SEED`            | 42      | Global seed                                |

## Development

```bash
# Format + lint
ruff format .
ruff check .

# Tests (CPU only, <30 s on a laptop)
pytest

# Type check
mypy src/pef
```

## Caveats & honest notes

This is a **research artifact**, not a production security library. Two
points to be aware of when reading the code or the paper draft:

1. **Lattice "signature".** `lattice_utils.lattice_sign` is a placeholder
   that derives a deterministic short vector from `SHA256(message)`. It is
   *not* a real SIS-based signature scheme. The verification routine
   (`verify_signature`) only checks norm, not a preimage-style relation.
2. **Rényi MI estimator.** `renyi_mutual_info.estimate_*_renyi_mi` uses a
   closed-form Gaussian approximation (`-½ log(1-ρ²)`) rather than a true
   KSG k-NN estimator. This is fast and reproducible but biased away from
   non-Gaussian structure. The framework is correct *up to this
   approximation* — please disclose it explicitly in the paper text.

These are both known to the authors and will be addressed in a follow-up
release that swaps in a real signature scheme and a kNN-based estimator.

## Citation

If you use this framework in academic work, please cite the accompanying
paper (to be added). A `CITATION.cff` file is included for GitHub's
built-in citation support.

## License

MIT — see [LICENSE](LICENSE).
