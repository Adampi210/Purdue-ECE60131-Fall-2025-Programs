# ECE 60131 - Generative Models (Fall 2025) Programs

This repository contains programming assignments, reference implementations, and experiment scripts for the course on generative models. Generative models provide:
- Sample generation from learned data distributions
- Principled handling of missing/latent variables
- (Sometimes) improved data efficiency over purely discriminative approaches

Trade-off: Improving generative fidelity can degrade downstream inference (and vice versa). The course explores algorithms that navigate this tension.

## Core Method Families Covered

1. Exact inference in directed graphical models (Expectation-Maximization, EM)
2. Sampling-based approximate inference (MCMC, Gibbs, Metropolis-Hastings)
3. Deterministic approximate inference (Variational EM / Variational Inference)
4. Energy-based latent variable models (RBM-like architectures, Contrastive Divergence)
5. Adversarial training (Generative Adversarial Networks, GANs)

## Repository Layout (planned)

```
/data/                # Sample datasets or download scripts (gitignored when large)
/common/              # Shared utilities (logging, metrics, plotting, schedulers)
/em/                  # EM algorithm assignments & examples
/sampling/            # MCMC samplers and experiments
/variational/         # Variational inference (ELBO optimization) code
/rbm/                 # RBM & contrastive divergence implementations
/gan/                 # GAN models, training loops, evaluation metrics
/experiments/         # Configuration + result tracking
/reports/             # Optional writeups / result summaries
requirements.txt
README.md
```

## Module Summaries

- EM: Gaussian mixtures, latent class models, convergence diagnostics.
- Sampling: Gibbs vs Metropolis trade-offs, mixing behavior, autocorrelation.
- Variational: Mean-field factorization, ELBO derivations, amortized inference.
- RBM / CD: Energy function design, negative phase estimation, mode coverage issues.
- GAN: Minimax objective, instability sources, evaluation (FID, Inception Score limits).

## Planned Assignments (tentative)

| Module | Task | Key Skills |
|--------|------|-----------|
| EM | Implement GMM EM + compare to k-means | Closed-form updates, log-likelihood tracking |
| Sampling | Build Gibbs sampler for Ising / topic model | Mixing analysis, burn-in tuning |
| Variational | Derive & code mean-field VI for a latent model | ELBO gradients, reparameterization (if applicable) |
| RBM | Train RBM on MNIST with CD-k | Energy-based modeling, sampling chains |
| GAN | Implement DCGAN & explore failure modes | Adversarial objectives, stability tricks |

## Getting Started

1. Clone: git clone <repo-url>
2. Create environment: python -m venv .venv && source .venv/Scripts/activate (Windows adjust accordingly)
3. Install: pip install -r requirements.txt
4. Run a module demo: python em/train_gmm.py --config experiments/gmm_baseline.yaml

## Evaluation & Reproducibility

- All experiments configured via YAML under /experiments
- Set RNG seeds for numpy, torch, and CUDA
- Save: model checkpoints, metrics JSON, sample grids (GAN/RBM)

## Prerequisites

- Linear algebra, probability (latent variable models, conditional independence)
- Python (3.10+), NumPy, PyTorch
- Basic optimization (gradient methods)

## References (starter set)

- Bishop, Pattern Recognition and Machine Learning
- Murphy, Probabilistic Machine Learning
- Goodfellow et al., GAN original paper
- Hinton, Training Products of Experts / RBM papers
- Kingma & Welling, VAE paper (for variational module extension)

## Contribution

Each directory will contain:
- README with task goals
- main training script
- module-specific utils
- (Optional) notebook for exploratory analysis

Issues / TODO will track incremental enhancements.

## License

TBD (add before public release).

---

Prepare to fill in code as modules are introduced in lectures.