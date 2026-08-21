"""Pure policy helpers for normalization-by-state-ratio baselines."""

from __future__ import annotations


DECODER_NORMALIZATION_CHOICES = ("batchnorm", "groupnorm")


def resolve_full_probability(run: str, requested: float | None) -> float:
    """Return the frozen Full-batch probability for a named baseline arm."""

    if run not in {"A", "B", "C"}:
        raise ValueError(f"unknown missing-modality run: {run!r}")
    default = {"A": 0.0, "B": 1.0, "C": 0.5}[run]
    probability = default if requested is None else float(requested)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("--full-probability must be in [0, 1]")
    if run == "A" and probability != 0.0:
        raise ValueError("Run A is the p(Full)=0 SAR-only control")
    if run == "B" and probability != 1.0:
        raise ValueError("Run B is the p(Full)=1 Full-only control")
    if run == "C" and not 0.0 < probability < 1.0:
        raise ValueError("Run C requires 0 < p(Full) < 1; use A/B at endpoints")
    return probability


def train_state(
    run: str,
    state_rng,
    full_probability: float | None = None,
) -> str:
    """Sample one homogeneous batch state.

    The comparison is deliberately written as ``random < 1-p`` so the default
    Run-C p=0.5 trace remains identical to the released V1 launcher.
    """

    probability = resolve_full_probability(run, full_probability)
    if probability == 0.0:
        return "sar"
    if probability == 1.0:
        return "full"
    return "sar" if state_rng.random() < 1.0 - probability else "full"


def experiment_slug(
    run: str,
    normalization: str,
    full_probability: float | None,
    seed: int,
) -> str:
    if normalization not in DECODER_NORMALIZATION_CHOICES:
        raise ValueError(f"unknown decoder normalization: {normalization!r}")
    probability = resolve_full_probability(run, full_probability)
    percent = int(round(probability * 100))
    if abs(probability * 100 - percent) > 1e-9:
        token = f"{probability:.4f}".replace(".", "p")
    else:
        token = f"{percent:03d}"
    norm_token = {"batchnorm": "bn", "groupnorm": "gn"}[normalization]
    return f"run_{run.lower()}_{norm_token}_p{token}_seed{int(seed)}"
