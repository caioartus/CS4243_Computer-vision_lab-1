"""Per-material diagonal normal model and dense anomaly prediction."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import NormalityConfig
from .supplied import box_mean


@dataclass
class NormalModel:
    mean: np.ndarray
    std: np.ndarray
    threshold: float
    feature_names: list[str] | None = None


def _scores(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray, epsilon: float
) -> np.ndarray:
    return np.sqrt(np.mean(((features - mean) / (std + epsilon)) ** 2, axis=-1)).astype(
        np.float32
    )


def _pool_score(score: np.ndarray, config: NormalityConfig) -> np.ndarray:
    """Suppress isolated patch responses while preserving coherent anomalies."""
    if config.score_pool_size < 1 or config.score_pool_size % 2 == 0:
        raise ValueError("score_pool_size must be a positive odd integer")
    return box_mean(score, config.score_pool_size).astype(np.float32)


def _interior(score: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return score
    if 2 * border >= min(score.shape):
        raise ValueError("ignore_border is too large for the score map")
    return score[border:-border, border:-border]


def fit_normal_model(
    feature_maps: list[np.ndarray] | np.ndarray,
    config: NormalityConfig,
    feature_names: list[str] | None = None,
) -> NormalModel:
    """Fit normal feature statistics using normal training maps only."""
    maps = (
        [feature_maps] if isinstance(feature_maps, np.ndarray) else list(feature_maps)
    )
    if not maps:
        raise ValueError("feature_maps must contain at least one map")

    feature_dim = maps[0].shape[-1]

    flattened = [m.reshape(-1, feature_dim) for m in maps]
    samples = np.concatenate(flattened, axis=0)
    if samples.shape[0] > config.max_samples:
        rng = np.random.default_rng(config.random_seed)
        indices = rng.choice(samples.shape[0], size=config.max_samples, replace=False)
        samples = samples[indices]

    mean = samples.mean(axis=0)
    std = samples.std(axis=0)

    if config.mask_threshold is not None:
        threshold = config.mask_threshold
    else:
        interior_scores = [
            _interior(
                _pool_score(_scores(m, mean, std, config.epsilon), config),
                config.ignore_border,
            )
            for m in maps
        ]
        pooled = np.concatenate([s.reshape(-1) for s in interior_scores])
        threshold = np.percentile(pooled, config.threshold_percentile)

    return NormalModel(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        threshold=float(threshold),
        feature_names=feature_names,
    )


def predict_anomaly(
    feature_map: np.ndarray, model: NormalModel, config: NormalityConfig
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return dense score, image-level score, and predicted mask."""
    #
    # TODO 1 — Validate feature compatibility
    #   - The final feature dimension must equal the fitted model mean length.
    #
    # TODO 2 — Compute the dense anomaly score
    #   - Use _scores for RMS standardized distance with config.epsilon.
    #   - Smooth coherent evidence with _pool_score.
    #
    # TODO 3 — Aggregate an image-level score
    #   - Crop the unreliable border with _interior and take
    #     config.image_percentile over that interior only.
    #
    # TODO 4 — Produce the binary localization mask
    #   - Threshold the full score map at model.threshold.
    #   - Force the configured outer border to False without altering scores.
    #   - Return (float32 score map, Python-float image score, Boolean mask).
    if feature_map.shape[-1] != model.mean.shape[-1]:
        raise ValueError("feature_map's feature dimension does not match the model")

    score = _scores(feature_map, model.mean, model.std, config.epsilon)
    score = _pool_score(score, config)

    image_score = float(
        np.percentile(_interior(score, config.ignore_border), config.image_percentile)
    )

    mask = score >= model.threshold
    border = config.ignore_border
    if border > 0:
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False

    return score.astype(np.float32), image_score, mask.astype(bool)


def select_mask_threshold(
    score_maps: list[np.ndarray], masks: list[np.ndarray], candidates: int = 80
) -> float:
    """Choose the pixel-F1-optimal threshold on public validation only."""
    #
    # TODO 1 — Validate the validation inputs
    #   - Require matching non-empty score/mask collections and candidates >= 2.
    #   - Convert scores to float and masks to bool; require matching shapes and
    #     finite scores for every pair.
    #   - Callers handle border cropping, so do not crop again here.
    if not score_maps or not masks:
        raise ValueError("score_maps and masks must be non-empty")
    if len(score_maps) != len(masks):
        raise ValueError("score_maps and masks must have the same length")
    if candidates < 2:
        raise ValueError("candidates must be >= 2")

    score_maps = [np.asarray(s, dtype=float) for s in score_maps]
    masks = [np.asarray(m, dtype=bool) for m in masks]
    for s, m in zip(score_maps, masks):
        if s.shape != m.shape:
            raise ValueError("each score map must match its mask's shape")
        if not np.all(np.isfinite(s)):
            raise ValueError("score maps must contain only finite values")

    # TODO 2 — Build candidate thresholds
    #   - Flatten and concatenate all validation pixels.
    #   - Use evenly spaced quantile levels from 0.5 through 0.999.

    flat_scores = np.concatenate([s.ravel() for s in score_maps])
    flat_masks = np.concatenate([m.ravel() for m in masks])

    levels = np.linspace(0.5, 0.999, candidates)
    thresholds = np.quantile(flat_scores, levels)

    # TODO 3 — Select by pixel F1
    #   - For each threshold, predict score >= threshold and compute TP/FP/FN.
    #   - Use 2*TP / max(1, 2*TP + FP + FN) to avoid division by zero.
    #   - Keep the threshold with the greatest F1; deterministic ties should
    #     retain the first encountered candidate.
    #   - Return the selected threshold as a Python float.

    F1_list = []
    for thresh in thresholds:
        predicted = flat_scores >= thresh
        TP = (predicted & flat_masks).sum()
        FP = (predicted & ~flat_masks).sum()
        FN = (~predicted & flat_masks).sum()
        F1 = 2 * TP / max(1, 2 * TP + FP + FN)
        F1_list.append(F1)
    best = int(np.argmax(F1_list))  # if tie returns first

    return float(thresholds[best])
