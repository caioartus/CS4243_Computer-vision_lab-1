"""Aligned local descriptor shared by all three tasks."""

from __future__ import annotations
from multiprocessing import Value
from xml.etree.ElementInclude import include
import numpy as np
from .config import FeatureConfig
from .edge_branch import detect_edges
from .gabor_branch import gabor_energy_maps, make_gabor_bank
from .supplied import box_mean, rgb2gray


# [AI-CODE][HUMAN-CHECK]
def extract_local_features(
    image: np.ndarray, config: FeatureConfig
) -> tuple[np.ndarray, list[str]]:
    """Return an aligned H x W x D map and one name per channel."""
    if not any(
        [
            config.include_colour,
            config.include_gabor,
            config.include_gradient,
            config.include_edges,
        ]
    ):
        raise ValueError("Configuration must enable at least one feature family")

    # TODO 1 — normalise input
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.max() > 1.0:
        image = image / 255.0
    rgb = image[..., :3]
    gray = rgb2gray(rgb)

    # TODO 2 — channel/name groups
    features, names = [], []
    if config.include_colour:
        features.append(rgb)
        names.extend(["colour_r", "colour_g", "colour_b"])

    # TODO 3 — Gabor branch
    if config.include_gabor:
        bank = make_gabor_bank(config.gabor)
        pooled = gabor_energy_maps(gray, bank, config.gabor.pool_size)  # H x W x K
        features.append(pooled)
        for _, meta in bank:
            names.append(
                f"gabor_f{meta['frequency']}_o{meta['orientation']}_p{meta['phase']}"
            )

    # TODO 4/5 — edge branch
    if config.include_gradient or config.include_edges:
        res = detect_edges(gray, config.edge)
        size = config.edge.density_size

        if config.include_gradient:
            grad_energy = box_mean(res["magnitude"] ** 2, size)
            features.append(grad_energy[..., None])
            names.append("gradient_energy")

        if config.include_edges:
            edges = res["edges"].astype(bool)
            features.append(box_mean(edges.astype(np.float32), size)[..., None])
            names.append("edge_density")

            n = config.edge.n_orientations
            theta = res["direction"] % np.pi
            bin_idx = np.minimum(np.floor(n * theta / np.pi).astype(int), n - 1)
            for i in range(n):
                mask = (edges & (bin_idx == i)).astype(np.float32)
                features.append(box_mean(mask, size)[..., None])
                names.append(f"edge_orientation_{i}")

    # TODO 6 — assemble
    final = np.concatenate(features, axis=2).astype(np.float32)
    if config.standardise_per_image:
        mean = final.mean(axis=(0, 1), keepdims=True)
        std = final.std(axis=(0, 1), keepdims=True)
        final = ((final - mean) / (std + 1e-6)).astype(np.float32)

    return final, names


def global_pool(
    feature_map: np.ndarray, statistics: tuple[str, ...] = ("mean", "std", "p90")
) -> np.ndarray:
    """Pool local channels into one reproducible image descriptor."""
    flat = feature_map.reshape(-1, feature_map.shape[-1])

    funcs = {
        "mean": lambda a: a.mean(axis=0),
        "std": lambda a: a.std(axis=0),
        "p10": lambda a: np.percentile(a, 10, axis=0),
        "p50": lambda a: np.percentile(a, 50, axis=0),
        "p90": lambda a: np.percentile(a, 90, axis=0),
    }

    parts = []
    for s in statistics:
        if s not in funcs:
            raise ValueError(f"Unsupported statistic: {s!r}")
        parts.append(funcs[s](flat))

    return np.concatenate(parts).astype(np.float32)


def feature_family_indices(names: list[str]) -> dict[str, list[int]]:
    """Map colour/Gabor/gradient/edge families to descriptor indices."""

    index_lists = {"colour": [], "gabor": [], "gradient": [], "edge": []}

    for i, name in enumerate(names):
        feature = name.split("_")[0]

        if feature not in index_lists:
            raise ValueError("Invalid or malformed feature name")

        index_lists[feature].append(i)

    return index_lists
