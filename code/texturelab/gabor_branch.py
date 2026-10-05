"""Student-facing Gabor branch. Core implementation is intentionally inspectable."""

from __future__ import annotations
from dataclasses import asdict
import numpy as np
from .config import GaborConfig
from .supplied import box_mean, correlate2d


def make_gabor_bank(config: GaborConfig) -> list[tuple[np.ndarray, dict]]:
    """Return ordered zero-mean, unit-norm kernels and their metadata."""
    # YOUR CODE HERE
    #
    # 1 — Validate the configuration
    #   - Require kernel_size to be odd and at least 3. Raise ValueError if not.

    if config.kernel_size % 2 == 0 or config.kernel_size < 3:
        raise ValueError("Kernel size must be odd and >= 3")

    kernels = []
    metadatas = []
    # TODO 2 — Build the centred sampling grid
    #   - Construct float32 x/y coordinates spanning equally on both sides of 0.
    #   - The resulting grid must have shape (kernel_size, kernel_size).

    K = config.kernel_size
    r = np.arange(K, dtype=np.float32) - K // 2
    n, m = np.meshgrid(r, r, indexing="ij")  # n = rows (y), m = cols (x))

    # TODO 3 — Generate every filter in a stable order
    #   - Loop over frequency first, orientation second, and phase last.
    #   - Rotate the x coordinate into the current orientation.
    #   - Multiply a Gaussian envelope by a sine carrier with the current frequency and phase.
    #   - Remember to add the phase in the metadata.

    for f in config.frequencies:
        for w in config.orientations:
            for phi in config.phases:
                env = (1 / (2 * np.pi * config.sigma**2)) * np.exp(
                    -(m**2 + n**2) / (2 * config.sigma**2)
                )
                carrier = np.sin(2 * np.pi * f * (np.cos(w) * m + np.sin(w) * n) + phi)

                gabor = env * carrier
                kernels.append(gabor)
                metadata = asdict(config)
                metadata["phase"] = phi
                metadata["orientation"] = w
                metadata["frequency"] = f
                metadatas.append(metadata)

    # TODO 4 — Normalise and validate each kernel
    #   - Subtract the kernel mean so constant images have little response.
    #   - Divide by its Euclidean norm and reject a near-zero/degenerate norm.
    #   - Store the final kernel as float32.

    for i in range(len(kernels)):
        kernels[i] = kernels[i] - kernels[i].mean()  # subtract the mean
        norm = np.linalg.norm(kernels[i])
        if np.isclose(norm, 0):
            raise ValueError(f"Near zero norm for kernel : {metadatas[i]}")
        kernels[i] = (kernels[i] / norm).astype(
            np.float32
        )  # normalise by euclidian norm

    # TODO 5 — Attach metadata and return
    #   - Include frequency, orientation, phase, and asdict(config) for each item.
    #   - Return a list of (kernel, metadata) tuples in the loop order above.

    return [result for result in zip(kernels, metadatas)]


def gabor_energy_maps(
    gray: np.ndarray,
    bank: list[tuple[np.ndarray, dict]],
    pool_size: int = 9,
    energy: str = "squared",
) -> np.ndarray:
    """Return finite H x W x K locally pooled energy maps."""
    # TODO 1 — Compute one response map per bank entry
    #   - Preserve bank order and correlate (do not convolve) gray with each kernel.
    response_maps = []
    for kernel, metadata in bank:
        response_maps.append(correlate2d(gray, kernel))

    # TODO 2 — Convert responses to non-negative energy
    #   - For "squared", square the signed response.
    #   - For "absolute", take its absolute value.
    #   - Raise ValueError for any other energy name.
    #

    if energy not in ("squared", "absolute"):
        raise ValueError(
            "Parameter energy must take on values : 'squared' or 'absolute'"
        )

    for i in range(len(response_maps)):
        if energy == "squared":
            response_maps[i] = response_maps[i] ** 2
        else:
            response_maps[i] = np.abs(response_maps[i])

    # TODO 3 — Pool and assemble the channels
    #   - Apply box_mean with pool_size to every energy map.
    #   - Pooling must preserve the original image height and width.
    #   - Stack maps on the final axis to obtain H x W x K float32 output.

    if pool_size < 0 or pool_size % 2 == 0:
        raise ValueError("Pool size must be an odd positive integer")

    final = np.zeros(
        (gray.shape[0], gray.shape[1], len(bank)), dtype=np.float32
    )  # final stacked responses

    for k, resmap in enumerate(response_maps):
        final[:, :, k] = box_mean(resmap, pool_size)

    # TODO 4 — Check numerical validity
    #   - Raise FloatingPointError if any sreturned value is NaN or infinite.
    if not np.isfinite(final).all():
        raise FloatingPointError(
            "At least one infinite or NaN value was found in the final array."
        )

    return final
