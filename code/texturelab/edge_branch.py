"""Student-facing exact-direction NMS, adaptive thresholds, and hysteresis."""

from __future__ import annotations
from collections import deque
import numpy as np
from .config import EdgeConfig
from .supplied import gaussian_smooth, sobel_gradients


def bilinear_sample(image: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sample a 2-D image at floating-point image coordinates."""
    # [AI-CODE] [HUMAN-CHECK]
    height, width = image.shape
    y = np.clip(np.asarray(y, dtype=np.float64), 0, height - 1)
    x = np.clip(np.asarray(x, dtype=np.float64), 0, width - 1)

    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)

    wy = y - y0
    wx = x - x0

    top = (1 - wx) * image[y0, x0] + wx * image[y0, x1]
    bottom = (1 - wx) * image[y1, x0] + wx * image[y1, x1]
    return (1 - wy) * top + wy * bottom


def nms_interpolated(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Thin gradient magnitude along the exact image-coordinate direction."""
    #
    # TODO 1 — Validate inputs
    #   - Require matching two-dimensional magnitude and direction arrays.
    if magnitude.shape != direction.shape:
        raise ValueError("Magnitude and direction shapes must match")
    if magnitude.ndim != 2:
        raise ValueError("Arrays should be 2D")

    # TODO 2 — Construct exact-direction comparison coordinates
    #   - Build row/column index grids for the complete image.
    #   - In image coordinates, the unit step is dy=sin(theta), dx=cos(theta).
    #   - Bilinearly sample magnitude one step forward and one step backward.

    x, y = np.meshgrid(np.arange(magnitude.shape[1]), np.arange(magnitude.shape[0]))

    dy = np.sin(direction)
    dx = np.cos(direction)
    x_fwd = x + dx
    y_fwd = y + dy
    x_back = x - dx
    y_back = y - dy
    sampled_fwd = bilinear_sample(magnitude, y_fwd, x_fwd)
    sampled_back = bilinear_sample(magnitude, y_back, x_back)

    # TODO 3 — Suppress non-maxima
    #   - Keep the original magnitude when it is >= both interpolated neighbours.
    #   - Write zero elsewhere, return float32, and explicitly zero all four borders.
    keep = (magnitude >= sampled_fwd) & (magnitude >= sampled_back)
    maximum = np.where(keep, magnitude, 0).astype(np.float32)
    # set all the borders to 0
    maximum[0, :], maximum[:, 0], maximum[-1, :], maximum[:, -1] = 0, 0, 0, 0
    return maximum


def adaptive_thresholds(nms: np.ndarray, config: EdgeConfig) -> tuple[float, float]:
    """Select reproducible low/high thresholds without test labels."""

    if config.threshold_method not in ("percentile", "robust"):
        raise ValueError("Threshold method must be 'percentile' or 'robust'")

    if config.low_ratio > 1 or config.low_ratio < 0:
        raise ValueError("Low ratio must be between 0 and 1")

    # TODO 1 — Isolate useful responses
    #   - Estimate thresholds from strictly positive NMS values only.
    #   - If none exist, return a pair that produces no strong/weak edges.
    pos = nms[nms > 0]
    if len(pos) == 0:
        return (np.inf, np.inf)

    # TODO 2 — Compute the high threshold
    #   - "percentile": use config.high_percentile on positive values.
    #   - "robust": use median + 2.5 * 1.4826 * median absolute deviation.
    #       - 1.4826 is a mathematically motivated conversion from MAD to a standard-deviation-like scale.
    #       - 2.5 is a scale multiplier for robust outlier rejection.
    #   - Reject unknown threshold_method values with ValueError.

    if config.threshold_method == "percentile":
        high = np.percentile(pos, config.high_percentile)
    else:
        high = np.median(pos) + 2.5 * 1.4826 * np.median(
            np.absolute(pos - np.median(pos))
        )
    # TODO 3 — Compute and return the low threshold
    #   - low is config.low_ratio multiplied by high.
    #   - Return ordinary Python floats in (low, high) order.

    low = config.low_ratio * high

    return (float(low), float(high))


def hysteresis(
    strong: np.ndarray, weak: np.ndarray, connectivity: int = 8
) -> np.ndarray:
    """Keep all weak pixels reachable from strong seeds."""
    # [AI-CODE] [HUMAN-CHECK]
    # AI was used for debugging
    #
    # TODO 1 — Validate and initialise
    #   - Require matching mask shapes and connectivity equal to 4 or 8.
    #   - Convert inputs to Boolean masks without modifying the caller's arrays.
    #   - Seed the result and a deque with every strong-pixel coordinate.
    if strong.shape != weak.shape:
        raise ValueError("Masks must have matching shapes")
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")
    strong_mask = strong.copy().astype(bool)
    weak_mask = weak.copy().astype(bool)
    result = strong.copy().astype(bool)
    coordsY, coordsX = np.nonzero(strong_mask)

    q = deque([coord for coord in zip(coordsY, coordsX)])

    # TODO 2 — Define the neighbourhood
    #   - Four-connectivity uses vertical/horizontal offsets.
    #   - Eight-connectivity additionally includes all diagonal offsets.
    neighs = [
        (0, -1),
        (-1, 0),
        (1, 0),
        (0, 1),
    ]
    if connectivity == 8:
        neighs.extend([(-1, -1), (1, -1), (-1, 1), (1, 1)])

    # TODO 3 — Traverse the full connected component
    #   - Pop a coordinate, check in-bounds allowed neighbours, and accept every
    #     unvisited weak pixel connected to a seed.
    #   - Enqueue newly accepted pixels so multi-pixel weak chains are retained.
    #   - Return the final Boolean mask after the queue is exhausted.

    while len(q) > 0:
        y, x = q.popleft()
        for dy, dx in neighs:
            if (y + dy < 0 or y + dy >= weak_mask.shape[0]) or (
                x + dx < 0 or x + dx >= weak_mask.shape[1]
            ):
                continue
            if weak_mask[y + dy, x + dx] == 1 and result[y + dy, x + dx] == 0:
                result[y + dy, x + dx] = 1
                q.append((y + dy, x + dx))

    return result


def detect_edges(gray: np.ndarray, config: EdgeConfig) -> dict[str, np.ndarray | float]:
    """Run the supplied smoothing/Sobel stages and student edge stages."""
    smoothed = gaussian_smooth(gray, config.gaussian_sigma)
    gx, gy, magnitude, direction = sobel_gradients(smoothed)
    # YOUR CODE HERE
    #
    # TODO 1 — Thin and threshold
    #   - Run nms_interpolated on magnitude/direction.
    #   - Obtain (low, high) from adaptive_thresholds.
    #   - Strong pixels meet/exceed high; weak pixels meet/exceed low but are not strong.

    nms = nms_interpolated(magnitude, direction)
    low, high = adaptive_thresholds(nms, config)
    strong = nms >= high
    weak = (nms >= low) & ~strong
    # TODO 2 — Link and package the result
    #   - Run hysteresis with config.connectivity.
    #   - Return gx, gy, magnitude, direction, nms, low, high, and edges using
    #     exactly those dictionary keys so downstream feature code remains stable.
    edges = hysteresis(strong, weak, config.connectivity)

    return {
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "direction": direction,
        "nms": nms,
        "low": low,
        "high": high,
        "edges": edges,
    }
