"""Summarise latency measurements without assigning performance pass/fail."""

import math
from statistics import fmean, median


def summarize_latencies(samples_ms: list[float]) -> dict:
    """Report median and nearest-rank p95 for nonnegative millisecond values."""
    for value in samples_ms:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("Latency samples must be finite nonnegative numbers")

    values = sorted(samples_ms)
    count = len(values)

    # An empty workload has no measurable latency distribution.
    if not values:
        return {
            "samples": 0,
            "minimum_ms": None,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "maximum_ms": None,
            "percentile_method": "nearest_rank",
        }

    # Nearest-rank p95 selects the ceil(0.95 * N)-th sorted observation.
    p95_position = math.ceil(0.95 * count) - 1

    return {
        "samples": count,
        "minimum_ms": values[0],
        "mean_ms": fmean(values),
        "median_ms": median(values),
        "p95_ms": values[p95_position],
        "maximum_ms": values[-1],
        "percentile_method": "nearest_rank",
    }