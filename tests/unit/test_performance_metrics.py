import pytest

from rag_eu_law.evaluation.performance_metrics import summarize_latencies


def test_unsorted_samples_are_summarized_correctly() -> None:
    result = summarize_latencies([40, 10, 30, 20])

    assert result["samples"] == 4
    assert result["minimum_ms"] == 10
    assert result["mean_ms"] == 25
    assert result["median_ms"] == 25
    assert result["p95_ms"] == 40
    assert result["maximum_ms"] == 40


def test_p95_uses_nearest_rank() -> None:
    result = summarize_latencies(list(range(1, 21)))

    assert result["median_ms"] == 10.5
    assert result["p95_ms"] == 19
    assert result["percentile_method"] == "nearest_rank"


def test_single_measurement_is_supported() -> None:
    result = summarize_latencies([12.5])

    assert result["samples"] == 1
    assert result["median_ms"] == 12.5
    assert result["p95_ms"] == 12.5


def test_empty_workload_has_no_latency_score() -> None:
    result = summarize_latencies([])

    assert result["samples"] == 0
    assert result["mean_ms"] is None
    assert result["median_ms"] is None
    assert result["p95_ms"] is None
    assert result["maximum_ms"] is None


@pytest.mark.parametrize(
    "invalid_value",
    [-1, float("nan"), float("inf"), True, "10"],
)
def test_invalid_measurements_are_rejected(invalid_value) -> None:
    with pytest.raises(ValueError, match="finite nonnegative"):
        summarize_latencies([invalid_value])