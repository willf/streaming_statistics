from streaming_statistics.stats import stats
import math
import numpy as np
import pytest


def test_stats():
    results = stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert results["n"] == 10
    assert results["mean"] == 5.5
    assert results["variance"] == 8.25
    assert results["stddev"] == math.sqrt(results["variance"])
    assert results["min"] == 1
    assert results["max"] == 10
    assert results["sum"] == 55


def test_stats_with_numpy():
    mu, sigma = 0, 0.1
    s = np.random.normal(mu, sigma, 10000)
    print("\n====The number of entries is", len(s))
    results = stats(s)
    assert results["n"] == np.size(s)
    assert results["mean"] == pytest.approx(np.mean(s))
    assert results["variance"] == pytest.approx(np.var(s))
    assert results["stddev"] == pytest.approx(np.std(s))
    assert results["min"] == np.min(s)
    assert results["max"] == np.max(s)
    assert results["sum"] == pytest.approx(np.sum(s))
    assert results["50th"] == pytest.approx(np.percentile(s, 50), rel=0.1)
