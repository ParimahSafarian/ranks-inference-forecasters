"""Tests for the Andrews (1991) plug-in bandwidth (Step 2, table row T5)."""
import numpy as np

from rankci.core.bandwidth import (
    ar1_coefficient,
    andrews_alpha1,
    andrews_bandwidth,
    _ANDREWS_C,
    _RHO_CLAMP,
)


def _ar1(n, rho, seed=0, sigma=1.0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n) * sigma
    y = np.empty(n)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = rho * y[t - 1] + e[t]
    return y


def test_alpha1_matches_closed_form():
    """andrews_alpha1 applies 4 rho^2 / ((1-rho)^2 (1+rho)^2) to the fitted rho."""
    d = _ar1(400, 0.5, seed=1)
    rho = ar1_coefficient(d)
    expected = 4 * rho**2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    assert np.isclose(andrews_alpha1(d), expected, rtol=1e-12)


def test_bandwidth_matches_closed_form():
    """andrews_bandwidth = floor(1.1447 (alpha n)^{1/3}), capped at n-1."""
    d = _ar1(400, 0.6, seed=2)
    n = d.size
    alpha1 = andrews_alpha1(d)
    expected = min(int(np.floor(_ANDREWS_C * (alpha1 * n) ** (1 / 3))), n - 1)
    assert andrews_bandwidth(d) == expected


def test_white_noise_small_bandwidth():
    """Near-iid series -> alpha(1) ~ 0 -> tiny bandwidth."""
    d = np.random.default_rng(3).standard_normal(1000)
    assert andrews_bandwidth(d) <= 2


def test_persistence_increases_bandwidth():
    """More persistent series get (weakly) larger bandwidths."""
    n = 2000
    L_low = andrews_bandwidth(_ar1(n, 0.2, seed=4))
    L_mid = andrews_bandwidth(_ar1(n, 0.6, seed=4))
    L_high = andrews_bandwidth(_ar1(n, 0.9, seed=4))
    assert L_low < L_mid < L_high


def test_rho_clamped():
    """A near-unit-root series is clamped, not sent to infinity."""
    d = _ar1(2000, 0.999, seed=5)
    # alpha at the clamp is finite; bandwidth stays <= n-1.
    rho_c = _RHO_CLAMP
    alpha_clamp = 4 * rho_c**2 / ((1 - rho_c) ** 2 * (1 + rho_c) ** 2)
    assert andrews_alpha1(d) <= alpha_clamp + 1e-9
    assert andrews_bandwidth(d) <= d.size - 1


def test_short_series_zero():
    assert andrews_bandwidth(np.array([1.0, 2.0])) == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
