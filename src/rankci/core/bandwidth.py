"""
Andrews (1991) data-dependent bandwidth for the Bartlett-kernel HAC estimator.

This replaces the Newey--West (1994) ``L = floor(4 (n/100)^{2/9})`` rule of
thumb. For each (difference) series we fit an AR(1), form the Andrews eq.(6.4)
persistence functional ``alpha_hat(1)``, and set the MSE-optimal Bartlett
bandwidth

    S_T* = 1.1447 * (alpha_hat(1) * n)^{1/3}      (Andrews 1991, eq. 6.2, Table I)

For a *single* AR(1) series the eq.(6.4) functional collapses — the innovation
variance cancels — to a function of the AR coefficient alone,

    alpha_hat(1) = 4 rho^2 / ((1 - rho)^2 (1 + rho)^2).

The Bartlett kernel and the T^{1/3} rate follow Andrews (1991, Thm. 1); the
AR(1) plug-in machinery follows Newey--West (1994). Rate T^{1/3} (rather than
the old T^{2/9}) is the MSE-optimal Bartlett rate: the old rule undersmooths,
biasing the long-run variance down, standard errors down, and confidence
intervals too narrow — the anti-conservative direction.
"""
import numpy as np

# AR-coefficient clamp: guards the rho -> +-1 blow-ups of alpha_hat(1).
_RHO_CLAMP = 0.97

# Andrews (1991) Bartlett constant c_gamma for the q=1 kernel (Table I).
_ANDREWS_C = 1.1447


def ar1_coefficient(d: np.ndarray) -> float:
    """
    OLS estimate of ``rho`` in ``d_t = c + rho * d_{t-1} + e_t``.

    The series is treated as a contiguous sample: callers pass either the
    compressed overlap series (matching :func:`nw_se`) or a gap-free segment.
    Returns 0.0 if there are fewer than 3 observations or no variation.
    """
    d = np.asarray(d, dtype=float)
    n = d.size
    if n < 3:
        return 0.0
    y = d[1:]
    x = d[:-1]
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(xc, y - y.mean()) / denom)


def andrews_alpha1(d: np.ndarray) -> float:
    """
    Andrews (1991) eq.(6.4) persistence functional for the Bartlett (q=1)
    kernel, univariate AR(1) case:

        alpha(1) = 4 rho^2 / ((1 - rho)^2 (1 + rho)^2).

    ``rho`` is clamped to +-``_RHO_CLAMP`` before evaluation.
    """
    rho = float(np.clip(ar1_coefficient(d), -_RHO_CLAMP, _RHO_CLAMP))
    num = 4.0 * rho ** 2
    den = (1.0 - rho) ** 2 * (1.0 + rho) ** 2
    return num / den


def andrews_bandwidth(d: np.ndarray) -> int:
    """
    Integer Bartlett bandwidth ``L`` for a series ``d`` via the Andrews plug-in.

        L = floor( 1.1447 * (alpha_hat(1) * n)^{1/3} ),

    floored at 0 and capped at ``n - 1``. White noise (rho ~ 0) gives L ~ 0;
    a persistent series gives a larger L.
    """
    d = np.asarray(d, dtype=float)
    n = d.size
    if n < 3:
        return 0
    L = _ANDREWS_C * (andrews_alpha1(d) * n) ** (1.0 / 3.0)
    return max(0, min(int(np.floor(L)), n - 1))
