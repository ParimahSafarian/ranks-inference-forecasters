"""
Joint moving block bootstrap (MBB) for an (n, p) time-indexed panel.

The target data are serially *and* cross-sectionally dependent, so the
bootstrap must resample whole time rows in contiguous blocks (Künsch 1989;
Liu & Singh 1992):

  * contiguous blocks of rows preserve the serial dependence within a block;
  * resampling *rows* keeps every column (and every pairwise difference)
    on the same draw, preserving cross-sectional dependence — including the
    dependence between overlapping pairs (j, k) and (j, l) that drives the
    max statistic;
  * on an unbalanced panel the NaN pattern travels with the row, so the
    bootstrap sample has a realistic missingness structure.

Block length
------------
The MBB variance of a sample mean with block length ``l`` is asymptotically
the Bartlett long-run-variance estimator with weights ``1 - h/l``, i.e. the
``nw_se`` estimator with bandwidth ``L = l - 1``. The default therefore ties
the block length to the HAC bandwidth already used for the standard errors:

    l = L + 1,

where ``L`` is the user-supplied Newey–West bandwidth if given, and otherwise
the median over pairwise difference series of the Andrews (1991) plug-in
bandwidth (:func:`rankci.core.bandwidth.andrews_bandwidth`). A single panel-wide
block length is required because all columns are resampled jointly.
"""
import numpy as np

from .bandwidth import andrews_bandwidth


def mbb_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """
    Row indices for one moving-block-bootstrap resample of length ``n``.

    Draws ``ceil(n / l)`` block start positions uniformly from
    ``{0, ..., n - l}``, concatenates the blocks ``start, ..., start + l - 1``
    and truncates to ``n`` rows. ``block_length == 1`` reduces to the i.i.d.
    row bootstrap.
    """
    l = int(block_length)
    if l < 1 or l > n:
        raise ValueError(f"block_length must be in [1, n={n}], got {l}.")
    n_blocks = -(-n // l)                                 # ceil(n / l)
    starts = rng.integers(0, n - l + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(l)[None, :]).ravel()
    return idx[:n]


def default_block_length(
    X: np.ndarray,
    L: int | None = None,
    min_overlap: int = 2,
) -> int:
    """
    Panel-wide MBB block length ``l = L + 1`` (see module docstring).

    Parameters
    ----------
    X           : (n, p) panel, may contain NaN.
    L           : explicit NW bandwidth; if given, returns ``L + 1``.
    min_overlap : pairs with fewer shared observations are ignored when
                  aggregating the Andrews bandwidths.

    Returns
    -------
    Integer block length in ``[1, n]``.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if L is not None:
        return int(max(1, min(int(L) + 1, n)))

    bandwidths = []
    for j in range(p):
        for k in range(j + 1, p):
            d = X[:, j] - X[:, k]
            d = d[np.isfinite(d)]
            if d.size < max(min_overlap, 2):
                continue
            bandwidths.append(andrews_bandwidth(d))
    if not bandwidths:
        return 1
    L_med = int(np.floor(np.median(bandwidths)))
    return int(max(1, min(L_med + 1, n)))


def pairwise_means(Xb: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """
    Mean of ``Xb[:, j] - Xb[:, k]`` over rows where both are observed, for
    every row ``(j, k)`` of ``pairs``. NaN where a pair has no overlap.
    """
    D = Xb[:, pairs[:, 0]] - Xb[:, pairs[:, 1]]           # (n, m)
    obs = np.isfinite(D)
    cnt = obs.sum(axis=0)
    s = np.where(obs, D, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return means
