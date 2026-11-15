"""Product Quantization (PQ) from scratch, numpy only.

Idea: split each D-dim vector into M sub-vectors of D/M dims, run k-means with
K centroids in every subspace, and store each vector as M small codes (uint8
when K<=256).  A 512-byte float32 vector (D=128) becomes M=16 bytes.

Search uses Asymmetric Distance Computation (ADC): the query stays exact, we
precompute a (M x K) table of squared distances query-subvector -> centroid,
and the distance to any database vector is just M table lookups + adds.
"""
import numpy as np


def kmeans(X, K, iters=20, seed=0):
    """Plain Lloyd's k-means. Returns (K, d) centroids."""
    rng = np.random.default_rng(seed)
    n = len(X)
    C = X[rng.choice(n, K, replace=False)].copy()
    x2 = (X * X).sum(1)[:, None]
    for _ in range(iters):
        d = x2 - 2 * X @ C.T + (C * C).sum(1)[None, :]
        assign = d.argmin(1)
        for k in range(K):
            pts = X[assign == k]
            # re-seed empty clusters with a random point
            C[k] = pts.mean(0) if len(pts) else X[rng.integers(n)]
    return C


class ProductQuantizer:
    def __init__(self, M=16, K=256, iters=20, seed=0):
        self.M, self.K, self.iters, self.seed = M, K, iters, seed

    def fit(self, X):
        n, D = X.shape
        assert D % self.M == 0, "D must be divisible by M"
        self.D, self.ds = D, D // self.M
        self.codebooks = np.stack([
            kmeans(X[:, m * self.ds:(m + 1) * self.ds], self.K, self.iters, self.seed + m)
            for m in range(self.M)
        ])  # (M, K, ds)
        return self

    def encode(self, X):
        codes = np.empty((len(X), self.M), dtype=np.uint8 if self.K <= 256 else np.uint16)
        for m in range(self.M):
            sub = X[:, m * self.ds:(m + 1) * self.ds]
            C = self.codebooks[m]
            d = (sub * sub).sum(1)[:, None] - 2 * sub @ C.T + (C * C).sum(1)[None, :]
            codes[:, m] = d.argmin(1)
        return codes

    def decode(self, codes):
        return np.concatenate([self.codebooks[m][codes[:, m]] for m in range(self.M)], axis=1)

    def lut(self, q):
        """(M, K) table: squared distance of each query sub-vector to each centroid."""
        qs = q.reshape(self.M, 1, self.ds)
        return ((self.codebooks - qs) ** 2).sum(2)

    def adc(self, lut, codes):
        """Approximate squared distances from the query (via its lut) to coded vectors."""
        return lut[np.arange(self.M)[None, :], codes].sum(1)
