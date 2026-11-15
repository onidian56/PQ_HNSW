"""Hierarchical Navigable Small World graph (Malkov & Yashunin) from scratch.

Every vector is assigned a random top layer ~ floor(-ln(U) * mL).  Upper layers
are sparse "express lanes"; layer 0 holds everybody.  Search greedily descends
from the top layer, then does a best-first beam search (width ef) on layer 0.

The graph is independent of how distances are computed: search takes a
`distfn(ids) -> distances` callable, so the same graph can be traversed with
exact vectors OR with PQ codes (ADC) -- that is the "PQ + HNSW" system.
"""
import heapq
import math
import numpy as np


class HNSW:
    def __init__(self, M=16, ef_construction=100, seed=0):
        self.M = M                # max links per node on layers >= 1
        self.M0 = 2 * M           # max links on layer 0
        self.efc = ef_construction
        self.mL = 1.0 / math.log(M)
        self.rng = np.random.default_rng(seed)
        self.graph = []           # graph[layer] = {node: [neighbors]}
        self.entry = None
        self.max_level = -1
        self.dist_count = 0       # number of distance evaluations (efficiency metric)

    # ---------------------------------------------------------------- build
    def build(self, X):
        self.X = X
        for i in range(len(X)):
            self._insert(i)
        return self

    def _exact(self, q):
        X = self.X

        def f(ids):
            diff = X[ids] - q
            return np.einsum("ij,ij->i", diff, diff)
        return f

    def _search_layer(self, distfn, eps, ef, layer):
        """Best-first beam search on one layer. eps: list of (dist, id).
        Returns list of (dist, id) sorted ascending, length <= ef."""
        g = self.graph[layer]
        visited = {i for _, i in eps}
        cand = list(eps)
        heapq.heapify(cand)                       # min-heap on distance
        res = [(-d, i) for d, i in eps]
        heapq.heapify(res)                        # max-heap (negated)
        while cand:
            d, c = heapq.heappop(cand)
            if d > -res[0][0] and len(res) >= ef:
                break
            nbrs = [n for n in g[c] if n not in visited]
            if not nbrs:
                continue
            visited.update(nbrs)
            ds = distfn(nbrs)
            self.dist_count += len(nbrs)
            for n, dn in zip(nbrs, ds.tolist()):
                if len(res) < ef or dn < -res[0][0]:
                    heapq.heappush(cand, (dn, n))
                    heapq.heappush(res, (-dn, n))
                    if len(res) > ef:
                        heapq.heappop(res)
        return sorted((-d, i) for d, i in res)

    def _select(self, cands, m):
        """Heuristic neighbor selection: keep a candidate only if it is closer to
        the base point than to every neighbor already kept (diversifies directions)."""
        X, sel = self.X, []
        for d, e in cands:                        # cands sorted by dist to base
            if len(sel) >= m:
                break
            if sel:
                diff = X[sel] - X[e]
                if (np.einsum("ij,ij->i", diff, diff) <= d).any():
                    continue
            sel.append(e)
        return sel

    def _insert(self, i):
        level = int(-math.log(1.0 - self.rng.random()) * self.mL)
        while len(self.graph) <= level:
            self.graph.append({})
        if self.entry is None:
            for l in range(level + 1):
                self.graph[l][i] = []
            self.entry, self.max_level = i, level
            return
        f = self._exact(self.X[i])
        ep = [(float(f([self.entry])[0]), self.entry)]
        for l in range(self.max_level, level, -1):           # greedy descent
            ep = self._search_layer(f, ep, 1, l)[:1]
        for l in range(min(level, self.max_level), -1, -1):  # connect on each layer
            W = self._search_layer(f, ep, self.efc, l)
            mmax = self.M0 if l == 0 else self.M
            nb = self._select(W, self.M)
            self.graph[l][i] = list(nb)
            for n in nb:
                lst = self.graph[l][n]
                lst.append(i)
                if len(lst) > mmax:                          # shrink overfull node
                    diff = self.X[lst] - self.X[n]
                    dd = np.einsum("ij,ij->i", diff, diff)
                    order = sorted(zip(dd.tolist(), lst))
                    self.graph[l][n] = self._select(order, mmax)
            ep = W
        if level > self.max_level:
            for l in range(self.max_level + 1, level + 1):   # new top layers: i is alone
                self.graph[l][i] = []
            self.entry, self.max_level = i, level

    # --------------------------------------------------------------- search
    def search(self, distfn, k=10, ef=64):
        """Return (ids, dists) of the k approx. nearest neighbors under distfn."""
        ep = [(float(distfn([self.entry])[0]), self.entry)]
        self.dist_count += 1
        for l in range(self.max_level, 0, -1):
            ep = self._search_layer(distfn, ep, 1, l)[:1]
        W = self._search_layer(distfn, ep, max(ef, k), 0)[:k]
        return [i for _, i in W], [d for d, _ in W]

    def n_edges(self):
        return sum(len(v) for layer in self.graph for v in layer.values())
