"""Benchmark 4 vector-search systems: brute force, PQ, HNSW, PQ+HNSW.

usage: python benchmark.py [N] [D] [n_queries]
"""
import sys
import time
import numpy as np
from pq import ProductQuantizer
from hnsw import HNSW

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
D = int(sys.argv[2]) if len(sys.argv) > 2 else 128
NQ = int(sys.argv[3]) if len(sys.argv) > 3 else 200
K_NN, PQ_M, PQ_K = 10, 16, 256
HNSW_M, EF_C, EF_S, RERANK = 16, 100, 64, 100

rng = np.random.default_rng(42)


LATENT = 24                                   # real embeddings have low intrinsic dimension
CENTERS = rng.normal(size=(100, LATENT)) * 2.0
PROJ = rng.normal(size=(LATENT, D)) / np.sqrt(LATENT)


def make_data(n):
    """Clustered, low-intrinsic-dim data -- a rough stand-in for text embeddings.
    Queries share the same clusters/projection as the database."""
    z = CENTERS[rng.integers(0, 100, n)] + rng.normal(size=(n, LATENT))
    return (z @ PROJ + 0.1 * rng.normal(size=(n, D))).astype(np.float32)


X, Q = make_data(N), make_data(NQ)


def exact_topk(q, k=K_NN):
    d = (X * X).sum(1) - 2 * X @ q          # |q|^2 dropped: same ranking
    idx = np.argpartition(d, k)[:k]
    return idx[np.argsort(d[idx])]


def rerank(q, ids, k=K_NN):
    ids = np.asarray(ids)
    diff = X[ids] - q
    d = np.einsum("ij,ij->i", diff, diff)
    return ids[np.argsort(d)[:k]]


print(f"N={N} D={D} queries={NQ} k={K_NN}\n")
truth = [set(exact_topk(q)) for q in Q]

# ------------------------------------------------------------------ build
t = time.perf_counter()
pq = ProductQuantizer(PQ_M, PQ_K).fit(X[:min(N, 10000)])
codes = pq.encode(X)
t_pq = time.perf_counter() - t
print(f"PQ trained+encoded in {t_pq:.1f}s")

t = time.perf_counter()
index = HNSW(M=HNSW_M, ef_construction=EF_C).build(X)
t_hnsw = time.perf_counter() - t
print(f"HNSW built in {t_hnsw:.1f}s  (levels={index.max_level + 1}, edges={index.n_edges()})\n")

# ----------------------------------------------------------------- systems
def sys_brute(q):
    return list(exact_topk(q)), N * D

def sys_pq(q, rr=0):
    lut = pq.lut(q)
    d = pq.adc(lut, codes)
    cand = np.argpartition(d, max(rr, K_NN))[:max(rr, K_NN)]
    cand = cand[np.argsort(d[cand])]
    out = rerank(q, cand) if rr else cand[:K_NN]
    return list(out), PQ_K * D + N * PQ_M + rr * D

def sys_hnsw(q, ef=EF_S):
    index.dist_count = 0
    f = index._exact(q)
    ids, _ = index.search(f, K_NN, ef)
    return ids, index.dist_count * D

def sys_pq_hnsw(q, ef=EF_S, rr=0):
    index.dist_count = 0
    lut = pq.lut(q)
    ar = np.arange(PQ_M)[None, :]
    f = lambda ids: lut[ar, codes[ids]].sum(1)
    ids, _ = index.search(f, max(rr, K_NN), max(ef, rr))
    out = rerank(q, ids) if rr else ids[:K_NN]
    return list(out), PQ_K * D + index.dist_count * PQ_M + rr * D


def run(fn, **kw):
    fn(Q[0], **kw)                                    # warm-up
    hits, flops, t = 0, 0, 0.0
    for q, tr in zip(Q, truth):
        s = time.perf_counter()
        ids, fl = fn(q, **kw)
        t += time.perf_counter() - s
        hits += len(set(ids) & tr)
        flops += fl
    return dict(recall=hits / (NQ * K_NN), ms=1000 * t / NQ, qps=NQ / t, flops=flops / NQ)


raw_mb = X.nbytes / 1e6
code_mb = (codes.nbytes + pq.codebooks.nbytes) / 1e6
graph_mb = index.n_edges() * 4 / 1e6                  # int32 neighbor ids

rows = [
    ("1. Brute force (exact)",        run(sys_brute),                       raw_mb),
    ("2. PQ (ADC scan)",              run(sys_pq),                          code_mb),
    ("3. HNSW (exact dists)",         run(sys_hnsw),                        raw_mb + graph_mb),
    ("4. PQ + HNSW",                  run(sys_pq_hnsw),                     code_mb + graph_mb),
    ("2b. PQ + exact rerank*",        run(sys_pq, rr=RERANK),               code_mb + raw_mb),
    ("4b. PQ+HNSW + exact rerank*",   run(sys_pq_hnsw, rr=RERANK),          code_mb + graph_mb + raw_mb),
]

base = rows[0][1]
print(f"{'system':30s} {'recall@10':>9s} {'ms/query':>9s} {'QPS':>8s} {'Mflop/q':>8s} {'flop cut':>8s} {'memory MB':>10s}")
for name, r, mb in rows:
    print(f"{name:30s} {r['recall']:9.3f} {r['ms']:9.2f} {r['qps']:8.0f} {r['flops']/1e6:8.2f} "
          f"{base['flops']/r['flops']:7.1f}x {mb:10.1f}")
print("* rerank keeps raw vectors in memory, so no memory saving.")

print("\nHNSW ef_search sweep (recall vs speed):")
print(f"{'ef':>5s} | {'HNSW recall':>11s} {'ms':>6s} | {'PQ+HNSW recall':>14s} {'ms':>6s}")
for ef in (16, 32, 64, 128, 256):
    a, b = run(sys_hnsw, ef=ef), run(sys_pq_hnsw, ef=ef)
    print(f"{ef:5d} | {a['recall']:11.3f} {a['ms']:6.2f} | {b['recall']:14.3f} {b['ms']:6.2f}")
