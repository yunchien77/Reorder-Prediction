"""Product-embedding similarity features (NOT part of the 54 base features).

Two product vector spaces
  i2v : item2vec-style vectors learned from the ORDER BASKETS of the TRAIN users only (PPMI of item co-occurrence + truncated SVD, dim 32).
        Learned per fold; val / test users' baskets are never used.  (Not external data: it only re-uses Instacart.)
  txt : pretrained sentence encoder (all-MiniLM-L6-v2) on the product NAME only (dim 384, fixed).  (Pretrained external model.)
Per (user, product) pair, cosine similarity of the product to what the user bought BEFORE the target order (prior orders only):
  sim_last_mean  mean of the other items of the last prior order   sim_last_max  most similar other item of the last prior order
  sim_last3_mean mean of the other items of the last 3 prior orders  sim_hist_mean count-weighted mean of the other items of the whole history
("other" = the product itself is excluded.)  NaN when there is nothing to compare with.  Nothing about the target order is used.
Also saved: product embedding dims (16) for use as product features.
  cd src && python -m embed.build [--folds 1 2 3 4 5]   ->  features/embed/{txt_name.npy, fold{k}.parquet, i2v_fold{k}.npy}"""
import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd

import common
from external.names import RAW, products

OUT = common.FEATURES_DIR / "embed"
LMAX, DIM, EMB = 40, 32, 16
SIMS = ["sim_last_mean", "sim_last_max", "sim_last3_mean", "sim_hist_mean"]


def history():
    o = pd.read_csv(RAW / "orders.csv", usecols=["order_id", "user_id", "eval_set", "order_number"])
    o = o[o.eval_set == "prior"]                                           # prior orders only (the labelled order is never touched)
    op = pd.read_csv(RAW / "order_products__prior.csv", usecols=["order_id", "product_id", "add_to_cart_order"])
    h = op.merge(o[["order_id", "user_id", "order_number"]], on="order_id")
    h["back"] = h.groupby("user_id").order_number.transform("max") - h.order_number    # 0 = last prior order
    return h


def text_vectors(P):
    path = OUT / "txt_name.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    p = products().set_index("product_id").product_name.reindex(np.arange(1, P)).fillna("").tolist()
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    V = np.vstack([np.zeros((1, 384), np.float32), m.encode(p, batch_size=512, normalize_embeddings=True, show_progress_bar=False)])
    np.save(path, V.astype(np.float32))
    return V.astype(np.float32)


def item2vec(h, train_users, P):
    """PPMI(item co-occurrence in a basket; context smoothing 0.75) + truncated SVD, from the train users' baskets only."""
    t = h[h.user_id.isin(train_users)]
    oid = pd.factorize(t.order_id)[0]
    B = sp.csr_matrix((np.ones(len(t), np.float32), (oid, t.product_id.values)), shape=(oid.max() + 1, P))
    C = (B.T @ B).tocsr(); C.setdiag(0); C.data[C.data < 2] = 0; C.eliminate_zeros()        # pairs seen in >= 2 baskets
    r = np.asarray(C.sum(1)).ravel().astype(np.float64); tot = r.sum()
    ctx = r ** 0.75; ctx /= ctx.sum()
    C = C.tocoo()
    pmi = np.log((C.data / tot) / ((r[C.row] / tot) * ctx[C.col]))
    keep = pmi > 0
    M = sp.csr_matrix((pmi[keep].astype(np.float32), (C.row[keep], C.col[keep])), shape=(P, P))
    U, S, _ = randomized_svd(M, n_components=DIM, n_iter=5, random_state=0)
    V = U * np.sqrt(S)
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    V[np.asarray(M.sum(1)).ravel() == 0] = 0                                   # products without any co-occurrence: no vector
    V[0] = 0
    return V.astype(np.float32)


def membership(h, P, U, uidx):
    """User x product matrices: history counts, last order (0/1), last 3 orders (0/1) + padded product list of the last order."""
    def mat(d, w=None):
        return sp.csr_matrix((np.ones(len(d), np.float32) if w is None else w, (uidx[d.user_id.values], d.product_id.values)), shape=(U, P))
    H = mat(h)
    H.sum_duplicates()
    L1 = mat(h[h.back == 0]); L3 = mat(h[h.back < 3]); L3.data[:] = 1
    last = h[h.back == 0].sort_values(["user_id", "add_to_cart_order"])
    last = last[last.groupby("user_id").cumcount() < LMAX]
    pad = np.zeros((U, LMAX), np.int64)
    pad[uidx[last.user_id.values], last.groupby("user_id").cumcount().values] = last.product_id.values
    return H, L1, L3, pad


def lookup(M, pu, pp, P):
    """M[pu, pp] for many pairs (0 where absent)."""
    c = M.tocoo(); key = c.row.astype(np.int64) * P + c.col
    o = np.argsort(key); key, val = key[o], c.data[o]
    q = pu.astype(np.int64) * P + pp
    pos = np.minimum(np.searchsorted(key, q), len(key) - 1)
    return np.where(key[pos] == q, val[pos], 0).astype(np.float32)


def cos_excl(V, M, pu, pp, P, chunk=200_000):
    """cos(v_p, sum of the other items of the user's set M) with the product itself removed from the sum (M holds weights)."""
    S = np.asarray(M @ V, np.float32); SS = (S ** 2).sum(1)
    m = lookup(M, pu, pp, P)
    out = np.full(len(pu), np.nan, np.float32)
    for i in range(0, len(pu), chunk):
        sl = slice(i, i + chunk)
        v = V[pp[sl]]; dot = (v * S[pu[sl]]).sum(1); vn = (v ** 2).sum(1)
        num = dot - m[sl] * vn
        n2 = SS[pu[sl]] - 2 * m[sl] * dot + m[sl] ** 2 * vn
        ok = (n2 > 1e-6) & (vn > 0)
        out[sl] = np.where(ok, num / np.sqrt(np.maximum(n2, 1e-12)) / np.sqrt(np.maximum(vn, 1e-12)), np.nan)
    return out


def max_last(V, pad, pu, pp):
    """max cosine between v_p and the OTHER items of the user's last prior order."""
    Vt = torch.from_numpy(V).cuda(); Lt = torch.from_numpy(pad).cuda()
    chunk = max(2000, int(4e7 / (LMAX * V.shape[1])))
    nrm = (Vt ** 2).sum(1)
    out = np.empty(len(pu), np.float32)
    for i in range(0, len(pu), chunk):
        p = torch.from_numpy(pp[i:i + chunk]).cuda(); l = Lt[torch.from_numpy(pu[i:i + chunk]).cuda()]
        s = torch.einsum("cd,cld->cl", Vt[p], Vt[l])
        s.masked_fill_((l == 0) | (l == p[:, None]) | (nrm[l] == 0), -2.0)
        mx = s.max(1).values
        mx[(mx < -1.5) | (nrm[p] == 0)] = float("nan")
        out[i:i + chunk] = mx.cpu().numpy()
    return out


def sims(V, tag, mats, pu, pp, P):
    H, L1, L3, pad = mats
    return {f"{tag}_sim_last_mean": cos_excl(V, L1, pu, pp, P), f"{tag}_sim_last_max": max_last(V, pad, pu, pp),
            f"{tag}_sim_last3_mean": cos_excl(V, L3, pu, pp, P), f"{tag}_sim_hist_mean": cos_excl(V, H, pu, pp, P)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P = int(products().product_id.max()) + 1
    h = history()
    txt = text_vectors(P)
    np.save(OUT / "txt_pca16.npy", PCA(EMB, random_state=0).fit_transform(txt).astype(np.float32))
    print(f"history {len(h):,} rows, text vectors {txt.shape} ({time.time() - t0:.0f}s)", flush=True)
    for fold in a.folds:
        sp_ = pd.read_csv(common.SPLITS / f"fold{fold}.csv")
        users = sp_.user_id.values; uidx = np.full(users.max() + 1, -1, np.int64); uidx[users] = np.arange(len(users))
        hh = h[h.user_id.isin(users)]
        mats = membership(hh, P, len(users), uidx)
        pairs = pd.read_parquet(common.FEATURES_DIR / f"fold{fold}.parquet", columns=["user_id", "product_id", "split"])
        pu, pp = uidx[pairs.user_id.values], pairs.product_id.values.astype(np.int64)
        V = item2vec(h, sp_.user_id[sp_.split == "train"].values, P)
        np.save(OUT / f"i2v_fold{fold}.npy", V[:, :EMB])
        cols = {**sims(V, "i2v", mats, pu, pp, P), **sims(txt, "txt", mats, pu, pp, P)}
        out = pd.concat([pairs.reset_index(drop=True), pd.DataFrame(cols)], axis=1)
        out.to_parquet(OUT / f"fold{fold}.parquet", index=False)
        cov = (V[pairs.product_id.values] != 0).any(1).mean()
        print(f"fold {fold}: i2v vector for {cov:.1%} of pairs; coverage " + ", ".join(f"{c} {out[c].notna().mean():.0%}" for c in cols) + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
