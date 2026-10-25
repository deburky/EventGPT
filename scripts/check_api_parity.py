"""Check that eventgpt.api reproduces the probe: same tokens from raw MBD rows, same embeddings as the paper's.

For a sample of test-fold client-months it reads the clients' raw transactions and geolocation pings,
tokenizes them through eventgpt.api, and compares (1) each history's tokens with those the probe cuts from
the token streams of `egpt prepare`, exactly, and (2) the API's embedding with the one the probe stored for
the paper, run on another device, so to a tolerance.

Example: `uv run python scripts/check_api_parity.py --hub /path/to/hub/aligned-s42 --probe-dir data/runs/hier-alq-s42/probe-now-2k`
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from eventgpt import api
from eventgpt.data.mbd import RAW
from eventgpt.evaluation.probe import Fold, history_tokens
from eventgpt.hub import load_folder
from eventgpt.training.train import device


def main() -> None:
    """Compare tokens and embeddings for a sample of test rows."""
    p = argparse.ArgumentParser()
    p.add_argument("--hub", type=Path, required=True, help="an exported model folder (eventgpt.hub)")
    p.add_argument("--probe-dir", type=Path, required=True, help="the probe output of the same checkpoint")
    p.add_argument("--n", type=int, default=500, help="test rows to check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-geo", action="store_true", help="a transactions-only model: pings only date the profile event")
    a = p.parse_args()

    t0 = time.time()
    model, vocab = load_folder(a.hub)
    model = model.to(device())
    rows = pd.read_parquet(a.probe_dir / "test_rows.parquet")
    stored = np.load(a.probe_dir / "embeddings.npz")["test_last"]
    pick = np.random.default_rng(a.seed).choice(np.flatnonzero(rows["n_hist"].to_numpy() > 0), a.n, replace=False)
    sample = rows.iloc[pick].reset_index(drop=True)
    ids = ", ".join(f"'{c}'" for c in sample["client_id"].unique())
    src = "select * exclude (fold) from read_parquet('{}/detail/{}/*/*.parquet', hive_partitioning=true) where client_id in ({})"
    trx = duckdb.sql(src.format(RAW, "trx", ids)).df()
    geo = duckdb.sql(src.format(RAW, "geo", ids)).df()
    cuts = pd.DataFrame({"client_id": sample["client_id"], "cut": pd.to_datetime(sample["cut"], unit="s")})
    print(f"{len(sample)} rows of {sample['client_id'].nunique()} clients, {len(trx):,} transactions, {len(geo):,} pings", flush=True)

    # tokens: the API's sequences against the probe's, cut from the prepared fold
    aligned = bool(model.cfg.aligned)
    evt = np.array([vocab.id("[EVT]")])
    mem = api.stream(vocab, trx, geo, geo_events=not a.no_geo)
    got, _, n_hist = api.sequences(mem, cuts, 2048, aligned, evt)
    fold = Fold(int(sample["fold"].iloc[0]), vocab)
    ci = sample["client_id"].map(fold.index).to_numpy()
    want_n = sample["n_hist"].to_numpy()
    last = fold.event_time[np.maximum(fold.client_events[ci] + want_n - 1, 0)]
    gap = vocab.dt_tokens(np.maximum(sample["cut"].to_numpy() - last, 0).astype(np.float64))
    want = [history_tokens(fold, int(c), int(n), 2048, np.r_[evt, g], aligned) for c, n, g in zip(ci, want_n, gap, strict=True)]
    same = [np.array_equal(x, y) for x, y in zip(got, want, strict=True)]
    print(f"history lengths equal: {np.array_equal(n_hist, want_n)}; token sequences equal: {sum(same)}/{len(same)}")

    # embeddings: the API's against those stored by the probe for the paper
    emb = api.embed(model, vocab, cuts, trx, geo, geo_events=not a.no_geo)
    ref = stored[pick]
    cos = (emb * ref).sum(1) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(ref, axis=1))
    rel = np.abs(emb - ref).max(1) / np.abs(ref).max(1)
    print(f"cosine to the paper's embeddings: min {cos.min():.6f}, median {np.median(cos):.6f}")
    print(f"max abs difference relative to the row's largest value: median {np.median(rel):.2e}, max {rel.max():.2e}")
    print(f"done in {time.time() - t0:.0f}s on {device()}")


if __name__ == "__main__":
    main()
