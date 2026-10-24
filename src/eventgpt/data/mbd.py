"""Build EventGPT token streams from MBD-mini (ai-lab/MBD-mini on the Hugging Face Hub).

Each client's stream opens with a profile event and then merges two event kinds in time order:

    profile  [EVT] name=profile dt=140 s21 s22 s31 s32
    trx      [EVT] name=trx_<event_type> dt=<b> sub=<subtype> amt_bin=<b> [cur=<c>] s11 s12 d11 d12
    geo      [EVT] name=geo dt=<b> g4=<cell> g5=<cell>

src_type21, src_type22, src_type31 and src_type32 never change within a client (measured on fold 1), so they
are client attributes rather than transaction fields. Repeated on every transaction they took 4 of 13 tokens;
the profile event states them once, stamped at the client's first event so any non-empty history prefix
includes it. The first real event after it still gets the top dt bucket.

Geo is thinned to one event per client per Moscow day: the day's most frequent geohash_5 with the
geohash_4 it sits in, stamped at the day's last ping so the event never runs ahead of what it summarises.
Raw geo is 66M pings against 39M transactions; one a day is 15M. Dialogue embeddings are left out.

Vocabulary values and amount bins are fitted on the pretraining folds only. Rare values of the
counterparty-like fields and of the geo cells fold into <key>=other. `currency` is emitted only when it is
not the dominant currency, since 99.9% of rows share one.

Per fold the build writes, under OUT/fold=<k>/:

    tokens.npy          int16 stream, [USR] then the client's events, client after client
    client_ids.npy      client id per client, in stream order
    client_events.npy   event index range per client, length n_clients + 1
    event_start.npy     token offset of each event's [EVT]; event_start[client_events[i]] - 1 is the client's [USR]
    event_time.npy      UTC epoch seconds per event

and OUT/pretrain/ holds train_tokens.npy and val_tokens.npy, the packed streams of every fold except the test
fold, split by client hash, which is what train.py samples windows from.

    uv run python -m eventgpt.data.mbd --test-fold 0
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from eventgpt.data.vocab import EVT, USR, Vocab

RAW = Path(os.environ.get("MBD_ROOT", "/Volumes/WD_BLACK/datasets/MBD-mini/extracted"))
OUT = Path(os.environ.get("MBD_OUT", "/Volumes/WD_BLACK/datasets/MBD-mini/tokens"))

N_FOLDS = 5
MSK = "interval 3 hour"  # timestamps are UTC and the bank's day starts at 21:00 UTC
DOMINANT_CURRENCY = 11
TRX_KEYS = {  # source column -> token key, all emitted as <key>=<value>
    "event_subtype": "sub",
    "src_type11": "s11",
    "src_type12": "s12",
    "dst_type11": "d11",
    "dst_type12": "d12",
}
PROFILE_KEYS = {"src_type21": "s21", "src_type22": "s22", "src_type31": "s31", "src_type32": "s32"}
TRX, GEO, PROFILE = 0, 1, 2
MIN_COUNT = {"s21": 1000, "s31": 1000, "g4": 100, "g5": 300}  # rarer values become <key>=other
EVENT_WIDTH = 10
VAL_SHARE = 0.02  # pretraining clients held out for the language-model validation loss


# -------------------------------------------------------------------------------
# Reading
# -------------------------------------------------------------------------------
def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"create view trx as select * from read_parquet('{RAW}/detail/trx/*/*.parquet', hive_partitioning=true)")
    con.execute(f"create view geo_raw as select * from read_parquet('{RAW}/detail/geo/*/*.parquet', hive_partitioning=true)")
    # one geo event per client per Moscow day: the most visited geohash_5, at the day's last ping; materialised
    # once, since every fold reads it
    con.execute(f"""
        create temp table geo as
        with c as (
            select client_id, fold, date_trunc('day', event_time + {MSK}) d, geohash_4, geohash_5, count(*) n
            from geo_raw group by all
        ),
        last_ping as (
            select client_id, date_trunc('day', event_time + {MSK}) d, max(event_time) t from geo_raw group by all
        ),
        top as (
            select *, row_number() over (partition by client_id, d order by n desc, geohash_5) rk from c
        )
        select top.client_id, top.fold, last_ping.t event_time, top.geohash_4, top.geohash_5
        from top join last_ping using (client_id, d) where rk = 1
    """)
    return con


def fit_vocab(con: duckdb.DuckDBPyConnection, folds: list[int]) -> Vocab:
    """Value tables and amount bins from the pretraining folds."""
    where = f"fold in ({', '.join(map(str, folds))})"
    cat: dict[str, list[object]] = {}

    def kept(table: str, col: str, key: str) -> list[object]:
        floor = MIN_COUNT.get(key, 1)
        rows = con.execute(f"select coalesce({col}, -1) v, count(*) n from {table} where {where} group by 1").fetchall()
        vals: list[object] = [v for v, n in rows if n >= floor]
        return [*vals, "other"] if len(vals) < len(rows) else vals

    names: list[object] = [f"trx_{v}" for v in kept("trx", "event_type", "name")]
    cat["name"] = [*names, "geo", "profile"]
    for col, key in (TRX_KEYS | PROFILE_KEYS).items():
        cat[key] = kept("trx", col, key)
    cat["cur"] = [v for v in kept("trx", "currency", "cur") if v != DOMINANT_CURRENCY]
    cat["g4"] = kept("geo", "geohash_4", "g4")
    cat["g5"] = kept("geo", "geohash_5", "g5")
    amounts = con.execute(f"select amount from trx where {where} and amount > 0").fetchnumpy()["amount"]
    return Vocab.fit(cat, np.asarray(amounts, dtype=np.float64))


def load_fold(con: duckdb.DuckDBPyConnection, fold: int) -> pd.DataFrame:
    """Every event of one fold in one frame, sorted by client then time, each client's profile event first."""
    trx_cols = ", ".join(f"coalesce({c}, -1) {c}" for c in TRX_KEYS)
    null_trx = ", ".join(f"null {c}" for c in TRX_KEYS)
    prof_cols = ", ".join(f"coalesce(max({c}), -1) {c}" for c in PROFILE_KEYS)
    null_prof = ", ".join(f"null {c}" for c in PROFILE_KEYS)
    return con.execute(f"""
        with e as (
            select client_id, epoch(event_time)::double t, {TRX} kind, coalesce(event_type, -1) event_type, amount,
                   coalesce(currency, -1) currency, {trx_cols}, {null_prof}, null::integer geohash_4, null::integer geohash_5
            from trx where fold = {fold}
            union all
            select client_id, epoch(event_time)::double t, {GEO} kind, null, null, null, {null_trx}, {null_prof}, geohash_4, geohash_5
            from geo where fold = {fold}
        ),
        prof as (
            select client_id, {prof_cols} from trx where fold = {fold} group by client_id
        ),
        p as (
            select e.client_id, min(e.t) t from e group by e.client_id
        )
        select * from (
        select * from e
        union all
        select p.client_id, p.t, {PROFILE}, null, null, null, {null_trx}, {", ".join(f"prof.{c}" for c in PROFILE_KEYS)}, null, null
        from p join prof using (client_id)
        )
        order by client_id, t, kind != {PROFILE}, kind
    """).df()


# -------------------------------------------------------------------------------
# Tokenizing
# -------------------------------------------------------------------------------
def _codes(vocab: Vocab, key: str, col: pd.Series) -> np.ndarray:
    """Token id per row for one key; values outside the vocabulary become <key>=other, or [UNK] if it has none."""
    vals = col.astype("float64").fillna(-1).astype(np.int64)
    lut = {v: vocab.structured[f"{key}={v}"] for v in vals.unique() if f"{key}={v}" in vocab.structured}
    return vals.map(lut).fillna(vocab.id(f"{key}=other")).to_numpy(dtype=np.int64)


def tokenize(df: pd.DataFrame, vocab: Vocab) -> tuple[np.ndarray, np.ndarray]:
    """Fill a fixed (events, EVENT_WIDTH) grid and drop its padding.

    Returns:
        (flat token ids, token offset of each event's [EVT] within the flat array)
    """
    n = len(df)
    pad = vocab.id("[PAD]")
    mat = np.full((n, EVENT_WIDTH), pad, dtype=np.int32)
    kind = df["kind"].to_numpy()
    is_trx, is_geo, is_prof = kind == TRX, kind == GEO, kind == PROFILE
    client = df["client_id"].to_numpy()
    first = np.ones(n, dtype=bool)
    first[1:] = (client[1:] != client[:-1]) | is_prof[:-1]
    gap = np.diff(df["t"].to_numpy(), prepend=0.0)
    gap[first] = 1e12  # a client's first event, and the profile, land in the top bucket
    mat[:, 0] = vocab.id(EVT)
    etype = df["event_type"].astype("float64").fillna(-1).astype(np.int64)
    name_lut = {v: vocab.structured[f"name=trx_{v}"] for v in etype.unique() if f"name=trx_{v}" in vocab.structured}
    mat[:, 1] = np.where(is_trx, etype.map(name_lut).fillna(vocab.id("name=trx_other")).to_numpy(dtype=np.int64), vocab.id("name=geo"))
    mat[is_prof, 1] = vocab.id("name=profile")
    mat[:, 2] = vocab.dt_tokens(gap)
    # transactions: subtype, amount, currency when unusual, then the four source and destination types
    mat[is_trx, 3] = _codes(vocab, "sub", df["event_subtype"])[is_trx]
    mat[is_trx, 4] = vocab.amount_tokens(df["amount"].to_numpy(dtype=np.float64))[is_trx]
    cur = df["currency"].to_numpy()
    odd = is_trx & (cur != DOMINANT_CURRENCY)
    mat[odd, 5] = _codes(vocab, "cur", df["currency"])[odd]
    for j, (col, key) in enumerate(list(TRX_KEYS.items())[1:]):
        mat[is_trx, 6 + j] = _codes(vocab, key, df[col])[is_trx]
    # profile: the client's static attributes
    for j, (col, key) in enumerate(PROFILE_KEYS.items()):
        mat[is_prof, 3 + j] = _codes(vocab, key, df[col])[is_prof]
    # geo: the day's cell at two resolutions
    mat[is_geo, 3] = _codes(vocab, "g4", df["geohash_4"])[is_geo]
    mat[is_geo, 4] = _codes(vocab, "g5", df["geohash_5"])[is_geo]
    keep = mat != pad
    counts = keep.sum(axis=1)
    flat = mat[keep].astype(np.int16)
    starts = np.cumsum(counts) - counts
    return flat, starts


def write_fold(df: pd.DataFrame, vocab: Vocab, out: Path) -> dict[str, int]:
    """Tokenize one fold, insert [USR] at each client boundary and write the stream with its indexes."""
    flat, starts = tokenize(df, vocab)
    client = df["client_id"].to_numpy()
    change = np.ones(len(df), dtype=bool)
    change[1:] = client[1:] != client[:-1]
    first_event = np.flatnonzero(change)
    stream = np.insert(flat, starts[first_event], np.int16(vocab.id(USR)))
    # each event's [EVT] moves right by the number of [USR] tokens inserted at or before its client
    shift = np.cumsum(change)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "tokens.npy", stream)
    np.save(out / "client_ids.npy", client[first_event].astype(str))
    np.save(out / "client_events.npy", np.append(first_event, len(df)).astype(np.int64))
    np.save(out / "event_start.npy", (starts + shift).astype(np.int64))
    np.save(out / "event_time.npy", df["t"].to_numpy().astype(np.int64))
    return {"events": len(df), "clients": len(first_event), "tokens": len(stream)}


# -------------------------------------------------------------------------------
# Pretraining streams
# -------------------------------------------------------------------------------
def client_slices(fold_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(client ids, token start of each client's [USR], token end) for a written fold."""
    ids = np.load(fold_dir / "client_ids.npy")
    ev = np.load(fold_dir / "client_events.npy")
    est = np.load(fold_dir / "event_start.npy")
    n_tok = len(np.load(fold_dir / "tokens.npy", mmap_mode="r"))
    begin = est[ev[:-1]] - 1
    end = np.append(begin[1:], n_tok)
    return ids, begin, end


def write_pretrain(folds: list[int]) -> dict[str, int]:
    """Concatenate the pretraining folds into train and val streams, split by a stable hash of the client id."""
    parts: dict[str, list[np.ndarray]] = {"train": [], "val": []}
    for k in folds:
        d = OUT / f"fold={k}"
        stream = np.load(d / "tokens.npy")
        ids, begin, end = client_slices(d)
        is_val = pd.util.hash_array(ids.astype(object)) % 10_000 < VAL_SHARE * 10_000
        token_is_val = np.repeat(is_val, end - begin)  # clients tile the stream, so this is one flag per token
        parts["train"].append(stream[~token_is_val])
        parts["val"].append(stream[token_is_val])
    out = OUT / "pretrain"
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for split, chunks in parts.items():
        s = np.concatenate(chunks)
        np.save(out / f"{split}_tokens.npy", s)
        sizes[f"{split}_tokens"] = len(s)
    return sizes


def main() -> None:
    """Fit the vocabulary on the pretraining folds, tokenize every fold, pack the pretraining streams."""
    p = argparse.ArgumentParser()
    p.add_argument("--test-fold", type=int, default=0, help="fold held out of pretraining and vocabulary fitting")
    p.add_argument("--no-geo", action="store_true", help="leave geolocation events out of every stream; the vocabulary keeps geo tokens")
    a = p.parse_args()
    pre = [k for k in range(N_FOLDS) if k != a.test_fold]
    con = _con()
    t0 = time.time()
    vocab = fit_vocab(con, pre)
    vocab.save(OUT / "vocab")
    print(f"vocab {vocab.size} tokens, fingerprint {vocab.fingerprint()[:12]} ({time.time() - t0:.0f}s)", flush=True)
    for k in range(N_FOLDS):
        df = load_fold(con, k)
        if a.no_geo:
            df = df[df["kind"] != GEO].reset_index(drop=True)
        stats = write_fold(df, vocab, OUT / f"fold={k}")
        print(f"fold {k}: {stats} ({time.time() - t0:.0f}s)", flush=True)
        del df
    (OUT / "test_fold.txt").write_text(f"{a.test_fold}\n")
    print(f"pretrain folds {pre}: {write_pretrain(pre)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
