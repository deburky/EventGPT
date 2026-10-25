"""Embeddings and zero-shot scores for event tables in memory, through the same code as the paper's probe.

Transactions come as rows with the columns of MBD-mini's detail/trx tables (client_id, event_time, amount,
event_type, event_subtype, currency, src_type11 to src_type32, dst_type11, dst_type12) and geolocation, when
the model was trained with it, as rows of detail/geo (client_id, event_time, geohash_4, geohash_5). Times are
UTC. Each history is tokenized as `egpt prepare` does (eventgpt.data.mbd), cut before its time as the probe
does (eventgpt.evaluation.probe), and ends in a "now" event at the cut.

Example:
    from eventgpt import api
    from eventgpt.hub import from_pretrained

    model, vocab = from_pretrained("deburky/EventGPT", "aligned-s42")
    cuts = pd.DataFrame({"client_id": ["c1", "c2"], "cut": pd.to_datetime(["2022-10-01", "2022-10-01"])})
    emb = api.embed(model, vocab, cuts, trx, geo)               # (2, 384)
    p = api.next_event_probs(model, vocab, cuts, trx, geo)      # (2, 5): the paper's five first-kind targets

Cuts at the start of a Moscow month, as MBD's labels use, come from eventgpt.evaluation.probe.month_cut.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from torch import nn

from eventgpt.data.mbd import GEO, build_stream, load_fold, thin_geo
from eventgpt.data.vocab import Vocab
from eventgpt.evaluation import probe
from eventgpt.evaluation.probe import FIRST_KINDS, Fold, history_tokens
from eventgpt.evaluation.zero_shot import next_event_scores, seen_before

GEO_COLUMNS = ["client_id", "event_time", "geohash_4", "geohash_5"]


# -------------------------------------------------------------------------------
# Tokenizing
# -------------------------------------------------------------------------------
def stream(vocab: Vocab, trx: pd.DataFrame, geo: pd.DataFrame | None = None, geo_events: bool = True) -> Fold:
    """Tokenize the clients in trx, with their geolocation pings when given, into an in-memory fold.

    With `geo_events` False the pings only date each client's profile event and are then dropped, as
    `egpt prepare --no-geo` built the streams of the transactions-only models.
    """
    if geo is None:
        geo = pd.DataFrame({c: pd.Series(dtype="datetime64[us]" if c == "event_time" else "int64") for c in GEO_COLUMNS})
        geo["client_id"] = geo["client_id"].astype(str)
    con = duckdb.connect()
    con.register("trx", trx)
    con.register("geo_raw", geo[GEO_COLUMNS])
    thin_geo(con, fold_col="0")
    events = load_fold(con, None)
    if not geo_events:
        events = events[events["kind"] != GEO].reset_index(drop=True)
    return Fold(0, vocab, arrays=build_stream(events, vocab))


def sequences(
    fold: Fold, cuts: pd.DataFrame, block: int, aligned: bool, head: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Token sequence per cut, ending in `head` and the gap from the last event; also client index and history length.

    The history of a cut is every event strictly before it. A cut with no event before it gets an empty sequence.
    """
    missing = set(cuts["client_id"].astype(str)) - set(fold.index)
    if missing:
        raise ValueError(f"{len(missing)} clients in cuts have no transactions, e.g. {sorted(missing)[:3]}")
    ci = cuts["client_id"].astype(str).map(fold.index).to_numpy()
    cut = pd.to_datetime(cuts["cut"]).to_numpy().astype("datetime64[s]").astype(np.int64)
    a, b = fold.client_events[ci], fold.client_events[ci + 1]
    n_hist = np.array([np.searchsorted(fold.event_time[s:e], c, side="left") for s, e, c in zip(a, b, cut, strict=True)])
    last = fold.event_time[np.maximum(a + n_hist - 1, 0)]
    gap = fold.vocab.dt_tokens(np.maximum(cut - last, 0).astype(np.float64))
    seqs = [
        history_tokens(fold, int(c), int(n), block, np.r_[head, g], aligned) if n > 0 else np.zeros(0, np.int64)
        for c, n, g in zip(ci, n_hist, gap, strict=True)
    ]
    return seqs, ci, n_hist


# -------------------------------------------------------------------------------
# Embeddings and zero-shot scores
# -------------------------------------------------------------------------------
def embed_all(
    model: nn.Module,
    vocab: Vocab,
    cuts: pd.DataFrame,
    trx: pd.DataFrame,
    geo: pd.DataFrame | None = None,
    geo_events: bool = True,
    block: int = 2048,
    batch: int = 16,
) -> dict[str, np.ndarray]:
    """Last, mean and recency-weighted history states per cut; a cut with no history before it is zero."""
    fold = stream(vocab, trx, geo, geo_events)
    seqs, _, n_hist = sequences(fold, cuts, block, bool(getattr(model.cfg, "aligned", False)), np.array([vocab.id("[EVT]")]))
    has = np.flatnonzero(n_hist > 0)
    got = probe.embed(model, [seqs[i] for i in has], next(model.parameters()).device, batch)
    out = {}
    for r, e in got.items():
        full = np.zeros((len(cuts), e.shape[1]), dtype=np.float32)
        full[has] = e
        out[r] = full
    return out


def embed(
    model: nn.Module,
    vocab: Vocab,
    cuts: pd.DataFrame,
    trx: pd.DataFrame,
    geo: pd.DataFrame | None = None,
    geo_events: bool = True,
    block: int = 2048,
    batch: int = 16,
) -> np.ndarray:
    """The history state at each cut: the embedding the paper's heads read (its "last" readout)."""
    return embed_all(model, vocab, cuts, trx, geo, geo_events, block, batch)["last"]


def next_event_probs(
    model: nn.Module,
    vocab: Vocab,
    cuts: pd.DataFrame,
    trx: pd.DataFrame,
    geo: pd.DataFrame | None = None,
    geo_events: bool = True,
    kinds: list[tuple[int, int]] = FIRST_KINDS,
    first_time: bool = True,
    block: int = 2048,
    batch: int = 16,
) -> np.ndarray:
    """(cuts, kinds) probability that the next event after each cut is a transaction of each (type, subtype).

    With `first_time`, clients who already had the kind before the cut score zero: the zero-shot rule for
    "first transaction of this kind" targets. A cut with no history before it scores NaN.
    """
    fold = stream(vocab, trx, geo, geo_events)
    head = np.array([vocab.id("[EVT]"), vocab.id(f"name=trx_{kinds[0][0]}")])
    seqs, ci, n_hist = sequences(fold, cuts, block, bool(getattr(model.cfg, "aligned", False)), head)
    out = np.full((len(cuts), len(kinds)), np.nan)
    has = np.flatnonzero(n_hist > 0)
    order = has[np.argsort([len(seqs[i]) for i in has])]
    dev = next(model.parameters()).device
    for j in range(0, len(order), batch):
        idx = order[j : j + batch]
        out[idx] = next_event_scores(model, [seqs[i] for i in idx], vocab, dev, kinds)
    if first_time:
        out[has] *= ~seen_before(fold, ci[has], n_hist[has], kinds)
    return out
