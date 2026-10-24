"""Invariants of the built token streams, and one decoded client to read.

Checks per fold that event_start points at [EVT], that each client's first event follows its [USR], that the
[USR] count matches the client count, that events are time-ordered within a client and how much of the stream
is [UNK]. Then prints the first client of fold 1 decoded and the share of stream tokens per key.

Example: `uv run egpt check-tokens`
"""

from __future__ import annotations

import numpy as np

from eventgpt.data.mbd import N_FOLDS, OUT
from eventgpt.data.vocab import EVT, UNK, USR, Vocab


def check_fold(k: int, vocab: Vocab) -> None:
    """Assert the stream invariants of one fold and print its [UNK] share."""
    d = OUT / f"fold={k}"
    s = np.load(d / "tokens.npy", mmap_mode="r")
    ev = np.load(d / "client_events.npy")
    est = np.load(d / "event_start.npy")
    et = np.load(d / "event_time.npy")
    assert (s[est] == vocab.id(EVT)).all(), "event_start must point at [EVT]"
    assert (s[est[ev[:-1]] - 1] == vocab.id(USR)).all(), "the token before a client's first event must be [USR]"
    assert (s == vocab.id(USR)).sum() == len(ev) - 1, "one [USR] per client"
    owner = np.repeat(np.arange(len(ev) - 1), np.diff(ev))
    within = owner[1:] == owner[:-1]
    assert (np.diff(et)[within] >= 0).all(), "events must be time-ordered within a client"
    print(f"fold {k}: ok, {len(ev) - 1:,} clients, {len(est):,} events, [UNK] share {(s == vocab.id(UNK)).mean():.5f}")


def main() -> None:
    """Check every fold, decode one client, report the token share per key."""
    vocab = Vocab.load(OUT / "vocab")
    names = {i: t for t, i in vocab.structured.items()}
    for k in range(N_FOLDS):
        check_fold(k, vocab)
    d = OUT / "fold=1"
    s = np.load(d / "tokens.npy", mmap_mode="r")
    est = np.load(d / "event_start.npy")
    print(" ".join(names[int(t)] for t in s[est[0] - 1 : est[12]]).replace(EVT, "\n" + EVT))
    counts = np.bincount(np.load(OUT / "pretrain" / "train_tokens.npy", mmap_mode="r")[:50_000_000], minlength=vocab.size)
    share: dict[str, int] = {}
    for i, c in enumerate(counts):
        key = names[i].split("=")[0]
        share[key] = share.get(key, 0) + int(c)
    total = sum(share.values())
    print({k: round(c / total, 3) for k, c in sorted(share.items(), key=lambda x: -x[1]) if c})


if __name__ == "__main__":
    main()
