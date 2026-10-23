"""Training windows: random slices of the packed stream, or client-aligned windows laid out as the probe reads a history.

A slice at a random offset of the packed stream usually starts inside a client's history, without [USR] or the
profile event and with nothing that says how long the client has banked. An aligned window that starts inside a
client gets [USR] and the client's first event put back in front, as eventgpt.probe cuts a long history, and the
first event after them carries the gap from the client's first event instead of from the event cut away before
it. Every dt token then measures the gap to the previous event in the sequence the model reads, and the time
positions of the history decoder count from the client's first event, so tenure is visible to the model.
eventgpt.probe.history_tokens applies the same rule to checkpoints trained with aligned windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from eventgpt.mbd import N_FOLDS, OUT, VAL_SHARE, client_slices
from eventgpt.vocab import Vocab


class PackedStream:
    """Random windows of the packed pretraining stream, as mbd.py wrote it."""

    def __init__(self, split: str) -> None:
        self.stream = np.load(OUT / "pretrain" / f"{split}_tokens.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.stream)

    def batch(self, block: int, bsz: int, rng: np.random.Generator, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Random windows of `block` tokens, with the stream shifted by one as targets."""
        ix = rng.integers(0, len(self.stream) - block - 1, size=bsz)
        w = np.stack([self.stream[i : i + block + 1] for i in ix]).astype(np.int64)
        return torch.from_numpy(np.ascontiguousarray(w[:, :-1])).to(dev), torch.from_numpy(np.ascontiguousarray(w[:, 1:])).to(dev)


class AlignedStream:
    """Client-aligned windows over the pretraining clients of one split, rebuilt from the fold files.

    The clients and their order are those of mbd.write_pretrain, so the stream holds the same tokens as
    pretrain/<split>_tokens.npy except the dt of each client's second event, which here is the gap from the
    first event (the profile event, stamped at the client's first transaction or ping).
    """

    def __init__(self, split: str, vocab: Vocab) -> None:
        test_fold = int((OUT / "test_fold.txt").read_text())
        streams, starts, times, firsts, usrs = [], [], [], [], []
        off, n_ev = 0, 0
        for k in [k for k in range(N_FOLDS) if k != test_fold]:
            d = OUT / f"fold={k}"
            ids, begin, end = client_slices(d)
            is_val = pd.util.hash_array(ids.astype(object)) % 10_000 < VAL_SHARE * 10_000
            keep = is_val if split == "val" else ~is_val
            ce = np.load(d / "client_events.npy")
            lens = (end - begin)[keep]
            new_begin = off + np.cumsum(lens) - lens
            shift = np.zeros(len(ids), dtype=np.int64)
            shift[keep] = new_begin - begin[keep]
            ev_client = np.repeat(np.arange(len(ids)), np.diff(ce))
            ev_kept = keep[ev_client]
            streams.append(np.load(d / "tokens.npy")[np.repeat(keep, end - begin)])
            starts.append(np.load(d / "event_start.npy")[ev_kept] + shift[ev_client[ev_kept]])
            times.append(np.load(d / "event_time.npy")[ev_kept])
            counts = np.diff(ce)[keep]
            firsts.append(n_ev + np.cumsum(counts) - counts)
            usrs.append(new_begin)
            off += int(lens.sum())
            n_ev += int(ev_kept.sum())
        self.stream = np.concatenate(streams)
        self.ev_start = np.concatenate(starts)
        self.ev_time = np.concatenate(times).astype(np.float64)
        self.first = np.concatenate(firsts)  # event index of each client's first event
        self.usr = np.concatenate(usrs)  # token offset of each client's [USR]
        self.ev_client = np.repeat(np.arange(len(self.first)), np.diff(np.append(self.first, n_ev)))
        self.vocab = vocab
        # a client's second event carries the gap from its first, where mbd.py gave it the top bucket
        f = self.first[np.diff(np.append(self.first, n_ev)) > 1]
        self.stream[self.ev_start[f + 1] + 2] = vocab.dt_tokens(self.ev_time[f + 1] - self.ev_time[f])

    def __len__(self) -> int:
        return len(self.stream)

    def window(self, i: int, block: int) -> np.ndarray | None:
        """The block + 1 tokens from the first event at or after offset i, or None if they run past the stream."""
        e = int(np.searchsorted(self.ev_start, i))
        if e >= len(self.ev_start):
            return None
        c = self.ev_client[e]
        f = int(self.first[c])
        if e == f:
            s = int(self.usr[c])
            return np.asarray(self.stream[s : s + block + 1]) if s + block + 1 <= len(self.stream) else None
        head_end = int(self.ev_start[f + 1])
        head = self.stream[self.usr[c] : head_end]
        s = int(self.ev_start[e])
        need = block + 1 - len(head)
        if s + need > len(self.stream):
            return None
        w = np.concatenate([head, self.stream[s : s + need]])
        if e > f + 1:
            w[len(head) + 2] = self.vocab.dt_tokens(np.array([self.ev_time[e] - self.ev_time[f]]))[0]
        return w

    def batch(self, block: int, bsz: int, rng: np.random.Generator, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Aligned windows of `block` tokens at random offsets, with targets shifted by one."""
        rows: list[np.ndarray] = []
        while len(rows) < bsz:
            w = self.window(int(rng.integers(0, len(self.stream) - block - 1)), block)
            if w is not None:
                rows.append(w)
        x = np.stack(rows).astype(np.int64)
        return torch.from_numpy(np.ascontiguousarray(x[:, :-1])).to(dev), torch.from_numpy(np.ascontiguousarray(x[:, 1:])).to(dev)
