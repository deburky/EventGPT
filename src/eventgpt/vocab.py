"""Token vocabulary: key=value tokens for categorical fields, log time buckets and quantile amount bins.

Token id layout (int16 safe, total under 32k):
    specials | categorical key=value tokens | dt buckets | amount bins
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PAD, USR, EVT, UNK = "[PAD]", "[USR]", "[EVT]", "[UNK]"
SPECIALS = [PAD, USR, EVT, UNK]  # [PAD] must stay 0: embedding code builds its key mask as x != 0

N_AMOUNT_BINS = 21
MAX_TIME_BUCKET = 140


def time_bucket(seconds: np.ndarray) -> np.ndarray:
    """PRAGMA time encoding 8*ln(1+t/8), floored to an integer bucket."""
    s = np.clip(np.nan_to_num(seconds, nan=0.0), 0, None)
    return np.minimum(np.floor(8.0 * np.log1p(s / 8.0)), MAX_TIME_BUCKET).astype(np.int64)


@dataclass
class Vocab:
    """Token table: specials, then key=value tokens, then dt and amount tokens."""

    structured: dict[str, int] = field(default_factory=dict)
    amount_edges: list[float] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Total vocabulary size."""
        return len(self.structured)

    def add(self, token: str) -> int:
        """Register a token and return its id."""
        if token not in self.structured:
            self.structured[token] = len(self.structured)
        return self.structured[token]

    def id(self, token: str) -> int:
        """Id of a token, [UNK] when unknown."""
        return self.structured.get(token, self.structured[UNK])

    def ids(self, tokens: Iterable[str]) -> np.ndarray:
        """Ids for an iterable of tokens, [UNK] when unknown."""
        unk = self.structured[UNK]
        return np.fromiter((self.structured.get(t, unk) for t in tokens), dtype=np.int64)

    @classmethod
    def fit(cls, categorical: Mapping[str, Iterable[object]], amounts: np.ndarray) -> Vocab:
        """Build the vocabulary from training data.

        Args:
            categorical: key -> kept values; emits one <key>=<value> token per value
            amounts: positive amounts whose quantiles become the amount bins
        """
        v = cls()
        for s in SPECIALS:
            v.add(s)
        for key, values in categorical.items():
            for val in sorted({str(x) for x in values}):
                v.add(f"{key}={val}")
        for b in range(MAX_TIME_BUCKET + 1):
            v.add(f"dt={b}")
        v.add("amt_bin=zero")
        for b in range(N_AMOUNT_BINS):
            v.add(f"amt_bin={b}")
        pos = amounts[np.isfinite(amounts) & (amounts > 0)]
        v.amount_edges = [float(x) for x in np.quantile(pos, np.linspace(0, 1, N_AMOUNT_BINS + 1)[1:-1])]
        return v

    def table(self, key: str, values: Iterable[object]) -> dict[object, int]:
        """Raw value -> token id for one key, so a column maps through a dict lookup rather than string building."""
        return {x: self.structured[f"{key}={x}"] for x in values if f"{key}={x}" in self.structured}

    def amount_tokens(self, amounts: np.ndarray) -> np.ndarray:
        """Bin token ids for an array of non-negative amounts; zero and NaN become `amt_bin=zero`."""
        a = np.nan_to_num(amounts, nan=0.0)
        bins = np.digitize(a, self.amount_edges)
        bin_ids = np.array([self.id(f"amt_bin={b}") for b in range(N_AMOUNT_BINS)])[bins]
        return np.where(a <= 0, self.id("amt_bin=zero"), bin_ids)

    def dt_tokens(self, seconds: np.ndarray) -> np.ndarray:
        """Token ids of the time buckets of gaps in seconds."""
        lut = np.array([self.id(f"dt={b}") for b in range(MAX_TIME_BUCKET + 1)])
        return lut[time_bucket(seconds)]

    def save(self, d: Path) -> None:
        """Write the token table and amount edges into directory d."""
        d.mkdir(parents=True, exist_ok=True)
        (d / "vocab.json").write_text(json.dumps({"tokens": self.structured, "amount_edges": self.amount_edges}))

    @classmethod
    def load(cls, d: Path) -> Vocab:
        """Read a vocabulary written by save()."""
        meta = json.loads((d / "vocab.json").read_text())
        return cls(structured=meta["tokens"], amount_edges=meta["amount_edges"])

    def fingerprint(self) -> str:
        """Hash of the id mapping and amount edges, stored with checkpoints so a same-sized other vocabulary is refused."""
        body = json.dumps({"tokens": dict(sorted(self.structured.items())), "amount_edges": self.amount_edges}, sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()
