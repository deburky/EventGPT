"""The paper's figures in one style, diagrams and plots alike drawn as SVG and converted to PDF with rsvg-convert.

Style: Inter for text and JetBrains Mono for tokens; color marks what a part of the model is (tokens, event
vectors, hidden states, the next-event decoder, the embedding), with a legend on every diagram; flat rounded
boxes, thin ink arrows, light containers.

Run from the repository root: `uv run python paper/figures/src/figures.py` (needs rsvg-convert, Inter and
JetBrains Mono installed, and the probe output of hier-alq-s42 for the two plots).
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
RUN = Path("data/runs/hier-alq-s42/probe-now-2k")

INK = "#1B2A3A"
MUTED = "#5B6B7C"
LINE = "#D3DAE2"
TOK = "#E2F4B0"
ENC, ENC_BAND = "#C7ECE8", "#EBF7F6"
HID, HID_BAND = "#CADEF9", "#EDF3FD"
DEC, DEC_BAND = "#DDD9FA", "#F3F1FE"
EMB = "#14283C"
CHURN = "#4F5BD5"
SANS = "'Inter 18pt'"  # the static cut, quoted for CSS, so weights resolve
MONO = "JetBrains Mono"


# -------------------------------------------------------------------------------
# SVG primitives
# -------------------------------------------------------------------------------
class Svg:
    """A diagram built up as SVG elements, written out and converted to PDF."""

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.items: list[str] = []

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", r: float = 10, dash: str = "") -> None:
        """A rounded rectangle."""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"{d}/>'
        )

    def text(
        self, x: float, y: float, s: str, size: float = 11, weight: int = 400, fill: str = INK, anchor: str = "start", mono: bool = False
    ) -> None:
        """Text at a baseline; `s` may hold <tspan> markup."""
        fam = MONO if mono else SANS
        self.items.append(
            f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}">{s}</text>'
        )

    def arrow(self, pts: list[tuple[float, float]], color: str = INK) -> None:
        """A polyline ending in an arrowhead."""
        d = " ".join(f"{x},{y}" for x, y in pts)
        self.items.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="1.4" marker-end="url(#head)"/>')

    def pill(self, x: float, y: float, s: str, fill: str = TOK) -> float:
        """A token in a rounded pill; returns its width."""
        w = len(s) * 6.0 + 12
        self.rect(x, y, w, 19, fill, r=6)
        self.text(x + w / 2, y + 13.2, s.replace("&", "&amp;").replace("<", "&lt;"), size=10, fill=INK, anchor="middle", mono=True)
        return w

    def legend(self, items: list[tuple[str, str]], y: float = 18) -> None:
        """Color keys in a row along the top."""
        x = 20.0
        for fill, label in items:
            self.rect(x, y - 10, 12, 12, fill, LINE if fill.upper() in ("WHITE", "#FFFFFF") else "none", r=3)
            self.text(x + 18, y, label, size=11, fill=MUTED)
            x += 18 + len(label) * 6.1 + 22

    def save(self, name: str) -> None:
        """Write name.svg to the sources and name.pdf beside the paper."""
        head = (
            '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,1 L9,5 L0,9 z" fill="{INK}"/></marker></defs>'
        )
        body = "\n".join(self.items)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
            f'{head}<rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>'
        )
        (SRC / f"{name}.svg").write_text(svg)
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(OUT / f"{name}.pdf"), str(SRC / f"{name}.svg")], check=True)


def sub(base: str, i: str) -> str:
    """A symbol with a subscript, in Unicode subscript digits."""
    return base + i.translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


# -------------------------------------------------------------------------------
# Diagrams
# -------------------------------------------------------------------------------
def hierarchy() -> None:
    """The three stages: event encoder, history decoder, token decoder."""
    g = Svg(760, 500)
    g.legend([(TOK, "Tokens"), (ENC, "Event vectors"), (HID, "Hidden states"), (DEC, "Next-event decoder"), (EMB, "Customer embedding")])
    cols = [105, 285, 465]
    events = [
        ("Event 1: profile", ["[EVT]", "name=profile", "dt=140", "s21=18808", "s22=86"]),
        ("Event 2: transaction", ["[EVT]", "name=trx_1", "dt=79", "sub=12", "amt_bin=19", "d12=21619"]),
        ("Event 3: geolocation", ["[EVT]", "name=geo", "dt=74", "g4=318", "g5=4410"]),
    ]
    for c, (title, toks) in zip(cols, events, strict=True):
        x0, y0 = c - 85.0, 46.0
        g.rect(x0, y0, 170, 116, "white", LINE, r=12)
        g.text(x0 + 12, y0 + 21, title, size=11.5, weight=600)
        x, y = x0 + 10, y0 + 34
        for t in toks:
            w = len(t) * 6.0 + 12
            if x + w > x0 + 162:
                x, y = x0 + 10, y + 25
            x += g.pill(x, y, t) + 5
        g.arrow([(c, y0 + 116), (c, 204)])
    # event encoder
    g.rect(20, 206, 530, 46, ENC_BAND, r=12)
    g.text(285, 225, "Event encoder", size=12.5, weight=600, anchor="middle")
    g.text(285, 242, "2 bidirectional layers within each event, mean-pooled to one vector", size=10.5, fill=MUTED, anchor="middle")
    for k, c in enumerate(cols, 1):
        g.arrow([(c, 252), (c, 270)])
        g.rect(c - 30, 272, 60, 28, ENC, r=8)
        g.text(c, 291, sub("e", str(k)), size=12, weight=600, anchor="middle")
    # history decoder
    g.rect(20, 326, 530, 96, HID_BAND, r=12)
    for k, c in enumerate(cols, 1):
        g.rect(c - 30, 346, 60, 28, HID, r=8)
        g.text(c, 365, sub("h", str(k)), size=12, weight=600, anchor="middle")
        if k < 3:
            g.arrow([(c + 30, 360), (cols[k] - 32, 360)])
        g.arrow([(c, 300), (c, 344)])
    g.text(285, 397, "History decoder", size=12.5, weight=600, anchor="middle")
    g.text(285, 413, "6 causal layers over events, positions from elapsed time", size=10.5, fill=MUTED, anchor="middle")
    # token decoder and the next event
    g.arrow([(495, 360), (578, 360)])
    g.rect(580, 326, 160, 70, DEC_BAND, DEC, r=12)
    g.text(660, 351, "Token decoder", size=12.5, weight=600, anchor="middle")
    g.text(660, 368, "2 causal layers within", size=10.5, fill=MUTED, anchor="middle")
    g.text(660, 383, "the next event, given " + sub("h", "3"), size=10.5, fill=MUTED, anchor="middle")
    g.arrow([(660, 326), (660, 266)])
    g.text(668, 300, "next-token", size=10, fill=MUTED)
    g.text(668, 313, "probabilities", size=10, fill=MUTED)
    g.rect(580, 182, 160, 82, "white", DEC, r=12)
    g.text(592, 203, "Next event " + sub("e", "4"), size=11.5, weight=600)
    x, y = 590.0, 214.0
    for t in ["[EVT]", "name=trx_7", "dt=12", "sub=3"]:
        w = len(t) * 6.0 + 12
        if x + w > 732:
            x, y = 590.0, y + 25
        x += g.pill(x, y, t, fill=DEC) + 5
    # embedding
    g.arrow([(465, 374), (465, 442)])
    g.rect(345, 444, 240, 46, EMB, r=12)
    g.text(465, 463, "Customer embedding", size=12.5, weight=600, fill="white", anchor="middle")
    g.text(465, 480, "last hidden state at the as-of date", size=10.5, fill="#BFD0E2", anchor="middle")
    g.save("hierarchy")


def attention() -> None:
    """Attention within an event and across events."""
    g = Svg(760, 290)
    g.legend([(ENC, "Attends within an event"), (HID, "Attends across events"), ("#FFFFFF", "Masked")])
    g.items.append(
        f'<rect x="{20 + 18 + 23 * 6.1 + 22 + 18 + 21 * 6.1 + 22}" y="8" width="12" height="12" rx="3" '
        f'fill="white" stroke="{LINE}" stroke-width="1.2"/>'
    )
    panels = [
        (20, "Within an event", "Bidirectional, no positions", ["[EVT]", "name", "dt", "amt"], True),
        (390, "Across events", "Causal, within one customer", ["e1", "e2", "e3", "e4"], False),
    ]
    for x0, title, subtitle, labs, mono in panels:
        g.text(x0 + 16, 60, title, size=12.5, weight=600)
        g.text(x0 + 16, 77, subtitle, size=10.5, fill=MUTED)
        gx, gy, cell = x0 + 130, 118, 34
        g.text(gx + 4 * cell, 96, "keys →", size=10, fill=MUTED, anchor="end")
        g.text(gx - 10, 96, "queries ↓", size=10, fill=MUTED, anchor="end")
        for i, lab in enumerate(labs):
            name = lab if mono else sub("e", lab[1])
            g.text(gx + i * cell + cell / 2, gy - 7, name, size=10 if mono else 11.5, anchor="middle", mono=mono)
            g.text(gx - 10, gy + i * cell + cell / 2 + 4, name, size=10 if mono else 11.5, anchor="end", mono=mono)
            for j in range(4):
                on = mono or j <= i
                fill = (ENC if mono else HID) if on else "white"
                g.rect(gx + j * cell + 1.5, gy + i * cell + 1.5, cell - 3, cell - 3, fill, "none" if on else LINE, r=5)
        note = (
            "Tokens of " + sub("e", "3") + " also read the hidden state after " + sub("e", "2")
            if mono
            else "Each event sees itself and the events before it"
        )
        g.text(x0 + 175, 272, note, size=10.5, fill=MUTED, anchor="middle")
    g.save("attention")


def decoder() -> None:
    """What the token decoder reads: the hidden state after the previous event and its own earlier tokens."""
    g = Svg(760, 322)
    g.legend([(TOK, "Tokens"), (ENC, "Event vectors"), (HID, "Hidden states"), (DEC, "Event being decoded")])
    start = len(g.items)
    cards = [
        (16, "Event 1", ["[EVT]", "name=trx_1", "dt=9", "sub=2"], TOK),
        (262, "Event 2", ["[EVT]", "name=geo", "dt=74", "g4=318"], TOK),
        (508, "Event 3: being decoded", ["[EVT]", "name=trx_7", "dt=12"], DEC),
    ]
    tops = []
    for x0, title, toks, fill in cards:
        g.rect(x0, 226, 236, 62, "white", DEC if fill == DEC else LINE, r=12)
        g.text(x0 + 118, 306, title, size=11, weight=600, anchor="middle")
        x, mids = float(x0 + 10), []
        for t in toks:
            w = g.pill(x, 260, t, fill=fill)
            mids.append(x + w / 2)
            x += w + 5
        if fill == DEC:
            g.rect(x, 260, 42, 19, "white", DEC, r=6, dash="3 2")
            g.text(x + 21, 273.2, "?", size=10, fill=MUTED, anchor="middle", mono=True)
            ghost = x + 21
        tops.append(mids)
    # mean pool into event vectors, history decoder, h2
    for k, (c, mids) in enumerate(zip([134, 380], tops[:2], strict=True), 1):
        for m in mids:
            g.items.append(f'<line x1="{m:.1f}" y1="259" x2="{c}" y2="196" stroke="#8FD0C8" stroke-width="1"/>')
        g.rect(c - 28, 168, 56, 28, ENC, r=8)
        g.text(c, 187, sub("e", str(k)), size=12, weight=600, anchor="middle")
        g.arrow([(c, 168), (c, 138)])
    g.text(414, 186, "mean pool", size=10, fill=MUTED)
    g.rect(16, 90, 482, 46, HID_BAND, r=12)
    g.text(257, 109, "History decoder", size=12.5, weight=600, anchor="middle")
    g.text(257, 126, "6 causal layers over " + sub("e", "1") + ", " + sub("e", "2"), size=10.5, fill=MUTED, anchor="middle")
    g.arrow([(498, 113), (558, 113)])
    g.rect(560, 99, 56, 28, HID, r=8)
    g.text(588, 118, sub("h", "2"), size=12, weight=600, anchor="middle")
    # event 3 reads h2 and its own earlier tokens
    g.arrow([(700, 226), (700, 113), (618, 113)])
    g.text(692, 160, "every token", size=10, fill=MUTED, anchor="end")
    g.text(692, 173, "reads " + sub("h", "2"), size=10, fill=MUTED, anchor="end")
    first = tops[2][0]
    g.items.append(
        f'<path d="M{ghost:.1f},257 Q{(ghost + first) / 2:.1f},246 {first + 4:.1f},257" '
        f'fill="none" stroke="{INK}" stroke-width="1.2" marker-end="url(#head)"/>'
    )
    g.text((ghost + first) / 2, 240, "its own earlier tokens", size=9.5, fill=MUTED, anchor="middle")
    # no path to the tokens of earlier events
    g.items.append(f'<line x1="600" y1="324" x2="400" y2="324" stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    g.text(500, 328.5, "✕", size=13, fill=MUTED, anchor="middle")
    g.text(500, 344, "no attention to the tokens inside earlier events", size=10, fill=MUTED, anchor="middle")
    g.items[start:] = ['<g transform="translate(0,-28)">', *g.items[start:], "</g>"]
    g.save("decoder")


def timeline() -> None:
    """Event-sequence models for banking by year of publication, colored by pretraining objective."""
    g = Svg(780, 340)
    g.legend([(ENC, "Contrastive"), (HID, "Masked"), (DEC, "Generative"), ("white", "Dataset"), (EMB, "This paper")])
    # per year, in order of publication: (title, subtitle, fill)
    years = {
        2021: [("TabBERT, TabGPT", "masked and causal", HID)],
        2022: [("CoLES", "contrastive", ENC)],
        2023: [],
        2024: [("MBD", "public dataset", "white")],
        2025: [
            ("nuFormer", "Nubank, causal", DEC),
            ("LATTE", "contrastive, with an LLM", ENC),
            ("TransactionGPT", "Visa, generative", DEC),
            ("Open Banking FM", "masked, with text", HID),
        ],
        2026: [("PRAGMA", "Revolut, masked", HID), ("EventGPT", "causal, open weights", EMB)],
    }
    line_y, top, step, w, h = 62, 98, 58, 118, 48
    g.items.append(f'<line x1="20" y1="{line_y}" x2="760" y2="{line_y}" stroke="{INK}" stroke-width="1.4"/>')
    for k, (year, papers) in enumerate(years.items()):
        cx = 75 + k * 126
        if papers:
            g.items.append(f'<line x1="{cx}" y1="{line_y}" x2="{cx}" y2="{top}" stroke="{MUTED}" stroke-width="1"/>')
        g.rect(cx - 24, line_y - 11, 48, 22, "white", INK, r=11)
        g.text(cx, line_y + 4, str(year), size=10.5, weight=600, anchor="middle")
        for n, (title, subtitle, fill) in enumerate(papers):
            y0 = top + n * step
            dark = fill == EMB
            g.rect(cx - w / 2, y0, w, h, fill, LINE if fill == "white" else "none", r=10)
            g.text(cx, y0 + 20, title, size=11.5, weight=600, fill="white" if dark else INK, anchor="middle")
            g.text(cx, y0 + 36, subtitle, size=9.5, fill="#BFD0E2" if dark else MUTED, anchor="middle")
    g.save("timeline")


def windows() -> None:
    """A random training window against an aligned one, on a real client's events (fold 1), shortened to 110 tokens."""
    g = Svg(760, 236)
    g.legend([(EMB, "[USR]"), (DEC, "Profile event"), (TOK, "Transaction"), (ENC, "Geolocation")])
    px, x0, bh = 3.2, 150, 22
    kinds = ["trx"] * 13 + ["geo", "trx"] + ["geo"] * 12 + ["trx"] + ["geo"] * 11
    client = [("usr", 1), ("profile", 7)] + [(k, 9 if k == "trx" else 5) for k in kinds]
    prev = [("trx", 9), ("geo", 5), ("geo", 5), ("trx", 9)]
    fill = {"usr": EMB, "profile": DEC, "trx": TOK, "geo": ENC}
    stream = prev + client
    starts = np.cumsum([0] + [n for _, n in stream])
    w0 = int(starts[len(prev) + 2 + 3]) + 4  # four tokens into the fourth event after the profile
    w1 = w0 + 110
    n_show = 180

    def strip(y: float, blocks: list[tuple[str, int, str]], label: str, start: float = x0) -> None:
        g.text(20, y + 15, label, size=11, weight=600)
        x = float(start)
        for kind, n, mark in blocks:
            w = n * px
            dash = "3 2" if mark == "cut" else ""
            stroke = BLUE if mark == "gap" else (MUTED if mark == "cut" else "none")
            g.rect(x, y, max(w - 1.2, 1), bh, fill[kind] if mark != "cut" else "white", stroke, r=3, dash=dash)
            x += w

    # packed stream with the random window marked
    blocks, t = [], 0
    for kind, n in stream:
        if t >= n_show:
            break
        blocks.append((kind, min(n, n_show - t), ""))
        t += n
    strip(44, blocks, "Packed stream")
    bx0, bx1 = x0 + w0 * px, x0 + w1 * px
    g.items.append(f'<path d="M{bx0:.1f},72 L{bx0:.1f},78 L{bx1:.1f},78 L{bx1:.1f},72" fill="none" stroke="{INK}" stroke-width="1.2"/>')
    g.text((bx0 + bx1) / 2, 92, "a window drawn at a random offset", size=10, fill=MUTED, anchor="middle")
    # random window: starts mid-event, no [USR] or profile
    blocks = []
    for (kind, n), a in zip(stream, starts[:-1], strict=True):
        lo, hi = max(a, w0), min(a + n, w1)
        if hi > lo:
            blocks.append((kind, int(hi - lo), "cut" if (a < w0 or a + n > w1) else ""))
    strip(112, blocks, "Random window", bx0)
    # aligned window: [USR] and the profile event first, then the most recent whole events
    budget = 110 - 8
    kept: list[tuple[str, int, str]] = []
    for (kind, n), a in reversed(list(zip(stream, starts[:-1], strict=True))):
        if a + n > w1 or kind in ("usr", "profile") or a < starts[len(prev)]:
            continue
        if n > budget:
            break
        kept.insert(0, (kind, n, ""))
        budget -= n
    kept[0] = (kept[0][0], kept[0][1], "gap")
    a0 = bx1 - (8 + sum(n for _, n, _ in kept)) * px
    strip(160, [("usr", 1, ""), ("profile", 7, ""), *kept], "Aligned window", a0)
    gx = a0 + 8 * px + kept[0][1] * px / 2
    g.items.append(f'<line x1="{gx:.1f}" y1="184" x2="{gx:.1f}" y2="200" stroke="{BLUE}" stroke-width="1.2"/>')
    g.text(gx - 6, 214, "its gap token counts from the client's first event", size=10, fill=MUTED)
    g.save("windows")


def time_positions() -> None:
    """Ordinal positions against positions from elapsed time, Eq. (1)."""
    g = Svg(760, 330)
    # ordinal
    g.text(20, 52, "Ordinal positions", size=12.5, weight=600)
    g.text(
        20,
        69,
        "Standard rotary embeddings: every event is one step from the next, whether a minute or a year passed",
        size=10.5,
        fill=MUTED,
    )
    g.items.append(f'<line x1="40" y1="128" x2="720" y2="128" stroke="{INK}" stroke-width="1.2"/>')
    for k, xo in enumerate([80, 280, 480, 680]):
        g.rect(xo - 22, 88, 44, 26, ENC, r=7)
        g.text(xo, 106, sub("e", str(k + 1)), size=12, weight=600, anchor="middle")
        g.items.append(f'<line x1="{xo}" y1="124" x2="{xo}" y2="132" stroke="{INK}" stroke-width="1.2"/>')
        g.text(xo, 147, str(k), size=10.5, fill=MUTED, anchor="middle")
    # time
    g.text(20, 190, "Temporal RoPE in EventGPT", size=12.5, weight=600)
    g.text(
        20,
        207,
        "position = 64 ln(1 + minutes since the first event): a minute to an hour is as far as a day to a month",
        size=10.5,
        fill=MUTED,
    )

    def pos(minutes: float) -> float:
        return 60 + 64 * math.log1p(minutes) * (660 / (64 * math.log1p(525600)))

    g.items.append(f'<line x1="40" y1="292" x2="720" y2="292" stroke="{INK}" stroke-width="1.2"/>')
    for m, lab in [(0, "0"), (1, "1 min"), (60, "1 hour"), (1440, "1 day"), (43200, "1 month"), (525600, "1 year")]:
        x = pos(m)
        g.items.append(f'<line x1="{x:.1f}" y1="288" x2="{x:.1f}" y2="296" stroke="{INK}" stroke-width="1.2"/>')
        g.text(x, 311, lab, size=10.5, fill=MUTED, anchor="middle")
    for k, (m, lab) in enumerate([(0, "t = 0"), (5, "+5 min"), (60, "+1 hour"), (525600, "+1 year")]):
        x = pos(m)
        g.text(x, 238, lab, size=10, fill=MUTED, anchor="middle")
        g.rect(x - 22, 246, 44, 26, ENC, r=7)
        g.text(x, 264, sub("e", str(k + 1)), size=12, weight=600, anchor="middle")
    g.save("time_positions")


def delta() -> None:
    """Which clients delta scoring rescores: a new event, or a now token whose dt bucket moved since yesterday."""
    g = Svg(760, 262)
    g.legend([(TOK, "Event"), (DEC, "Now event at the as-of date"), (HID_BAND, "Rescored today")])
    x_prev, x_now, x_end = 440, 550, 590
    # (client, outcome, rescored, event positions, gap from the last event to yesterday's run, dt yesterday, dt today)
    rows = [
        ("Client A", "new event", True, [180, 250, 360, 500], "1 day", "dt=74", "dt=63"),
        ("Client B", "bucket moved", True, [200, 270, 330], "3 days", "dt=83", "dt=85"),
        ("Client C", "unchanged", False, [170, 230], "45 days", "dt=104", "dt=104"),
    ]
    ys = [70 + k * 60 for k in range(len(rows))]
    for y, row in zip(ys, rows, strict=True):
        if row[2]:
            g.rect(14, y - 6, 732, 50, HID_BAND, r=8)
    for x, s in ((x_prev, "yesterday's run"), (x_now, "today's run")):
        g.text(x, 50, s, size=10.5, weight=600, anchor="middle")
        g.items.append(f'<line x1="{x}" y1="57" x2="{x}" y2="240" stroke="{MUTED}" stroke-width="1" stroke-dasharray="3 3"/>')
    for y, (name, outcome, rescored, events, gap, dt_prev, dt_now) in zip(ys, rows, strict=True):
        g.text(24, y + 16, name, size=11.5, weight=600)
        g.text(24, y + 32, outcome, size=10, fill=MUTED)
        g.items.append(f'<line x1="140" y1="{y + 19}" x2="{x_end}" y2="{y + 19}" stroke="{LINE}" stroke-width="1.4"/>')
        for x in events:
            g.rect(x - 7, y + 12, 14, 14, TOK, r=3)
        last = max(x for x in events if x < x_prev)
        g.text((last + x_prev - 30) / 2, y + 40, gap, size=9.5, fill=MUTED, anchor="middle")
        for x, s in ((x_prev, dt_prev), (x_now, dt_now)):
            w = len(s) * 6.0 + 12
            g.rect(x - w / 2, y + 9, w, 19, DEC, r=6)
            g.text(x, y + 22.2, s, size=10, anchor="middle", mono=True)
        verdict = "rescore" if rescored else "keep yesterday's score"
        g.text(x_end + 14, y + 23, verdict, size=10.5, weight=600 if rescored else 400, fill=INK if rescored else MUTED)
    g.save("delta")


# -------------------------------------------------------------------------------
# Plots, drawn with the same primitives as the diagrams
# -------------------------------------------------------------------------------
NAVY = ["#14283C", "#3E5A78", "#7189A3", "#A3B4C6"]
TEAL = ["#0B5F58", "#13827A", "#2FA597", "#5DBDB0", "#8ACFC5"]
BLUE = "#3B7DDD"


def labels() -> None:
    """Share of rows positive for each label by month: one log-scale plot, a legend beside it."""
    tr, te = pd.read_parquet(RUN / "train_rows.parquet"), pd.read_parquet(RUN / "test_rows.parquet")
    rows = pd.concat([tr, te.assign(weight=1.0)], ignore_index=True)
    rows["month"] = pd.to_datetime(rows["mon"]).dt.to_period("M").dt.to_timestamp()

    def rate(lab: str) -> np.ndarray:
        g = rows.assign(v=rows[lab] * rows["weight"]).groupby("month")
        return (100 * g["v"].sum() / g["weight"].sum()).to_numpy()

    months = pd.DatetimeIndex(sorted(rows["month"].unique()))
    series: list[tuple[str, str, np.ndarray, bool]] = []
    for k in range(4):
        series.append((NAVY[k], f"product {k + 1} (target_{k + 1})", rate(f"target_{k + 1}"), False))
    for k, (t, s_) in enumerate([(48, 31), (23, 49), (26, 28), (48, 10), (50, 19)]):
        y = rate(f"first_{t}_{s_}")
        y[-1] = np.nan  # no transactions after the last month
        series.append((TEAL[k], f"first transaction {t}/{s_}", y, True))
    ch = rate("churn")
    ch[-1] = np.nan
    series.append((BLUE, "churn: no transaction in the month", ch, False))

    g = Svg(760, 300)
    px0, px1, py0, py1 = 70, 500, 262, 16
    lo, hi = math.log10(0.004), math.log10(40)

    def ty(v: float) -> float:
        return py0 - (math.log10(v) - lo) / (hi - lo) * (py0 - py1)

    def tx(k: int) -> float:
        return px0 + 8 + k * (px1 - px0 - 16) / (len(months) - 1)

    for v, lab in [(0.01, "0.01%"), (0.1, "0.1%"), (1, "1%"), (10, "10%")]:
        g.items.append(f'<line x1="{px0}" y1="{ty(v):.1f}" x2="{px1}" y2="{ty(v):.1f}" stroke="#E9EDF1" stroke-width="1"/>')
        g.text(px0 - 8, ty(v) + 3.5, lab, size=10, fill=MUTED, anchor="end")
    g.items.append(f'<line x1="{px0}" y1="{py0}" x2="{px1}" y2="{py0}" stroke="{MUTED}" stroke-width="1"/>')
    for k, m in enumerate(months):
        g.items.append(f'<line x1="{tx(k):.1f}" y1="{py0}" x2="{tx(k):.1f}" y2="{py0 + 4}" stroke="{MUTED}" stroke-width="1"/>')
        if m.month % 2 == 0:
            g.text(tx(k), py0 + 18, m.strftime("%b %Y"), size=10, fill=MUTED, anchor="middle")
    g.items.append(
        f'<text transform="translate(18,{(py0 + py1) / 2}) rotate(-90)" font-family="{SANS}" font-size="10.5" '
        f'fill="{INK}" text-anchor="middle">share of rows positive (log scale)</text>'
    )
    for color, _, ys, dashed in series:
        pts = [(tx(k), ty(float(v))) for k, v in enumerate(ys) if not np.isnan(v)]
        d = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        dash = ' stroke-dasharray="4 2.5"' if dashed else ""
        width = 2.0 if color == BLUE else 1.5
        g.items.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{dash}/>')
        g.items.extend(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="1.8" fill="{color}"/>' for a, b in pts)
    ly = 62.0
    for color, label, _, dashed in series:
        dash = ' stroke-dasharray="4 2.5"' if dashed else ""
        g.items.append(f'<line x1="528" y1="{ly - 3.5}" x2="556" y2="{ly - 3.5}" stroke="{color}" stroke-width="1.8"{dash}/>')
        g.text(564, ly, label.replace("_", "_"), size=10, fill=INK)
        ly += 17
    g.save("labels")


if __name__ == "__main__":
    import sys

    names = sys.argv[1:] or ["timeline", "hierarchy", "decoder", "attention", "windows", "time_positions", "delta", "labels"]
    for name in names:
        globals()[name]()
