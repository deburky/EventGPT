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
GRAY = "#EEF1F4"
DARK = "#3B4856"
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
    """A symbol with a subscript: Unicode subscript digits, or a lowered smaller letter, which has no Unicode form in Inter."""
    if i.isdigit():
        return base + i.translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
    return f'{base}<tspan baseline-shift="sub" font-size="75%">{i}</tspan>'


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
PRODUCT = ["#DB6F44", "#835ECF", "#DC58B5", "#14283C"]  # one distinct hue per product target
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
        series.append((PRODUCT[k], f"product {k + 1} (target_{k + 1})", rate(f"target_{k + 1}"), False))
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


def futures() -> None:
    """One test client's real month beside three months sampled from EventGPT, and the scores read off the model."""
    import json

    d = json.loads((RUN.parent / "futures.json").read_text())
    accent = "#DB6F44"
    g = Svg(760, 282)
    g.legend([(TOK, "Real event"), (DEC, "Sampled by EventGPT"), (accent, f"First {d['target']}, the target")])
    gx = 560.0  # the geolocation key is a tick, not a swatch
    for dx in (0, 4, 8):
        g.items.append(f'<line x1="{gx + dx}" y1="{8}" x2="{gx + dx}" y2="{20}" stroke="{MUTED}" stroke-width="1" opacity="0.5"/>')
    g.text(gx + 16, 18, "Geolocation, one per day", size=11, fill=MUTED)
    h0, h1, m0, m1 = 132.0, 246.0, 268.0, 540.0  # history and label-month scales
    lo = min(-56.0, min(float(e["day"]) for e in d["history"]))

    def x_of(day: float) -> float:
        if day < 0:
            return h0 + (day - lo) / -lo * (h1 - h0)
        return m0 + day / d["month_days"] * (m1 - m0)

    rows = [("What happened", d["history"] + d["actual"], TOK)] + [
        (f"Sampled month {k + 1}", path, DEC) for k, path in enumerate(d["paths"])
    ]
    ys = [74 + 46 * k for k in range(len(rows))]
    top, bottom = ys[0] - 22, ys[-1] + 26
    g.rect(m0 - 6, top, m1 - m0 + 12, bottom - top, HID_BAND, r=8)
    g.text((m0 + m1) / 2, top - 6, "label month", size=10, fill=MUTED, anchor="middle")
    g.text((h0 + h1) / 2, top - 6, "history", size=10, fill=MUTED, anchor="middle")
    cut = (h1 + m0) / 2
    g.items.append(f'<line x1="{cut}" y1="{top - 2}" x2="{cut}" y2="{bottom + 4}" stroke="{INK}" stroke-width="1.2"/>')
    g.text(cut, bottom + 17, "as-of date", size=10, weight=600, anchor="middle")
    for day, lab in ((-50, "-50 d"), (-25, "-25 d")):
        g.text(x_of(day), bottom + 17, lab, size=9.5, fill=MUTED, anchor="middle")
    for day in (7, 14, 21, 28):
        g.text(x_of(day), bottom + 17, f"+{day} d", size=9.5, fill=MUTED, anchor="middle")
    for y, (name, evs, fill) in zip(ys, rows, strict=True):
        g.text(16, y + 4, name, size=11, weight=600 if fill == TOK else 400)
        g.items.append(f'<line x1="{h0 if fill == TOK else m0}" y1="{y}" x2="{m1}" y2="{y}" stroke="{LINE}" stroke-width="1.2"/>')
        if fill != TOK:
            g.text((h0 + h1) / 2, y + 4, "same history", size=9.5, fill=MUTED, anchor="middle")
        last_label = -1e9
        for e in sorted(evs, key=lambda e: float(e["day"])):
            x, kind = x_of(float(e["day"])), str(e["kind"])
            if kind == "geo":
                tick = f'x1="{x:.1f}" y1="{y - 6}" x2="{x:.1f}" y2="{y + 6}"'
                g.items.append(f'<line {tick} stroke="{MUTED}" stroke-width="1" opacity="0.5"/>')
                continue
            hit = kind == d["target"]
            g.rect(x - 5, y - 5, 10, 10, accent if hit else fill, r=2.5)
            if hit or x - last_label >= 34:
                g.text(x, y - 10, kind, size=8.5, fill=accent if hit else MUTED, anchor="middle", mono=True)
                last_label = x
    # the scores read off the model for this client
    px = 568
    g.text(px, 52, "Next event type, one step", size=11, weight=600)
    bars = [(c["name"].replace("trx_", ""), c["p"], False) for c in d["candidates"]]
    bars.append((f"{d['next_kind'].split('/')[0]}, actual (rank {d['next_rank']})", d["next_p"], True))
    for k, (name, p, actual) in enumerate(bars):
        y = 68 + 17 * k + (6 if actual else 0)
        g.text(px, y + 8, name, size=9.5, fill=accent if actual else INK, mono=not actual and name != "geo")
        g.rect(px + 112, y, max(2.0, p * 180), 10, accent if actual else DEC, r=2)
        g.text(px + 116 + max(2.0, p * 180), y + 8.5, f"{p:.1%}", size=9, fill=MUTED)
    g.text(px, 196, f"First {d['target']} in the month", size=11, weight=600)
    for k, (lab, v) in enumerate([("rollout, 64 months", d["rollout"]), ("one step", d["single_step"]), ("base rate", d["prevalence"])]):
        g.text(px, 214 + 16 * k, lab, size=10, fill=MUTED)
        g.text(px + 170, 214 + 16 * k, f"{v:.1%}", size=10, weight=600 if k == 0 else 400, anchor="end")
    g.text(px, 270, f"top {d['share_above']:.1%} of test rows", size=10, fill=INK)
    g.save("futures")


def futures_compare(cases: tuple[int, ...] = (3, 1)) -> None:
    """Random active test clients: a month predicted by EventGPT above the month that followed, colored by event type."""
    import json

    many = json.loads((RUN.parent / "futures_many.json").read_text())
    palette = ["#DB6F44", "#835ECF", "#53AD7D", "#DB9643", "#DC58B5", "#4FAC9B", "#9A8DEB"]
    m0, m1, block = 150.0, 740.0, 128.0
    rows = []
    counts: dict[str, int] = {}
    for case in cases:
        d = many[case]
        lengths = sorted(len(path) for path in d["paths"])
        predicted = next(path for path in d["paths"] if len(path) == lengths[len(lengths) // 2])
        rows.append((d, predicted))
        for e in d["actual"] + predicted:
            if e["kind"] != "geo":
                counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    top = sorted(counts, key=lambda k: (-counts[k], k))[: len(palette)]
    color = {k: palette[j] for j, k in enumerate(top)}
    g = Svg(880, int(44 + block * len(cases)))
    lx = 16.0
    for k in top:
        g.items.append(f'<circle cx="{lx + 5}" cy="14" r="5" fill="{color[k]}"/>')
        g.text(lx + 14, 18, k, size=10, fill=INK, mono=True)
        lx += 70
    g.items.append(f'<circle cx="{lx + 5}" cy="14" r="5" fill="{LINE}"/>')
    g.text(lx + 14, 18, "other", size=10, fill=MUTED)
    lx += 62
    g.items.append(f'<line x1="{lx + 2}" y1="8" x2="{lx + 2}" y2="20" stroke="{MUTED}" stroke-width="1" opacity="0.6"/>')
    g.text(lx + 10, 18, "geolocation", size=10, fill=MUTED)
    for c, (d, predicted) in enumerate(rows):
        y0 = 44 + block * c

        def x_of(day: float, d: dict = d) -> float:
            return m0 + day / d["month_days"] * (m1 - m0)

        g.text(16, y0 + 8, f"Client {c + 1}", size=11, weight=600)
        # events matched by type, ignoring timing: the overlap of the two type counts, transactions and geolocation apart
        pred_n: dict[str, int] = {}
        real_n: dict[str, int] = {}
        for evs, n in ((predicted, pred_n), (d["actual"], real_n)):
            for e in evs:
                n[e["kind"]] = n.get(e["kind"], 0) + 1
        trx = [k for k in real_n if k != "geo"]
        hit = sum(min(pred_n.get(k, 0), real_n[k]) for k in trx)
        lines = [f"{hit} of {sum(real_n[k] for k in trx)} transactions"]
        if "geo" in real_n:
            lines.append(f"{min(pred_n.get('geo', 0), real_n['geo'])} of {real_n['geo']} geolocation")
        lines.append(f"{sum(k in pred_n for k in trx)} of {len(trx)} types seen")
        g.text(764, y0 + 34, "Matched by type", size=10, weight=600)
        for j, line in enumerate(lines):
            g.text(764, y0 + 52 + 16 * j, line, size=10, fill=MUTED if j == len(lines) - 1 else INK)
        for r, (name, evs) in enumerate([("Predicted", predicted), ("What followed", d["actual"])]):
            y = y0 + 30 + 36 * r
            g.text(16, y + 4, name, size=11)
            g.text(126, y + 4, str(len(evs)), size=9.5, fill=MUTED, anchor="end")
            g.items.append(f'<line x1="{m0}" y1="{y}" x2="{m1}" y2="{y}" stroke="{LINE}" stroke-width="1.2"/>')
            for e in evs:
                x = x_of(float(e["day"]))
                if e["kind"] == "geo":
                    tick = f'x1="{x:.1f}" y1="{y - 7}" x2="{x:.1f}" y2="{y + 7}"'
                    g.items.append(f'<line {tick} stroke="{MUTED}" stroke-width="1" opacity="0.6"/>')
            for e in evs:
                if e["kind"] != "geo":
                    dot = f'cx="{x_of(float(e["day"])):.1f}" cy="{y}" r="5" fill="{color.get(e["kind"], LINE)}"'
                    g.items.append(f'<circle {dot} stroke="white" stroke-width="1.5"/>')
        yb = y0 + 98
        g.items.append(f'<line x1="{m0}" y1="{y0 + 18}" x2="{m0}" y2="{yb - 12}" stroke="{INK}" stroke-width="1.2"/>')
        g.text(m0, yb + 2, "as-of date", size=9.5, weight=600, anchor="middle")
        for day in (7, 14, 21, 28):
            g.text(x_of(day), yb + 2, f"+{day} d", size=9.5, fill=MUTED, anchor="middle")
    g.save("futures_compare")


def blocks() -> None:
    """Inside the three stages: pre-norm transformer blocks, what each attends over, and what goes in and out."""
    g = Svg(760, 512)
    g.legend([(TOK, "Tokens"), (ENC, "Event vectors"), (HID, "Hidden states"), (DEC, "Next-event decoder")])
    width, gap = 220.0, 30.0
    stages = [
        (
            "Event encoder",
            "2 layers, within each event",
            ENC_BAND,
            ENC,
            "2",
            ("Self-attention", "bidirectional, within the event"),
            ("LayerNorm, mean pool", ""),
            sub("e", "i"),
            ("Token embeddings", "of one event, no positions"),
            TOK,
        ),
        (
            "History decoder",
            "6 layers, over events",
            HID_BAND,
            HID,
            "6",
            ("Causal self-attention", "temporal RoPE on queries and keys"),
            ("LayerNorm", ""),
            sub("h", "i"),
            (sub("e", "1") + " to " + sub("e", "i"), "one vector per event"),
            ENC,
        ),
        (
            "Token decoder",
            "2 layers, within the next event",
            DEC_BAND,
            DEC,
            "2",
            ("Causal self-attention", "within the event"),
            ("LayerNorm, tied linear head", ""),
            "softmax over 12,828 tokens",
            ("Token embeddings + " + sub("h", "i"), "and the position in the event"),
            TOK,
        ),
    ]
    for c, (title, subtitle, band, fill, n, attn, top, out, inp, inp_fill) in enumerate(stages):
        x0 = 20 + c * (width + gap)
        cx = x0 + width / 2
        g.rect(x0, 40, width, 430, band, r=12)
        g.text(cx, 62, title, size=12.5, weight=600, anchor="middle")
        g.text(cx, 78, subtitle, size=10.5, fill=MUTED, anchor="middle")
        # output, final norm
        g.rect(cx - 80, 92, 160, 28, fill if c < 2 else DEC, r=8)
        g.text(cx, 110.5, out, size=12 if c < 2 else 10.5, weight=600 if c < 2 else 500, anchor="middle")
        g.arrow([(cx, 136), (cx, 122)])
        g.rect(cx - 80, 136, 160, 24, "white", LINE, r=6)
        g.text(cx, 152, top[0], size=10, anchor="middle")
        # the repeated block
        g.rect(x0 + 8, 174, width - 16, 212, "white", LINE, r=10, dash="4 3")
        g.text(x0 + width - 16, 192, f"× {n}", size=12, weight=600, fill=MUTED, anchor="end")
        boxes = [(208, 24, "MLP, GELU", ""), (244, 20, "LayerNorm", ""), (302, 36, attn[0], attn[1]), (350, 22, "LayerNorm", "")]
        for y, h, a, b in boxes:
            g.rect(cx - 75, y, 150, h, "white", INK if a != "LayerNorm" else LINE, r=6)
            if b:
                g.text(cx, y + 15, a, size=10, weight=600, anchor="middle")
                g.text(cx, y + 29, b, size=8.8, fill=MUTED, anchor="middle")
            else:
                g.text(cx, y + h / 2 + 4, a, size=10, weight=600 if a != "LayerNorm" else 400, anchor="middle")
        for cy in (194, 288):
            g.items.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="white" stroke="{INK}" stroke-width="1.2"/>')
            g.items.append(f'<line x1="{cx - 4}" y1="{cy}" x2="{cx + 4}" y2="{cy}" stroke="{INK}" stroke-width="1.2"/>')
            g.items.append(f'<line x1="{cx}" y1="{cy - 4}" x2="{cx}" y2="{cy + 4}" stroke="{INK}" stroke-width="1.2"/>')
        # main path, bottom to top
        g.arrow([(cx, 350), (cx, 340)])
        g.arrow([(cx, 302), (cx, 297)])
        g.arrow([(cx, 281), (cx, 266)])
        g.arrow([(cx, 244), (cx, 234)])
        g.arrow([(cx, 208), (cx, 203)])
        g.arrow([(cx, 187), (cx, 162)])
        # residual paths around attention and around the MLP
        rx = cx - 92
        g.arrow([(cx, 380), (rx, 380), (rx, 288), (cx - 9, 288)])
        g.arrow([(cx, 274), (rx + 6, 274), (rx + 6, 194), (cx - 9, 194)])
        # input
        g.arrow([(cx, 404), (cx, 374)])
        g.rect(cx - 90, 404, 180, 36, inp_fill, r=8)
        g.text(cx, 419, inp[0], size=10.5, weight=600, anchor="middle")
        g.text(cx, 433, inp[1], size=9, fill=MUTED, anchor="middle")
    for c in range(2):
        x0 = 20 + c * (width + gap)
        gx = x0 + width + gap / 2
        g.arrow([(x0 + width / 2 + 80, 106), (gx, 106), (gx, 422), (gx + gap / 2 + 18, 422)])
    g.text(
        20,
        494,
        "Every block is pre-norm with residual adds; attention has 12 heads of width 32 with query-key RMSNorm, "
        "and the MLP widens 384 to 1,536.",
        size=10,
        fill=MUTED,
    )
    g.save("blocks")


def inference() -> None:
    """Past, as-of date, futures: the embedding sits at the as-of date and the future branches from it."""
    import json

    g = Svg(760, 268)
    axis, now, end = 130.0, 400.0, 712.0
    violet, pale = "#9A8DEB", "#C9C2F5"
    # the past: real events on one line
    g.items.append(f'<line x1="30" y1="{axis}" x2="{now}" y2="{axis}" stroke="{LINE}" stroke-width="2"/>')
    past = [(52, "profile"), (98, "1/12"), (131, "geo"), (158, "geo"), (214, "40/52"), (251, "1/12"), (300, "11/32"), (352, "1/12")]
    for k, (x, t) in enumerate(past):
        g.items.append(f'<circle cx="{x}" cy="{axis}" r="6" fill="{TOK}" stroke="{INK}" stroke-width="0.8"/>')
        g.text(x, axis - 14 if k % 2 == 0 else axis + 24, t, size=9.5, fill=MUTED, anchor="middle", mono=True)
    # the as-of date and the state there
    g.items.append(f'<line x1="{now}" y1="34" x2="{now}" y2="228" stroke="{INK}" stroke-width="1" stroke-dasharray="3 3"/>')
    g.text(now, 26, "as-of date", size=10.5, weight=600, anchor="middle")
    # the next event: one branch per type, as wide as its probability for the client of Figure~futures, all the way
    # to the horizon; the dots are the later events of a future sampled down that branch
    d = json.loads((RUN.parent / "futures.json").read_text())
    p_of = {c["name"].removeprefix("trx_"): c["p"] for c in d["candidates"]}
    branches = [(58.0, "40/52", p_of["40"]), (98.0, "1/12", p_of["1"]), (138.0, "geo", p_of["geo"]), (212.0, "50/19", d["next_p"])]
    for y, t, p in branches:
        w = max(1.0, p * 32)
        g.items.append(
            f'<path d="M{now},{axis} C{now + 40},{axis} {now + 40},{y} {now + 84},{y}" fill="none" stroke="{violet}" '
            f'stroke-width="{w:.1f}" stroke-linecap="round" opacity="0.85"/>'
        )
        pw = 48.0  # one width for every type, so a box's size says nothing about its probability
        g.rect(now + 88, y - 9.5, pw, 19, DEC, r=6)
        g.text(now + 88 + pw / 2, y + 3.7, t, size=10, anchor="middle", mono=True)
        x0 = now + 92 + pw
        g.items.append(f'<line x1="{x0}" y1="{y}" x2="{end}" y2="{y}" stroke="{pale}" stroke-width="{w:.1f}"/>')
        g.text(now + 88 + pw / 2, y - 14, f"{p:.0%}" if p >= 0.05 else f"{p:.1%}", size=10, weight=600, anchor="middle")
        for dx in (0.3, 0.55, 0.85) if t != "50/19" else (0.6,):
            g.items.append(f'<circle cx="{x0 + dx * (end - x0):.1f}" cy="{y}" r="4" fill="{DEC}" stroke="{violet}" stroke-width="0.8"/>')
    # the other types: thin unlabeled branches, between geo and the target in rank
    for y in (160.0, 168.0, 176.0, 184.0):
        g.items.append(
            f'<path d="M{now},{axis} C{now + 40},{axis} {now + 40},{y} {now + 84},{y}" fill="none" stroke="{violet}" '
            f'stroke-width="0.8" opacity="0.6"/>'
        )
    g.text(now + 92, 176, f"{d['names'] - len(branches)} other event types", size=9.5, fill=MUTED)
    g.items.append(f'<circle cx="{now}" cy="{axis}" r="10" fill="{EMB}"/>')
    g.text(now, axis + 4, "h", size=11, weight=600, fill="white", anchor="middle")
    g.items.append(f'<line x1="{end}" y1="34" x2="{end}" y2="228" stroke="{INK}" stroke-width="1" stroke-dasharray="3 3"/>')
    g.text(end, 26, "horizon", size=10.5, weight=600, anchor="middle")
    # what each part is read as
    for x, head, note in [
        (200.0, "Event history", "real events"),
        (now, "Embedding", "h at the as-of date"),
        (now + 118, "Next event", "width shows probability"),
        (end - 66, "Sampled futures", "up to the horizon"),
    ]:
        g.text(x, 244, head, size=11, weight=600, anchor="middle")
        g.text(x, 259, note, size=9.5, fill=MUTED, anchor="middle")
    g.save("inference")


def box(
    g: Svg, cx: float, y: float, label: str, fill: str = "white", stroke: str = INK, w: float = 210, h: float = 26, note: str = ""
) -> None:
    """A labeled box centered at cx, with an optional second line."""
    g.rect(cx - w / 2, y, w, h, fill, stroke, r=6)
    ink = "white" if fill == DARK else INK
    if note:
        g.text(cx, y + h / 2 - 2, label, size=10.5, weight=600, fill=ink, anchor="middle")
        g.text(cx, y + h / 2 + 11, note, size=9, fill="#D5DEE8" if fill == DARK else MUTED, anchor="middle")
    else:
        g.text(cx, y + h / 2 + 4, label, size=10.5, weight=500, fill=ink, anchor="middle")


def plus(g: Svg, cx: float, cy: float) -> None:
    """A residual add."""
    g.items.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="white" stroke="{INK}" stroke-width="1.2"/>')
    g.items.append(f'<line x1="{cx - 4.5}" y1="{cy}" x2="{cx + 4.5}" y2="{cy}" stroke="{INK}" stroke-width="1.2"/>')
    g.items.append(f'<line x1="{cx}" y1="{cy - 4.5}" x2="{cx}" y2="{cy + 4.5}" stroke="{INK}" stroke-width="1.2"/>')


def curve(g: Svg, x0: float, y0: float, x1: float, y1: float) -> None:
    """A thin curved pointer from a callout to a part of the model."""
    g.items.append(
        f'<path d="M{x0},{y0} C{x0 - 60},{y0} {x1 + 70},{y1} {x1},{y1}" fill="none" stroke="{MUTED}" '
        'stroke-width="1" marker-end="url(#head)"/>'
    )


def fan(
    g: Svg,
    x0: float,
    x1: float,
    y_top: float,
    y_bot: float,
    n_top: int,
    n_bot: int,
    top_ends: tuple[str, str],
    bot_ends: tuple[str, str],
    fill: str,
) -> None:
    """Fully connected units: every bottom unit to every top unit."""

    def xs(n: int, a: float, b: float) -> list[float]:
        return [a + (b - a) * k / max(n - 1, 1) for k in range(n)] if n > 1 else [(a + b) / 2]

    tops = xs(n_top, x0, x1)
    pad = (x1 - x0) * 0.18 if n_bot < n_top else 0
    bots = xs(n_bot, x0 + pad, x1 - pad)
    for xb in bots:
        for xt in tops:
            g.items.append(
                f'<line x1="{xb:.1f}" y1="{y_bot}" x2="{xt:.1f}" y2="{y_top}" stroke="{INK}" stroke-width="0.45" opacity="0.55"/>'
            )
    for row, y in ((tops, y_top), (bots, y_bot)):
        for x in row:
            c = fill if row is tops else HID
            g.items.append(f'<circle cx="{x:.1f}" cy="{y}" r="6" fill="{c}" stroke="{INK}" stroke-width="0.9"/>')
    if top_ends[0]:
        g.text(tops[0], y_top - 11, top_ends[0], size=10, anchor="middle")
    if top_ends[1]:
        g.text(tops[-1], y_top - 11, top_ends[1], size=10, anchor="middle")
    g.text(bots[0], y_bot + 21, bot_ends[0], size=10, anchor="middle")
    g.text(bots[-1], y_bot + 21, bot_ends[1], size=10, anchor="middle")


def lines(g: Svg, x: float, y: float, rows: list[str]) -> None:
    """A callout caption, one line per row."""
    for k, s in enumerate(rows):
        g.text(x, y + k * 15, s, size=10.5, weight=600)


def explainer() -> None:
    """The model as one stack, bottom to top, with panels on the output layer, the embedding and the event pooling."""
    g = Svg(900, 762)
    cx = 235.0
    # outputs and the model container
    g.text(cx, 22, "Next-event tokens", size=11, anchor="middle", mono=True)
    g.arrow([(cx, 86), (cx, 30)])
    g.rect(40, 46, 390, 676, GRAY, r=16)
    g.text(52, 710, "EventGPT", size=13, weight=700)

    # token decoder
    g.rect(110, 60, 300, 148, DEC_BAND, r=12)
    g.text(400, 76, "Token decoder", size=10.5, weight=600, fill=MUTED, anchor="end")
    box(g, cx, 86, "Linear output layer", h=22)
    box(g, cx, 116, "Final LayerNorm", h=22)
    box(g, cx, 146, "Causal self-attention block", DEC, "none", note="within the next event", h=30)
    g.text(cx + 112, 166, "× 2", size=11, weight=600, fill=MUTED)
    box(g, cx, 184, "Token embeddings + " + sub("h", "i"), w=170, h=20)
    for y0, y1 in ((116, 108), (146, 138), (184, 176)):
        g.arrow([(cx, y0), (cx, y1)])

    # the hidden state, the embedding
    g.items.append(f'<circle cx="{cx}" cy="226" r="12" fill="{EMB}"/>')
    g.text(cx, 230.5, sub("h", "i"), size=12, weight=600, fill="white", anchor="middle")
    g.arrow([(cx, 214), (cx, 204)])
    box(g, cx, 250, "Final LayerNorm", h=22)
    g.arrow([(cx, 250), (cx, 238)])

    # history decoder: the repeated block
    top, bot = 284, 500
    g.rect(110, top, 250, bot - top, HID_BAND, HID, r=14)
    g.text(122, top + 17, "History decoder", size=10.5, weight=600, fill=MUTED)
    mid = (top + bot) / 2
    g.text(84, mid + 6, "6 ×", size=17, weight=500, anchor="end")
    g.items.append(
        f'<path d="M104,{top + 8} Q96,{top + 8} 96,{top + 30} L96,{mid - 10} Q96,{mid} 90,{mid} '
        f'Q96,{mid} 96,{mid + 10} L96,{bot - 30} Q96,{bot - 8} 104,{bot - 8}" fill="none" stroke="{INK}" stroke-width="1.4"/>'
    )
    plus(g, cx, 304)
    box(g, cx, 320, "Feed forward", "#E4E9EF", note="GELU, 384 to 1,536", h=32)
    box(g, cx, 362, "LayerNorm 2", h=20)
    plus(g, cx, 398)
    box(g, cx, 414, "Causal multi-head attention", DARK, "none", note="over events, temporal RoPE", h=34)
    box(g, cx, 458, "LayerNorm 1", h=20)
    g.arrow([(cx, 296), (cx, 274)])
    g.arrow([(cx, 320), (cx, 314)])
    g.arrow([(cx, 362), (cx, 354)])
    g.arrow([(cx, 390), (cx, 384)])
    g.arrow([(cx, 414), (cx, 408)])
    g.arrow([(cx, 458), (cx, 450)])
    g.arrow([(cx, 518), (cx, 480)])
    # residuals
    rx = cx + 118
    g.arrow([(cx, 490), (rx, 490), (rx, 398), (cx + 10, 398)])
    g.arrow([(cx, 388), (rx - 8, 388), (rx - 8, 304), (cx + 10, 304)])

    # event vectors and the event encoder
    box(g, cx, 518, sub("e", "1") + " to " + sub("e", "i") + ", one vector per event", ENC, "none", w=230, h=22)
    g.rect(110, 552, 300, 98, ENC_BAND, r=12)
    g.text(400, 568, "Event encoder", size=10.5, weight=600, fill=MUTED, anchor="end")
    box(g, cx, 576, "Mean pool", h=22)
    box(g, cx, 610, "Self-attention block", ENC, "none", note="bidirectional, within each event", h=32)
    g.text(cx + 112, 631, "× 2", size=11, weight=600, fill=MUTED)
    g.arrow([(cx, 576), (cx, 542)])
    g.arrow([(cx, 610), (cx, 600)])
    box(g, cx, 662, "Token embedding layer", h=22)
    g.arrow([(cx, 662), (cx, 644)])
    box(g, cx, 692, "Tokenized events", TOK, "none", h=22)
    g.arrow([(cx, 692), (cx, 686)])
    g.arrow([(cx, 734), (cx, 716)])
    g.text(cx, 752, "A client's events", size=11, anchor="middle", mono=True)

    ax, aw, ah = 470.0, 410.0, 205.0
    # callout A: the output layer
    ay = 40.0
    g.rect(ax, ay, aw, ah, "white", INK, r=18)
    fan(g, ax + 30, ax + aw - 30, ay + 36, ay + 100, 16, 9, ("1", "12,828"), ("1", "384"), TOK)
    lines(
        g,
        ax + 24,
        ay + 148,
        [
            "The linear output layer maps 384 hidden units to",
            "12,828 units, one per token in the vocabulary.",
            "Zero-shot scores read the target's tokens here.",
        ],
    )
    curve(g, ax, ay + 90, cx + 105, 97)

    # callout B: the embedding read by a head
    by = 265.0
    g.rect(ax, by, aw, ah, "white", INK, r=18)
    fan(g, ax + 60, ax + aw - 60, by + 36, by + 100, 1, 9, ("label", ""), ("1", "384"), "#F6D7C3")
    lines(
        g,
        ax + 24,
        by + 148,
        [
            "For a supervised label, a logistic regression reads",
            sub("h", "i") + " at the as-of date, frozen: 384 hidden units",
            "to one probability.",
        ],
    )
    g.items.append(
        f'<path d="M{ax},{by + 15} C{ax - 40},226 {cx + 65},226 {cx + 14},226" fill="none" stroke="{MUTED}" '
        'stroke-width="1" marker-end="url(#head)"/>'
    )

    # callout C: one vector per event
    cy0 = 490.0
    g.rect(ax, cy0, aw, ah, "white", INK, r=18)
    toks = ["[EVT]", "name=trx_1", "dt=79", "sub=12", "amt_bin=19"]
    total = sum(len(t) * 6.0 + 12 for t in toks) + 5 * (len(toks) - 1)
    x, centers = ax + (aw - total) / 2, []
    for t in toks:
        w = g.pill(x, cy0 + 100, t)
        centers.append(x + w / 2)
        x += w + 5
    mid = ax + aw / 2
    for xc in centers:
        g.items.append(f'<line x1="{xc}" y1="{cy0 + 98}" x2="{mid}" y2="{cy0 + 60}" stroke="{INK}" stroke-width="0.6" opacity="0.6"/>')
    g.rect(mid - 70, cy0 + 34, 140, 26, ENC, r=8)
    g.text(mid, cy0 + 51, sub("e", "i") + ", 384 units", size=10.5, weight=600, anchor="middle")
    g.text(mid, cy0 + 24, "mean pool", size=9.5, fill=MUTED, anchor="middle")
    lines(
        g,
        ax + 24,
        cy0 + 148,
        [
            "The event encoder pools the tokens of each event",
            "into one vector, so the history decoder has one",
            "position per event and attends over events.",
        ],
    )
    curve(g, ax, cy0 + 95, cx + 105, 587)

    g.save("explainer")


if __name__ == "__main__":
    import sys

    names = sys.argv[1:] or ["timeline", "hierarchy", "decoder", "attention", "windows", "time_positions", "delta", "labels", "futures"]
    for name in names:
        globals()[name]()
