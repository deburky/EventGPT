"""EventGPT-D diagrams in the style of the EventGPT paper, on Home Credit test case 244050.

The primitives and colors come from paper/figures/src/figures.py. The case is row 563 of the test fold.

EventGPT-D predicts the next event and states its confidence q as its own probability of that prediction. It keeps EventGPT's
output layer W0 and adds the residual head z = W0 y + A std(y) + b of Geuter, Cheng and Alvarez-Melis (2026,
arXiv:2610.17205), trains by cross-entropy on the true next event, and solves for the intercept b by Fisher scoring.
The case's numbers are in case563.json (from eventgpt.homecredit.rlcr, seed 0); the reliability diagrams read the
saved per-case confidences of data/runs/hc-rlcr-s42/v4.

Run from the repository root: `uv run python paper/eventgpt-d/figures/src/figures_d.py` (needs rsvg-convert, Inter
and JetBrains Mono).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
OUT = SRC.parent
_spec = importlib.util.spec_from_file_location("figures", SRC.parents[2] / "figures" / "src" / "figures.py")
assert _spec is not None and _spec.loader is not None
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
INK, MUTED, LINE, TOK = F.INK, F.MUTED, F.LINE, F.TOK
HID, HID_BAND, DEC, DEC_BAND, EMB = F.HID, F.HID_BAND, F.DEC, F.DEC_BAND, F.EMB
RES, RES_BAND = "#FBDDBF", "#FEF3E8"
ENC_FILL = F.ENC  # the residual head and its training, new in EventGPT-D
VIOLET, WARM = "#9A8DEB", "#E58A3C"
RUN = Path("data/runs/hc-rlcr-s42/v4")
D = json.loads((SRC / "case563.json").read_text())


class Svg(F.Svg):
    """The paper's SVG builder, saving into this folder."""

    def save(self, name: str) -> None:
        """Write name.svg to src/ and name.pdf and name.png beside it."""
        head = (
            '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,1 L9,5 L0,9 z" fill="{INK}"/></marker></defs>'
        )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
            f'{head}<rect width="100%" height="100%" fill="white"/>\n' + "\n".join(self.items) + "\n</svg>"
        )
        (SRC / f"{name}.svg").write_text(svg)
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(OUT / f"{name}.pdf"), str(SRC / f"{name}.svg")], check=True)
        subprocess.run(["rsvg-convert", "-z", "3", "-o", str(OUT / f"{name}.png"), str(SRC / f"{name}.svg")], check=True)


def pill_w(s: str) -> float:
    """Width of a token pill, as Svg.pill draws it."""
    return len(s) * 6.0 + 12


TEX = SRC / "tex"
Z = r"z = W_0\,y + A\,\mathrm{std}(y) + b"


def tex(g: Svg, x: float, y: float, expr: str, size: float = 11.0, anchor: str = "start", color: str = INK) -> float:
    """A LaTeX formula typeset by latex and dvisvgm as paths, centered on the x-height of text with baseline y.

    Returns its width. Each formula is cached in tex/ under a hash of its source.
    """
    key = "t" + hashlib.md5(f"{expr}|{color}".encode()).hexdigest()[:10]
    path = TEX / f"{key}.svg"
    if not path.exists():
        TEX.mkdir(exist_ok=True)
        doc = (
            r"\documentclass[preview,border=0pt]{standalone}\usepackage{amsmath,amssymb,xcolor}\begin{document}"
            + r"\definecolor{ink}{HTML}{"
            + color.lstrip("#")
            + r"}\color{ink}$"
            + expr
            + r"$\end{document}"
        )
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "f.tex").write_text(doc)
            subprocess.run(["latex", "-interaction=nonstopmode", "f.tex"], cwd=d, check=True, capture_output=True)
            subprocess.run(["dvisvgm", "--no-fonts", "--exact-bbox", "-o", str(path), "f.dvi"], cwd=d, check=True, capture_output=True)
    src = path.read_text()
    root = re.search(r"<svg[^>]*>", src)
    assert root is not None
    attrs = dict(re.findall(r"(\w+)='([^']*)'", root.group(0)))
    w_pt, h_pt, vb = float(attrs["width"].rstrip("pt")), float(attrs["height"].rstrip("pt")), attrs["viewBox"]
    inner = src[root.end() : src.rindex("</svg>")]
    inner = re.sub(r"id='([^']+)'", lambda m: f"id='{key}{m.group(1)}'", inner)
    inner = re.sub(r"href='#([^']+)'", lambda m: f"href='#{key}{m.group(1)}'", inner)
    k = size / 10.0
    w, h = w_pt * k, h_pt * k
    x0 = {"start": x, "middle": x - w / 2, "end": x - w}[anchor]
    y0 = y - 0.35 * size - h / 2
    g.items.append(
        f'<svg xmlns:xlink="http://www.w3.org/1999/xlink" x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'viewBox="{vb}" overflow="visible">{inner}</svg>'
    )
    return w


# -------------------------------------------------------------------------------
# 2. Confidence as a product of the decoder's probabilities
# -------------------------------------------------------------------------------
def confidence() -> None:
    """The prediction as a path through the decoder's choices; branch width is probability, q is the product along the path."""
    g = Svg(800, 300)
    g.legend([(HID, "History state"), (DEC, "Chosen token"), (TOK, "Alternatives"), (EMB, "Confidence")])
    steps = D["eventgpt_d"]["steps"]
    cols = [175.0, 325.0, 475.0, 625.0]
    y_main = 150.0
    x_prev, y_prev = 40.0, y_main
    g.items.append(f'<circle cx="{x_prev}" cy="{y_main}" r="11" fill="{HID}" stroke="{INK}" stroke-width="0.9"/>')
    g.text(x_prev, y_main + 4, "h", size=11, weight=600, anchor="middle")
    heads = ["kind", "n", "late", "pdpd"]
    for k, (x, st) in enumerate(zip(cols, steps, strict=True)):
        alts = st["top"][:3]
        ys = [y_main, y_main - 62, y_main + 62]
        for (tok, p), yy in zip(alts, ys, strict=True):
            chosen = tok == st["chosen"]
            w = max(0.8, p * 14)
            g.items.append(
                f'<path d="M{x_prev + 12},{y_prev} C{x_prev + 50},{y_prev} {x - 70},{yy} {x - 42},{yy}" fill="none" '
                f'stroke="{VIOLET}" stroke-width="{w:.1f}" stroke-linecap="round" opacity="{0.9 if chosen else 0.45}"/>'
            )
            pw = pill_w(tok)
            g.rect(x - 40, yy - 9.5, pw, 19, DEC if chosen else TOK, r=6)
            g.text(x - 40 + pw / 2, yy + 3.7, tok, size=10, anchor="middle", mono=True)
            g.text(
                x - 40 + pw / 2,
                yy - 14,
                f"{p:.2f}" if p >= 0.005 else "&lt;0.01",
                size=10,
                weight=600 if chosen else 400,
                fill=INK if chosen else MUTED,
                anchor="middle",
            )
        g.text(x - 40 + pill_w(st["chosen"]) / 2, 268, heads[k], size=11, weight=600, anchor="middle")
        x_prev, y_prev = x - 40 + pill_w(st["chosen"]), y_main
    qt = D["q_terms_d"]
    g.arrow([(x_prev + 6, y_main), (x_prev + 30, y_main)])
    g.rect(x_prev + 32, y_main - 22, 88, 44, EMB, r=10)
    g.text(x_prev + 76, y_main - 2, f"q = {qt['q']:.2f}", size=12, weight=600, fill="white", anchor="middle")
    g.text(x_prev + 76, y_main + 13, "product, n summed", size=8.5, fill="#D5DEE8", anchor="middle")
    w = tex(g, 150, 292, r"q = P(\text{kind}) \textstyle\sum_n P(n)\,P(\text{late} \mid n)\,P(\text{pdpd} \mid n, \text{late})", size=10.5)
    g.text(156 + w, 292, f"= {qt['p_kind']:.3f} \u00d7 {qt['sum_n']:.3f}; width shows probability", size=10, fill=MUTED)
    g.save("confidence")


# -------------------------------------------------------------------------------
# 3. The Fisher step on the intercept
# -------------------------------------------------------------------------------
def fisher() -> None:
    """Largest score per step, and each token's predicted count against its observed count before and after."""
    g = Svg(760, 312)
    g.legend([(RES, "Before the Fisher step"), (HID, "After")])
    # left: convergence
    log = D["fisher_log"][:5]
    x0, y0, h = 60.0, 250.0, 170.0
    g.text(40, 52, "Largest absolute score over all tokens", size=11.5, weight=600)
    for k, v in enumerate(log):
        bh = max(1.0, h * v / log[0])
        x = x0 + k * 56
        g.rect(x, y0 - bh, 36, bh, RES if k == 0 else HID, r=4)
        g.text(x + 18, y0 - bh - 6, f"{v:.1f}", size=10, weight=600, anchor="middle")
        g.text(x + 18, y0 + 16, "before" if k == 0 else f"step {k}", size=9.5, fill=MUTED, anchor="middle")
    g.items.append(f'<line x1="{x0 - 8}" y1="{y0}" x2="{x0 + 5 * 56}" y2="{y0}" stroke="{LINE}" stroke-width="1"/>')
    # right: relative error of the predicted count
    ts = D["token_sums"]
    toks = list(ts["after"])
    cx, top, row = 560.0, 82.0, 25.0
    scale = 520.0  # pixels per unit of relative error
    g.text(400, 52, "Predicted minus observed count, as a share of observed", size=11.5, weight=600)
    g.items.append(f'<line x1="{cx}" y1="{top - 10}" x2="{cx}" y2="{top + row * len(toks)}" stroke="{INK}" stroke-width="1"/>')
    for k, t in enumerate(toks):
        obs, before, after = ts["after"][t][0], ts["before"][t][1], ts["after"][t][1]
        rb, ra = (before - obs) / obs, (after - obs) / obs
        y = top + k * row
        g.text(cx - 140, y + 4, t, size=10, mono=True, anchor="end")
        x_a, x_b = cx, cx + rb * scale
        g.rect(min(x_a, x_b), y - 7, abs(x_b - x_a), 14, RES, r=3)
        g.text(
            x_b + (6 if rb >= 0 else -6),
            y + 4,
            f"{rb:+.1%}" if abs(rb) >= 0.0005 else "0.0%",
            size=9.5,
            fill=MUTED,
            anchor="start" if rb >= 0 else "end",
        )
        g.items.append(f'<circle cx="{cx + ra * scale}" cy="{y}" r="4.5" fill="{HID}" stroke="{INK}" stroke-width="0.9"/>')
    g.text(
        cx,
        top + row * len(toks) + 18,
        f"0: predicted = observed on the {ts['rows']:,} training rows",
        size=9.5,
        fill=MUTED,
        anchor="middle",
    )
    g.text(
        380,
        304,
        "Each step adds the pseudo-inverse of the Fisher information times the score s = Σ(y − p), as in the fisher-scoring package.",
        size=10,
        fill=MUTED,
        anchor="middle",
    )
    g.save("fisher")


# -------------------------------------------------------------------------------
# 4. Calibration: level and shape
# -------------------------------------------------------------------------------
def calibration() -> None:
    """Reliability diagrams of EventGPT zero-shot, the RLCR variant after its temperature, and EventGPT-D."""
    z = np.load(RUN / "s0" / "claims.npz")
    m = json.loads((RUN / "all_metrics.json").read_text())

    def mean(arm: str, key: str) -> float:
        return float(np.mean([r[key] for r in m[arm]]))

    panels = [
        ("EventGPT zero-shot", "native", "#8A99AA"),
        ("RLCR reward, temperature 1.4", "jev", WARM),
        ("EventGPT-D", "jev_ce", "#4F5BD5"),
    ]
    g = Svg(780, 330)
    g.legend([(DEC, "One bin of q; dot size shows the number of predictions"), (LINE, "q equal to the share that came true")])
    size, gap, x0, y0 = 190.0, 62.0, 64.0, 62.0
    for k, (title, arm, col) in enumerate(panels):
        q, c = z[f"q_{arm}"], z[f"c_{arm}"]
        px = x0 + k * (size + gap)
        g.text(px + size / 2, y0 - 8, title, size=11.5, weight=600, anchor="middle")
        g.rect(px, y0, size, size, "white", LINE, r=0)
        g.items.append(f'<line x1="{px}" y1="{y0 + size}" x2="{px + size}" y2="{y0}" stroke="{LINE}" stroke-width="1.2"/>')
        pts = []
        b = np.clip((q * 10).astype(int), 0, 9)
        for j in range(10):
            sel = b == j
            if sel.sum() >= 20:
                pts.append((px + size * q[sel].mean(), y0 + size * (1 - c[sel].mean()), sel.sum()))
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
        g.items.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="1.6"/>')
        for x, y, n in pts:
            g.items.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2.5 + 7 * (n / len(q)) ** 0.5:.1f}" fill="{col}" opacity="0.85"/>')
        for t in (0, 0.5, 1):
            g.text(px + size * t, y0 + size + 14, f"{t:g}", size=9, fill=MUTED, anchor="middle")
            g.text(px - 6, y0 + size * (1 - t) + 3, f"{t:g}", size=9, fill=MUTED, anchor="end")
        g.text(px + size / 2, y0 + size + 30, "stated confidence q", size=9.5, fill=MUTED, anchor="middle")
        g.text(
            px + size / 2,
            y0 + size + 50,
            f"slope {mean(arm, 'cal_slope'):.2f}, Gini gap {mean(arm, 'gini_gap'):+.3f}",
            size=10.5,
            weight=600,
            anchor="middle",
        )
        g.text(
            px + size / 2, y0 + size + 65, f"ECE {mean(arm, 'ece'):.3f}, ICE {mean(arm, 'ice'):.3f}", size=10, fill=MUTED, anchor="middle"
        )
    g.text(22, y0 + size / 2, "", size=9)
    g.save("calibration")


# -------------------------------------------------------------------------------
# The model: what is frozen, what fine-tuning trains, the residual head, the Fisher step
# -------------------------------------------------------------------------------
GRAY, DARK = F.GRAY, F.DARK
FROZEN = "#E4E9EF"
CX = 235.0


def mini_fan(g: Svg, x0: float, x1: float, y_top: float, y_bot: float, n_top: int, n_bot: int, color: str, top_fill: str) -> None:
    """Fully connected units, every bottom unit to every top unit, in one line color."""
    tops = [x0 + (x1 - x0) * k / (n_top - 1) for k in range(n_top)]
    pad = (x1 - x0) * 0.12
    bots = [x0 + pad + (x1 - x0 - 2 * pad) * k / (n_bot - 1) for k in range(n_bot)]
    for xb in bots:
        for xt in tops:
            g.items.append(
                f'<line x1="{xb:.1f}" y1="{y_bot}" x2="{xt:.1f}" y2="{y_top}" stroke="{color}" stroke-width="0.5" opacity="0.7"/>'
            )
    for x in tops:
        g.items.append(f'<circle cx="{x:.1f}" cy="{y_top}" r="5" fill="{top_fill}" stroke="{INK}" stroke-width="0.8"/>')
    for x in bots:
        g.items.append(f'<circle cx="{x:.1f}" cy="{y_bot}" r="5" fill="{HID}" stroke="{INK}" stroke-width="0.8"/>')


def legend_d(g: Svg) -> None:
    """The three roles a part of EventGPT-D can have."""
    g.legend([(FROZEN, "Frozen, from EventGPT"), (RES, "Trained by cross-entropy"), (EMB, "Intercept b, solved by Fisher scoring")])


def token_decoder(g: Svg) -> None:
    """The fine-tuned top of the stack, from the history state h to the next-event tokens."""
    box = F.box
    cx = CX
    g.text(cx, 50, "Next-event tokens and q", size=11, anchor="middle", mono=True)
    g.arrow([(cx, 104), (cx, 58)])
    g.rect(100, 80, 270, 176, RES_BAND, r=12)
    g.text(cx - 125, 97, "Token decoder", size=10.5, weight=600, fill=MUTED)
    g.rect(cx - 105, 106, 210, 34, RES, r=6)
    g.text(cx, 119, "Residual head", size=10.5, weight=600, anchor="middle")
    tex(g, cx, 133, Z, size=9, anchor="middle", color=MUTED)
    box(g, cx, 152, "Final LayerNorm", "white", WARM, h=22)
    box(g, cx, 184, "Causal self-attention block", RES, "none", note="within the next event", h=30)
    g.text(cx + 110, 204, "× 2", size=11, weight=600, fill=MUTED)
    box(g, cx, 226, "Token embeddings + h", FROZEN, "none", w=170, h=20)
    for y0, y1 in ((152, 140), (184, 174), (226, 214)):
        g.arrow([(cx, y0), (cx, y1)])
    g.items.append(f'<circle cx="{cx}" cy="276" r="12" fill="{EMB}"/>')
    g.text(cx, 280.5, "h", size=12, weight=600, fill="white", anchor="middle")
    g.arrow([(cx, 264), (cx, 246)])
    g.items.append(f'<line x1="56" y1="296" x2="414" y2="296" stroke="{WARM}" stroke-width="1.2" stroke-dasharray="5 4"/>')
    g.text(408, 290, "gradients stop at h", size=9.5, fill=WARM, anchor="end")
    g.text(cx + 20, 312, "h is computed once and cached", size=9.5, fill=MUTED)


def frozen_full(g: Svg) -> None:
    """The frozen part below h, block by block: the history decoder and the event encoder of EventGPT."""
    box = F.box
    cx = CX
    box(g, cx, 326, "Final LayerNorm", FROZEN, "none", h=22)
    g.arrow([(cx, 326), (cx, 288)])
    g.rect(100, 362, 270, 108, HID_BAND, r=12)
    g.text(cx - 125, 379, "History decoder", size=10.5, weight=600, fill=MUTED)
    box(g, cx, 390, "Causal multi-head attention", FROZEN, "none", note="over events, temporal RoPE", h=34)
    box(g, cx, 432, "Feed forward", FROZEN, "none", note="residual blocks, GELU", h=30)
    g.text(cx + 110, 452, "× 6", size=11, weight=600, fill=MUTED)
    g.arrow([(cx, 390), (cx, 348)])
    g.arrow([(cx, 432), (cx, 424)])
    box(g, cx, 488, F.sub("e", "1") + " to " + F.sub("e", "i") + ", one vector per event", ENC_FILL, "none", w=230, h=22)
    g.arrow([(cx, 488), (cx, 462)])
    g.rect(100, 524, 270, 84, F.ENC_BAND, r=12)
    g.text(cx - 125, 541, "Event encoder", size=10.5, weight=600, fill=MUTED)
    box(g, cx, 552, "Self-attention block, mean pool", FROZEN, "none", note="bidirectional, within each event", h=34)
    g.text(cx + 110, 596, "× 2", size=11, weight=600, fill=MUTED)
    g.arrow([(cx, 552), (cx, 510)])
    box(g, cx, 628, "Token embedding layer", FROZEN, "none", h=22)
    g.arrow([(cx, 628), (cx, 586)])
    box(g, cx, 664, "Tokenized events", TOK, "none", h=22)
    g.arrow([(cx, 664), (cx, 650)])
    g.arrow([(cx, 716), (cx, 686)])
    g.text(cx, 732, "A client's events", size=11, anchor="middle", mono=True)


def frozen_compact(g: Svg) -> None:
    """The frozen part below h as one block."""
    box = F.box
    cx = CX
    box(g, cx, 326, "EventGPT: event encoder, history decoder", FROZEN, "none", note="2 + 6 layers, frozen", w=250, h=36)
    g.arrow([(cx, 326), (cx, 288)])
    box(g, cx, 380, "Tokenized events", TOK, "none", h=22)
    g.arrow([(cx, 380), (cx, 362)])
    g.arrow([(cx, 430), (cx, 402)])
    g.text(cx, 446, "A client's events", size=11, anchor="middle", mono=True)


def callout_head(g: Svg, ay: float) -> None:
    """The residual head: frozen W0 beside the trainable A, which starts at zero."""
    ax, aw, ah = 470.0, 410.0, 232.0
    g.rect(ax, ay, aw, ah, "white", INK, r=18)
    mini_fan(g, ax + 28, ax + 188, ay + 44, ay + 104, 7, 6, "#9AA6B2", FROZEN)
    mini_fan(g, ax + 222, ax + 382, ay + 44, ay + 104, 7, 6, WARM, RES)
    w = tex(g, ax + 70, ay + 28, "W_0", size=11)
    g.text(ax + 72 + w, ay + 28, ", frozen", size=10, weight=600)
    w = tex(g, ax + 245, ay + 28, "A", size=11, color=WARM)
    g.text(ax + 247 + w, ay + 28, ", zero at the start", size=10, weight=600, fill=WARM)
    tex(g, ax + 108, ay + 124, r"y \in \mathbb{R}^{384}", size=9.5, anchor="middle", color=MUTED)
    tex(g, ax + 302, ay + 124, r"\mathrm{std}(y)", size=9.5, anchor="middle", color=MUTED)
    F.plus(g, ax + 205, ay + 44)
    tex(g, ax + aw / 2, ay + 150, Z + r",\quad z \in \mathbb{R}^{755}", size=11, anchor="middle")
    F.lines(
        g,
        ax + 24,
        ay + 178,
        [
            "EventGPT's output layer stays frozen. A residual on the",
            "standardized state starts at zero, so training starts at",
            "EventGPT zero-shot (Geuter, Cheng and Alvarez-Melis 2026).",
        ],
    )
    F.curve(g, ax, ay + 70, CX + 105, 123)


def callout_ce(g: Svg, by: float, bh: float = 196.0) -> None:
    """Cross-entropy on the true next event of case 244050, with each token's probability before and after."""
    ax, aw = 470.0, 410.0
    g.rect(ax, by, aw, bh, "white", INK, r=18)
    g.text(ax + 24, by + 26, "True next event, case 244050", size=10, weight=600)
    toks = [("[EVT]", None), ("name=cb_pay", 0), ("dt=101", None), ("n=3", 1), ("late=1", 2), ("pdpd=b2", 3)]
    x = ax + 24
    zs, ds = D["truth_steps_zero_shot"], D["truth_steps_d"]
    for t, k in toks:
        w = pill_w(t)
        g.rect(x, by + 40, w, 19, DEC if k is not None else TOK, r=6)
        g.text(x + w / 2, by + 53.2, t, size=10, anchor="middle", mono=True)
        if k is not None:
            g.text(x + w / 2, by + 76, f"{zs[k]['p_chosen']:.2f}", size=9.5, fill=MUTED, anchor="middle")
            g.text(x + w / 2, by + 91, f"{ds[k]['p_chosen']:.2f}", size=10, weight=600, fill=WARM, anchor="middle")
        x += w + 5
    g.text(ax + aw - 18, by + 76, "zero-shot", size=9, fill=MUTED, anchor="end")
    g.text(ax + aw - 18, by + 91, "trained", size=9, fill=WARM, anchor="end")
    tex(g, ax + 24, by + 116, r"\mathcal{L} = -\sum_{t} \log p(x_t \mid x_{<t}, h)", size=11)
    F.lines(
        g,
        ax + 24,
        by + 144,
        [
            "It trains the token decoder and the residual head on",
            "24,000 cases; gradients stop at h, so EventGPT itself",
            "is unchanged.",
        ],
    )
    g.arrow([(ax, by + 40), (456, by + 40), (456, 199), (372, 199)], MUTED)


def callout_fisher(g: Svg, cy0: float, ch: float = 222.0) -> None:
    """The Fisher step on the intercept b, with the largest score by step."""
    ax, aw = 470.0, 410.0
    g.rect(ax, cy0, aw, ch, "white", INK, r=18)
    tex(g, ax + 24, cy0 + 32, r"s = \textstyle\sum (y - p)", size=11)
    tex(g, ax + 24, cy0 + 58, r"I = \textstyle\sum \left(\mathrm{diag}(p) - p\,p^{\top}\right)", size=11)
    tex(g, ax + 24, cy0 + 84, r"b \leftarrow b + I^{+} s", size=11)
    log = D["fisher_log"][:5]
    bx, base = ax + 250.0, cy0 + 92
    for k, v in enumerate(log):
        hgt = max(1.0, 62 * v / log[0])
        g.rect(bx + k * 28, base - hgt, 18, hgt, RES if k == 0 else EMB, r=3)
        g.text(bx + k * 28 + 9, base + 12, str(k), size=8.5, fill=MUTED, anchor="middle")
        g.text(bx + k * 28 + 9, base - hgt - 4, f"{v:.0f}" if v >= 10 else f"{v:.1f}", size=8.5, fill=INK, anchor="middle")
    g.text(bx + 64, cy0 + 18, "largest score by step", size=9, fill=MUTED, anchor="middle")
    F.lines(
        g,
        ax + 24,
        cy0 + 136,
        [
            "After training, Fisher scoring solves for b alone. In",
            "three steps the predicted probabilities of every token",
            "sum to its observed count on the training rows: the",
            "maximum-likelihood fit of an intercept.",
        ],
    )
    g.arrow([(ax, cy0 + 100), (444, cy0 + 100), (444, 132), (342, 132)], MUTED)


def callout_q(g: Svg, qy: float, qh: float = 176.0) -> None:
    """The confidence q: the model's own probability of the event it writes, summed over n."""
    ax, aw = 470.0, 410.0
    g.rect(ax, qy, aw, qh, "white", INK, r=18)
    toks = ["[EVT]", "name=cb_pay", "dt=101", "n=3", "late=1", "pdpd=b2"]
    x = ax + 24
    for t in toks:
        w = pill_w(t)
        g.rect(x, qy + 22, w, 19, TOK if t in ("[EVT]", "dt=101") else DEC, r=6)
        g.text(x + w / 2, qy + 35.2, t, size=10, anchor="middle", mono=True)
        x += w + 5
    expr = r"q = P(\text{kind}) \textstyle\sum_n P(n)\,P(\text{late} \mid n)\,P(\text{pdpd} \mid n, \text{late})"
    tex(g, ax + 24, qy + 72, expr, size=11)
    g.text(ax + 24, qy + 98, f"= {D['q_terms_d']['q']:.2f} for case 244050", size=10.5, weight=600)
    F.lines(
        g,
        ax + 24,
        qy + 130,
        [
            "The confidence is the model's own probability of the",
            "event it writes, summed over n, which is not judged.",
            "There is no separate confidence head.",
        ],
    )
    g.arrow([(ax, qy + 60), (452, qy + 60), (452, 40), (CX + 84, 40)], MUTED)


def callout_history(g: Svg, hy: float, hh: float = 250.0) -> None:
    """The frozen half: the case's history read once into h, which fine-tuning reuses."""
    ax, aw = 470.0, 410.0
    g.rect(ax, hy, aw, hh, "white", INK, r=18)
    g.text(ax + 24, hy + 30, f"Case 244050: {D['case']['n_events']} events before the cut", size=10, weight=600)
    rows = [[t for t in e if t != "[EVT]"] for e in D["case"]["last_events"][-3:]]
    for r, row in enumerate(rows):
        x = ax + 24
        for t in row:
            if x + pill_w(t) > ax + aw - 20:
                break
            x += g.pill(x, hy + 50 + r * 30, t) + 5
    g.text(ax + 24, hy + 158, "the last three, oldest first", size=9.5, fill=MUTED)
    F.lines(
        g,
        ax + 24,
        hy + 196,
        [
            "EventGPT reads the whole history once. Its state h at",
            "the last event before the cut is cached, so fine-tuning",
            "runs only the token decoder: one cached h per case.",
        ],
    )
    g.arrow([(ax, hy + 60), (452, hy + 60), (452, 416), (372, 416)], MUTED)


def model() -> None:
    """EventGPT-D as one stack with all three callouts."""
    g = Svg(900, 790)
    legend_d(g)
    g.rect(40, 66, 390, 690, "white", LINE, r=16)
    g.text(52, 742, "EventGPT-D", size=13, weight=700)
    token_decoder(g)
    frozen_full(g)
    callout_head(g, 66.0)
    callout_ce(g, 318.0)
    callout_fisher(g, 534.0)
    g.save("model")


def architecture() -> None:
    """The stack with the residual head: what is frozen and what fine-tuning trains."""
    g = Svg(900, 790)
    legend_d(g)
    g.rect(40, 66, 390, 690, "white", LINE, r=16)
    g.text(52, 742, "EventGPT-D", size=13, weight=700)
    token_decoder(g)
    frozen_full(g)
    callout_head(g, 66.0)
    callout_q(g, 314.0)
    callout_history(g, 506.0)
    g.save("architecture")


def training() -> None:
    """Training in two steps: cross-entropy on the true next event, then the Fisher step on the intercept."""
    g = Svg(900, 486)
    legend_d(g)
    g.rect(40, 66, 390, 400, "white", LINE, r=16)
    g.text(52, 452, "EventGPT-D", size=13, weight=700)
    token_decoder(g)
    frozen_compact(g)
    callout_ce(g, 66.0, 186.0)
    callout_fisher(g, 268.0, 198.0)
    g.save("training")


if __name__ == "__main__":
    import sys

    for name in sys.argv[1:] or ["model", "architecture", "training", "confidence", "fisher", "calibration"]:
        globals()[name]()
