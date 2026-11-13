"""EventGPT-D diagrams in the style of the EventGPT paper, on Home Credit test case 244050.

The primitives and colors come from paper/figures/src/figures.py. The case is row 563 of the test fold.

EventGPT-D claims the next event and states its confidence q as its own probability of the claim. It keeps EventGPT's
output layer W0 and adds the residual head z = W0 y + A std(y) + b of Geuter, Cheng and Alvarez-Melis (2026,
arXiv:2610.17205), trains by cross-entropy on the true next event, and solves for the intercept b by Fisher scoring.
The case's numbers are in case563.json (from eventgpt.homecredit.rlcr, seed 0); the reliability diagrams read the
saved per-case confidences of data/runs/hc-rlcr-s42/v4.

Run from the repository root: `uv run python paper/eventgpt-d/figures/src/figures_d.py` (needs rsvg-convert, Inter
and JetBrains Mono).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
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
RES, RES_BAND = "#FBDDBF", "#FEF3E8"  # the residual head and its training, new in EventGPT-D
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


# -------------------------------------------------------------------------------
# 1. EventGPT-D at a glance
# -------------------------------------------------------------------------------
def claim() -> None:
    """History state, the claim written token by token, the residual head, q, and the two training steps."""
    g = Svg(760, 430)
    g.legend([(TOK, "Tokens"), (HID, "Hidden states"), (DEC, "Claim"), (RES, "Residual head"), (EMB, "Confidence")])
    # the history
    g.rect(20, 46, 196, 150, "white", LINE, r=12)
    g.text(32, 67, "Case 244050: 60 events", size=11.5, weight=600)
    hist = [["name=cb_pay", "late=1", "pdpd=b2"], ["name=tax", "src=b"], ["name=cb_pay", "late=1", "pdpd=b1"]]
    y = 80.0
    for row in hist:
        x = 30.0
        for t in row:
            x += g.pill(x, y, t) + 4
        y += 25
    g.text(32, 172, "last events before the cut", size=9.5, fill=MUTED)
    g.arrow([(118, 196), (118, 222)])
    g.rect(78, 224, 80, 30, HID, r=8)
    g.text(118, 244, "h", size=12, weight=600, anchor="middle")
    g.text(118, 272, "history state at the cut", size=9.5, fill=MUTED, anchor="middle")
    # the claim, token by token
    g.rect(246, 46, 494, 150, DEC_BAND, r=12)
    g.text(262, 67, "Token decoder writes the claim", size=11.5, weight=600)
    g.text(262, 83, "probability of each token as written, EventGPT-D", size=9.5, fill=MUTED)
    toks = [("[EVT]", None), ("name=cb_pay", 0), ("dt=101", None), ("n=3", 1), ("late=1", 2), ("pdpd=b2", 3)]
    x = 262.0
    steps = D["eventgpt_d"]["steps"]
    centers = []
    for t, k in toks:
        w = pill_w(t)
        fill = DEC if k is not None else TOK
        g.rect(x, 128, w, 19, fill, r=6)
        g.text(x + w / 2, 141.2, t, size=10, anchor="middle", mono=True)
        if k is not None:
            g.text(x + w / 2, 118, f"{steps[k]['p_chosen']:.2f}", size=10.5, weight=600, anchor="middle")
        centers.append(x + w / 2)
        x += w + 8
    g.text(centers[2], 118, "given", size=9.5, fill=MUTED, anchor="middle")
    g.text(262, 180, "The claim is judged on its kind and key fields; n is written but not judged.", size=9.5, fill=MUTED)
    g.arrow([(158, 239), (246, 239), (246, 196)])
    # the output head under the claim
    g.rect(246, 222, 300, 64, RES_BAND, r=12)
    g.text(262, 243, "Output head", size=11.5, weight=600)
    g.text(262, 262, "z = " + F.sub("W", "0") + "y + A std(y) + b", size=11, mono=True)
    g.text(262, 278, F.sub("W", "0") + " pretrained and frozen; A = 0, b = 0 at the start", size=9.5, fill=MUTED)
    g.arrow([(396, 222), (396, 197)])
    # confidence
    q = D["q_terms_d"]["q"]
    g.rect(574, 222, 166, 64, EMB, r=12)
    g.text(657, 247, f"q = {q:.2f}", size=13, weight=600, fill="white", anchor="middle")
    g.text(657, 265, "probability of the claim,", size=9.5, fill="#D5DEE8", anchor="middle")
    g.text(657, 278, "summed over n", size=9.5, fill="#D5DEE8", anchor="middle")
    g.arrow([(657, 197), (657, 222)])
    # training
    g.rect(20, 318, 350, 92, RES_BAND, r=12)
    g.text(36, 340, "1. Cross-entropy on the true next event", size=11.5, weight=600)
    g.text(36, 359, "trains the token decoder and the residual head;", size=10, fill=MUTED)
    g.text(36, 374, "the history decoder stays frozen", size=10, fill=MUTED)
    g.text(36, 395, "Geuter, Cheng and Alvarez-Melis (2026)", size=9.5, fill=MUTED)
    g.arrow([(370, 364), (390, 364)])
    g.rect(390, 318, 350, 92, RES_BAND, r=12)
    g.text(406, 340, "2. Fisher scoring of the intercept b", size=11.5, weight=600)
    g.text(406, 359, "b + I⁺ s, everything else frozen, until for every token", size=10, fill=MUTED)
    g.text(406, 374, "Σp = Σy on the training rows", size=10, fill=MUTED)
    g.text(406, 395, "maximum-likelihood score equations of b", size=9.5, fill=MUTED)
    g.save("claim")


# -------------------------------------------------------------------------------
# 2. Confidence as a product of the decoder's probabilities
# -------------------------------------------------------------------------------
def confidence() -> None:
    """The claim as a path through the decoder's choices; branch width is probability, q is the product along the path."""
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
    g.text(
        400,
        292,
        f"q = P(kind) × Σn P(n) P(late | n) P(pdpd | n, late) = {qt['p_kind']:.3f} × {qt['sum_n']:.3f}; width shows probability",
        size=10,
        fill=MUTED,
        anchor="middle",
    )
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
    g.legend([(DEC, "One bin of q; dot size shows the number of claims"), (LINE, "q equal to the share that came true")])
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


if __name__ == "__main__":
    import sys

    for name in sys.argv[1:] or ["claim", "confidence", "fisher", "calibration"]:
        globals()[name]()
