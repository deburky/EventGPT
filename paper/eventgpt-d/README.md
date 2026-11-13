# EventGPT-D figures

Diagrams of EventGPT-D in the style of the EventGPT paper. They reuse the primitives and colors of
`paper/figures/src/figures.py` and add one color, orange, for the residual head and its training.

EventGPT-D claims a client's next event and states its confidence q as its own probability of the claim. It keeps
EventGPT's output layer W0 and adds the residual head z = W0 y + A std(y) + b of Geuter, Cheng and Alvarez-Melis
(2026, arXiv:2610.17205). It trains by cross-entropy on the true next event and solves for the intercept b by Fisher
scoring. The code is `eventgpt.homecredit.rlcr` (arm `jev_ce`, Fisher step on by default), and the calibration metrics
are in `eventgpt.homecredit.calibration`.

| figure | shows |
|---|---|
| `claim` | the history state, the claim written token by token, the output head, q, and the two training steps |
| `confidence` | the claim as a path through the decoder's choices, with q as the product along the path |
| `fisher` | the Fisher step: the largest score per step, and predicted against observed counts per token |
| `calibration` | reliability diagrams of EventGPT zero-shot, the RLCR variant after its temperature, and EventGPT-D |

The case is Home Credit test case 244050 (row 563 of the test fold, seed 0). Its numbers are in `src/case563.json`. The
reliability diagrams read `data/runs/hc-rlcr-s42/v4`.

    uv run python paper/eventgpt-d/figures/src/figures_d.py

This writes `figures/<name>.pdf` and `figures/<name>.png`, and the SVG sources to `figures/src/`. It needs
rsvg-convert, Inter and JetBrains Mono.
