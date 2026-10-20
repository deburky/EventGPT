# EventGPT: A Hierarchical Foundation Model for Customer Event Histories

Working draft by Denis Burakov. EventGPT is a causal transformer pretrained by next-token prediction on
customer event histories: an event encoder pools each event into a vector, a history decoder models the
event vectors with rotary positions set by log elapsed time, and a token decoder predicts the next event.
The paper evaluates its frozen embeddings on the product-propensity task of the public Multimodal
Banking Dataset (MBD-mini).

Numbers marked **TBD** in the PDF are pending runs; every other number is measured.

## Contents

| File | What it is |
|------|------------|
| `eventgpt.tex` | LaTeX source (XeLaTeX) |
| `figures/*.pdf` | Figures, drawn in Illustrator; editable sources in `figures/src/*.ai` |
| `Makefile` | `make` builds `eventgpt.pdf`, `make zip` stages the arXiv upload in `tarball/` |

The code that produces the results (tokenization, model, training, probe, baseline) is in `../src/eventgpt`.

## Build

```bash
make          # or, from the repository root: just paper
```
