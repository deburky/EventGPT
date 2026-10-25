# EventGPT: Next-Event Pretraining on Banking Customer Histories

Working draft by Denis Burakov. EventGPT is a causal transformer pretrained by next-token prediction on
customer event histories: an event encoder pools each event into a vector, a history decoder models the
event vectors with rotary positions set by elapsed time, and a token decoder predicts the next event.
The paper evaluates its zero-shot scores and frozen embeddings on the public Multimodal Banking Dataset
(MBD-mini).

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
