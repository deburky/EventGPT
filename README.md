# EventGPT

EventGPT is an open generative pretrained transformer for banking event sequences. It reads a customer's
history of transactions and other events, learns by predicting the next event token by token, and gives
one embedding per customer and point in time. Its next-event distribution also scores event targets with
no labels at all.

The model is pretrained on the public Multimodal Banking Dataset (MBD-mini: 100,224 clients, 38.8 million
transactions, 66.3 million geolocation pings). The weights are on the Hugging Face Hub as
[`deburky/EventGPT`](https://huggingface.co/deburky/EventGPT), and every number in the paper can be rerun
from this repository.

Paper: *EventGPT: Next-Event Pretraining on Banking Customer Histories* (Burakov, 2026), in [`paper/`](paper/).

## Results

ROC-AUC on MBD-mini. The product targets are averaged over five test folds and two pretraining seeds;
the event targets are on test fold 0.

| | EventGPT | CatBoost, hand-built features |
|---|---|---|
| Five "first transaction of a kind next month" targets, no labels | 0.79 to 0.92 | passes after 10,000 to 300,000 labeled rows on four targets, still below at 300,000 on the fifth |
| Four product targets, logistic head on the frozen embedding | 0.787 | 0.808 |
| The same, with profile fields and full-history token counts beside the embedding | 0.804 | 0.808 |
| Rank average of the two models | 0.824 | |

The rank average scores 0.014 to 0.018 above the hand-built features on every fold and seed. On
transactions alone the embedding reaches 0.793, level with token counts over the full transaction
history, and pretraining only on transactions before the label period lowers it by 0.004. The paper has
the comparisons, standard errors and ablations.

## Install

```bash
pip install git+https://github.com/deburky/EventGPT
```

or, for development, clone the repository and run `uv sync`.

## Quickstart

Download MBD-mini and unpack its event tables:

```bash
hf download ai-lab/MBD-mini --repo-type dataset --local-dir data/MBD-mini/raw
mkdir -p data/MBD-mini/extracted && tar -xzf data/MBD-mini/raw/detail.tar.gz -C data/MBD-mini/extracted
```

Embed three clients at the start of October 2022 and score the five first-kind targets:

```python
import duckdb
import pandas as pd

from eventgpt import api
from eventgpt.hub import from_pretrained

root = "data/MBD-mini/extracted/detail"
clients = duckdb.sql(f"select distinct client_id from read_parquet('{root}/trx/fold=0/*.parquet') limit 3").df()["client_id"].tolist()
ids = ", ".join(f"'{c}'" for c in clients)
query = "select * exclude (fold) from read_parquet('{}/{}/*/*.parquet', hive_partitioning=true) where client_id in ({})"
trx = duckdb.sql(query.format(root, "trx", ids)).df()
geo = duckdb.sql(query.format(root, "geo", ids)).df()

model, vocab = from_pretrained("deburky/EventGPT", "aligned-s42")
cuts = pd.DataFrame({"client_id": clients, "cut": pd.Timestamp("2022-10-01")})
emb = api.embed(model, vocab, cuts, trx, geo)          # (3, 384): the history state at each cut
p = api.next_event_probs(model, vocab, cuts, trx, geo) # (3, 5): P(next event is each kind), zero for clients who had it
```

A history is every event strictly before its cut, so an embedding never holds anything after its cut.
`api.embed` goes through the same code as the paper's probe: `scripts/check_api_parity.py` checks that
the tokens are identical and the embeddings match the paper's to float precision.

## Variants

| variant | events | training windows |
|---|---|---|
| `aligned-s42`, `aligned-s43` | transactions and geolocation | aligned, QK-norm |
| `aligned-profile-s42`, `aligned-profile-s43` | as above, with the profile path | aligned, QK-norm |
| `trx-s42`, `trx-s43` | transactions only | random offsets |

For the transactions-only variants pass the geolocation pings with `geo_events=False`: they only date
each client's profile event, as in training.

## Batch scoring

The intended use is a batch job: once a month, embed every customer at the month's start and store the
vectors as point-in-time features beside the label table. Section 4 of the paper gives the conditions
under which a stored embedding holds nothing after its cut. The paper's probe embedded the 242,604
client-months of the test fold, with histories of up to 2,048 tokens, in about 20 minutes on one NVIDIA L4.

A daily job can rescore only the clients whose input changed: those with a new event since the last run,
and those whose now event moved to a new time-gap bucket as their silence grew. Everyone else keeps
yesterday's scores, and the result equals a full pass. On MBD-mini that is 90% of clients on an average
day (79% on transactions alone), since most clients act every few days; the saving grows on books where
clients stay silent for weeks. Section 4 of the paper has the rule.

## Reproducing the paper

The `egpt` command runs each step; `egpt <command> --help` shows its flags. Run it from the repository
root: it reads the unpacked MBD-mini tables from `data/MBD-mini/extracted` (unpack `targets.tar.gz` and
`client_split.tar.gz` there too) and writes the token streams to `data/MBD-mini/tokens`. The environment
variables `MBD_ROOT` and `MBD_OUT` move them.

```bash
egpt prepare --test-fold 0                                  # vocabulary and token streams
egpt train --run hier-alq-s42 --arch eventgpt --rope --aligned --qk-norm \
    --steps 20000 --batch 8 --accum 2 --seed 42             # about 2 hours on one NVIDIA L4, to data/ckpt
egpt probe data/ckpt/hier-alq-s42.pt --now --out data/runs/hier-alq-s42/probe   # embeddings and logistic heads
egpt zero-shot data/ckpt/hier-alq-s42.pt --probe-dir data/runs/hier-alq-s42/probe
egpt catboost                                               # hand-built aggregates baseline
egpt token-counts --probe-dir data/runs/hier-alq-s42/probe  # token-count baseline
```

`scripts/gcp.py` runs training and the probe on a Google Cloud L4 that deletes itself when done
(`uv run --group gcp scripts/gcp.py up <run> [train args]`, with Application Default Credentials).

## Your own events

The released weights read MBD's event schema. Events from another institution need their own vocabulary
and pretraining: map them to the event table that `eventgpt.data.mbd` builds (one row per event with a
time, a kind and its fields), then run `egpt prepare` and `egpt train` on it. The model and the training
code do not depend on MBD.

## Repository

| path | contents |
|---|---|
| `src/eventgpt/data` | tokenization of MBD-mini, vocabulary, training windows |
| `src/eventgpt/models` | EventGPT and the flat decoder it is compared with |
| `src/eventgpt/training` | pretraining and fine-tuning |
| `src/eventgpt/evaluation` | probe, zero-shot scoring, profile readout, standard errors |
| `src/eventgpt/baselines` | CatBoost on hand-built features, token counts and the embedding |
| `src/eventgpt/api.py`, `hub.py` | embeddings from event tables, released checkpoints |
| `paper/` | LaTeX source of the paper |

## License and citation

The code is under Apache-2.0. The weights are under CC BY 4.0, the license of MBD-mini, on which they
were trained.

```bibtex
@misc{burakov2026eventgpt,
  title  = {EventGPT: Next-Event Pretraining on Banking Customer Histories},
  author = {Burakov, Denis},
  year   = {2026},
  url    = {https://github.com/deburky/EventGPT}
}
```
