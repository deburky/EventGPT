"""Data for the futures figure: one test client's real month beside months sampled from EventGPT.

The client is drawn with a fixed seed among the test rows positive for the commonest event target, outside
January 2023. For it we record the events of the two weeks before the as-of date, the events of the label
month, the model's distribution over the next event's name, three sampled months with every sampled event,
and the rollout estimate of the target's monthly probability from 64 sampled months, beside the single-step
zero-shot score. Everything is written to one JSON file that figures.py draws.

Run from the repository root: `uv run python paper/figures/src/futures_data.py`
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eventgpt.data.mbd import OUT
from eventgpt.data.vocab import Vocab
from eventgpt.evaluation.probe import FIRST_KINDS, Fold, history_tokens, mark_undefined
from eventgpt.evaluation.rollout import decode, pad, path_mean, prefill, sample_month, tables
from eventgpt.evaluation.zero_shot import next_event_scores
from eventgpt.models.flat import load_checkpoint

RUN = Path("data/runs/hier-alq-s42")
TARGET = (50, 19)
SHOWN_PATHS = 3
HISTORY_EVENTS = 10
SCORED_PATHS = 64
DAY = 86_400.0


# -------------------------------------------------------------------------------
# Events read back off the stream
# -------------------------------------------------------------------------------
def events(fold: Fold, inv: dict[int, str], start: int, stop: int, cut: int) -> list[dict[str, object]]:
    """Events start..stop of the fold as (days from the cut, kind), geolocation as kind "geo"."""
    out: list[dict[str, object]] = []
    for e in range(start, stop):
        if fold.is_trx[e]:
            kind = f"{int(fold.etype[e])}/{int(fold.esub[e])}"
        else:
            name = inv[int(fold.tokens[fold.event_start[e] + 1])]
            kind = "geo" if name == "name=geo" else name.removeprefix("name=")
        out.append({"day": float((fold.event_time[e] - cut) / DAY), "kind": kind})
    return out


def main() -> None:
    """Pick the client, score and sample its month, write futures.json."""
    torch.manual_seed(0)
    vocab = Vocab.load(OUT / "vocab")
    inv = {i: t for t, i in vocab.structured.items()}
    fold = Fold(0, vocab)
    te = mark_undefined(pd.read_parquet(RUN / "probe-now-2k" / "test_rows.parquet"))
    label = f"first_{TARGET[0]}_{TARGET[1]}"
    pool = te[(te[label] == 1) & (pd.to_datetime(te["mon"]) < "2023-01-01")]
    row = pool.iloc[int(np.random.default_rng(0).integers(len(pool)))]
    ci, n_hist, n_month = int(fold.index[row["client_id"]]), int(row["n_hist"]), int(row["n_month"])
    cut, cut_end = int(row["cut"]), int(row["cut_end"])
    a = int(fold.client_events[ci])

    model = load_checkpoint(str(RUN / "hier-alq-s42.pt")).eval()
    block = int(json.loads((RUN / "probe-now-2k" / "results.json").read_text())["block"])
    hist = history_tokens(fold, ci, n_hist, block, aligned=model.cfg.aligned)
    silence = float(cut - fold.event_time[a + n_hist - 1])
    month = float(cut_end - cut)
    dev = torch.device("cpu")
    tb = tables(vocab, dev)

    # the next event's name, given the history
    xt, real = pad([hist], dev)
    st, _ = prefill(model, xt, real, 1)
    p_name = decode(model, torch.full((1, 1), tb.evt), st.h, 0).exp()[0, tb.names].numpy()
    top = np.argsort(-p_name)[:5]
    next_kind = events(fold, inv, a + n_hist, a + n_hist + 1, cut)[0]["kind"]
    candidates = [{"name": inv[int(tb.names[j])].removeprefix("name="), "p": float(p_name[j])} for j in top]
    next_name = f"name=trx_{str(next_kind).split('/')[0]}" if str(next_kind)[0].isdigit() else f"name={next_kind}"
    nj = int((tb.names == vocab.id(next_name)).nonzero()[0, 0])
    next_p, next_rank = float(p_name[nj]), int((p_name > p_name[nj]).sum()) + 1

    # sampled months: a few traced for drawing, more for the monthly probability
    def sample(k: int, seed: int, trace: list[dict[str, np.ndarray]] | None) -> dict[str, torch.Tensor]:
        st, _ = prefill(model, xt, real, 120)
        st = st.repeat(k)
        gen = torch.Generator().manual_seed(seed)
        s_t, m_t = torch.full((k,), silence), torch.full((k,), month)
        return sample_month(model, st, tb, s_t, m_t, 120, gen, trace)

    trace: list[dict[str, np.ndarray]] = []
    sample(SHOWN_PATHS, 0, trace)
    paths = []
    for p in range(SHOWN_PATHS):
        evs = []
        for step in trace:
            if not step["inside"][p]:
                continue
            name = inv[int(step["name"][p])].removeprefix("name=")
            kind = f"{name.removeprefix('trx_')}/{int(step['sub'][p])}" if name.startswith("trx_") else name
            evs.append({"day": float(step["t"][p] / DAY), "kind": kind})
        paths.append(evs)
    k_target = f"{TARGET[0]}/{TARGET[1]}"
    scored = sample(SCORED_PATHS, 1, None)
    j = FIRST_KINDS.index(TARGET)
    rollout = float(path_mean(scored["comp"], SCORED_PATHS)[0, j])
    head = np.array([vocab.id("[EVT]"), vocab.id(f"name=trx_{TARGET[0]}")])
    zs_seq = history_tokens(fold, ci, n_hist, block, np.r_[head, vocab.dt_tokens(np.array([silence]))], model.cfg.aligned)
    single = float(next_event_scores(model, [zs_seq], vocab, dev)[0, j])
    # where the client's single-step score ranks among the uniform test sample (scores saved by zero_shot)
    saved = pd.read_parquet(RUN / "zero_shot_scores.parquet")
    uniform = saved[saved["in_rand"]]
    score = uniform[f"zs_{label}"] * uniform[f"unseen_{label}"]
    share_above = float((score > single).mean())

    out = {
        "client_row": {"mon": str(row["mon"])[:10], "pool": len(pool)},
        "target": k_target,
        "history": events(fold, inv, max(a, a + n_hist - HISTORY_EVENTS), a + n_hist, cut),
        "actual": events(fold, inv, a + n_hist, a + n_month, cut),
        "month_days": month / DAY,
        "next_kind": next_kind,
        "next_p": next_p,
        "next_rank": next_rank,
        "names": len(tb.names),
        "share_above": share_above,
        "prevalence": float(te[label].mean()),
        "candidates": candidates,
        "paths": paths,
        "rollout": rollout,
        "single_step": single,
        "scored_paths": SCORED_PATHS,
    }
    dest = RUN / "futures.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("history", "actual", "paths")}, indent=1))
    print(f"{len(out['history'])} history events, {len(out['actual'])} actual, {[len(p) for p in paths]} sampled; written to {dest}")


if __name__ == "__main__":
    main()
