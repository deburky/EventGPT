"""Multi-step rollouts on Home Credit: simulated days-past-due paths over six months against what happened.

The history is cut at the first day of the month six months before the decision month, and the six calendar months
from there to the decision month are held out. Each case's observed path is the worst days past due (`pdpd`, buckets
as in eventgpt.homecredit.migration) of its bureau payment month in each held-out month, or missing when the stream
has no payment month there. The current bucket is the `pdpd` of the last payment month before the cut.

A rollout writes one payment month per held-out month: after the history the generator gets
`[EVT] name=cb_pay dt=<g>`, with g the gap to the first day of the month, samples n, late, pdpd and pov one token at
a time from its own softmax, appends the event to the history and moves on to the next month. Each case gets
`--paths` independent paths. The rollout conditions on a payment month in every month and writes no other events,
so it simulates the bureau's monthly days past due and nothing else. The history decoder runs on the whole window
at every step; the token decoder reads its state at the last event, as in eventgpt.homecredit.rlcr.

Generators: EventGPT zero-shot (the pretrained checkpoint) and EventGPT-D (the jev_ce arm of
eventgpt.homecredit.rlcr, rebuilt from its run folder with the Fisher-scored intercept). Both were trained or
fine-tuned on the next event only, so every step after the first feeds them their own samples.

Baselines from the training folds (1 to 4), with the same cut: the one-month roll-rate matrix M, chained as
M^(j+1) for month j (a Markov chain, also sampled for path events), and the direct j-month matrix from the current
bucket to month j, which needs no chaining but gives no joint path.

Metrics per held-out month, on uniform rows with an observed payment month: the share of each bucket, simulated
against observed; log loss and Brier over buckets; and calibration of P(late, b1 or worse) and P(30+ days, b3 or
worse). Path events (late, or 30+ days, in any observed month of the six) compare the rollout with the sampled
Markov chain. Example:

    MBD_OUT=data/home-credit/tokens uv run python -m eventgpt.homecredit.rollout data/runs/hc-alq-s42/ckpt/hc-alq-s42.pt \
        --d-run data/runs/hc-rlcr-s42/v4/s0 --out data/runs/hc-roll-s42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from eventgpt.data.mbd import OUT
from eventgpt.data.vocab import Vocab
from eventgpt.evaluation.probe import Fold, history_tokens
from eventgpt.homecredit import calibration
from eventgpt.homecredit.migration import BUCKETS, pay_bucket
from eventgpt.homecredit.prepare import N_FOLDS
from eventgpt.homecredit.rlcr import Grammar, decode, evaluate, fisher_intercept, install_jev_head, masked_logp, metrics, split
from eventgpt.homecredit.zero_shot import Events, auc
from eventgpt.models.flat import load_checkpoint
from eventgpt.training.train import device

H = 6  # held-out months
FIELDS = ["n", "late", "pdpd", "pov"]  # the cb_pay fields in stream order
EV_LEN = 3 + len(FIELDS)
NB = len(BUCKETS)
LATE = slice(1, 7)  # b1..b6
DPD30 = slice(3, 7)  # b3..b6


# -------------------------------------------------------------------------------
# Rows
# -------------------------------------------------------------------------------
def month_bounds(dd: pd.Series) -> np.ndarray:
    """(rows, H + 1) epoch seconds of the first day of each held-out month and of the decision month."""
    first = pd.to_datetime(dd).dt.to_period("M").dt.to_timestamp()
    cols = [first - pd.offsets.MonthBegin(H - j) for j in range(H)] + [first]
    return np.stack([c.to_numpy().astype("datetime64[s]").astype(np.int64) for c in cols], axis=1)


def horizon_rows(fold: Fold, vocab: Vocab, k: int, max_cases: int | None, seed: int) -> pd.DataFrame:
    """Cases of fold k with a payment month before the cut: current bucket and the observed bucket of each held-out month."""
    t = pd.read_parquet(OUT / "labels.parquet")
    t = t[t["fold"] == k].reset_index(drop=True)
    if max_cases:
        t = t.sample(min(max_cases, len(t)), random_state=seed).reset_index(drop=True)
    t["case_id"] = t["case_id"].astype(str)
    t = t[t["case_id"].isin(fold.index.keys())].reset_index(drop=True)
    ci = t["case_id"].map(fold.index).to_numpy()
    bounds = month_bounds(t["date_decision"])
    ev = Events(fold)
    pdpd = {vocab.id(f"pdpd={b}"): j for j, b in enumerate(BUCKETS)}
    a, b = fold.client_events[ci], fold.client_events[ci + 1]
    n_hist = np.zeros(len(t), dtype=np.int64)
    cur, cur_time = np.full(len(t), -1), np.zeros(len(t), dtype=np.int64)
    obs = np.full((len(t), H), -1)
    for r in range(len(t)):
        times = ev.time[a[r] : b[r]]
        edges = np.searchsorted(times, bounds[r], side="left")
        n_hist[r] = edges[0]
        pay = np.flatnonzero(ev.is_pay[a[r] : a[r] + edges[0]])
        if len(pay):
            cur[r] = pay_bucket(fold, int(a[r] + pay[-1]), pdpd)
            cur_time[r] = times[pay[-1]]
        for j in range(H):
            inside = np.flatnonzero(ev.is_pay[a[r] + edges[j] : a[r] + edges[j + 1]])
            if len(inside):
                obs[r, j] = pay_bucket(fold, int(a[r] + edges[j] + inside[0]), pdpd)
    t["client_idx"], t["n_hist"], t["cur"], t["cur_time"] = ci, n_hist, cur, cur_time
    for j in range(H + 1):
        t[f"bound_{j}"] = bounds[:, j]
    for j in range(H):
        t[f"obs_{j}"] = obs[:, j]
    keep = (t["n_hist"] >= 2) & (t["cur"] >= 0) & (obs >= 0).any(1)
    return t[keep].reset_index(drop=True)


def obs_matrix(rows: pd.DataFrame) -> np.ndarray:
    """(rows, H) observed bucket per held-out month, -1 where missing."""
    return rows[[f"obs_{j}" for j in range(H)]].to_numpy()


# -------------------------------------------------------------------------------
# Baselines
# -------------------------------------------------------------------------------
def roll_rate(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One-month matrix M from consecutive payment months, and the direct matrices D[j] from the current bucket to month j.

    The pair (current, month 0) counts for M only when the current payment month is the month before the cut.
    Add-one smoothing on every row.
    """
    obs, cur = obs_matrix(rows), rows["cur"].to_numpy()
    m = np.ones((NB, NB))
    prev_month = rows["bound_0"].to_numpy() - 32 * 86400 <= rows["cur_time"].to_numpy()
    ok = prev_month & (obs[:, 0] >= 0)
    np.add.at(m, (cur[ok], obs[ok, 0]), 1)
    for j in range(1, H):
        ok = (obs[:, j - 1] >= 0) & (obs[:, j] >= 0)
        np.add.at(m, (obs[ok, j - 1], obs[ok, j]), 1)
    d = np.ones((H, NB, NB))
    for j in range(H):
        ok = obs[:, j] >= 0
        np.add.at(d[j], (cur[ok], obs[ok, j]), 1)
    return m / m.sum(1, keepdims=True), d / d.sum(2, keepdims=True)


def markov_paths(m: np.ndarray, cur: np.ndarray, paths: int, rng: np.random.Generator) -> np.ndarray:
    """(rows, paths, H) bucket paths sampled from the chain M, starting at the current bucket."""
    cdf = np.cumsum(m, 1)
    s = np.repeat(cur[:, None], paths, 1)
    out = np.zeros((len(cur), paths, H), dtype=np.int64)
    for j in range(H):
        u = rng.random(s.shape)
        s = np.minimum((u[..., None] > cdf[s]).sum(-1), NB - 1)
        out[:, :, j] = s
    return out


# -------------------------------------------------------------------------------
# Generators
# -------------------------------------------------------------------------------
def load_generator(arm: str, ckpt: Path, d_run: Path | None, seed: int, vocab: Vocab) -> tuple[Any, dict[str, float]]:
    """The generator: EventGPT zero-shot, or EventGPT-D rebuilt from its rlcr run folder, checked against the run."""
    model = load_checkpoint(str(ckpt)).eval()
    if arm == "eventgpt":
        return model, {}
    assert d_run is not None, "EventGPT-D needs --d-run"
    g = Grammar(vocab)
    z = np.load(d_run / "rows.npz")
    tr = {k[3:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("tr_")}
    te = {k[3:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("te_")}
    tr_idx, _, rng = split(len(tr["ctx"]), seed)
    with torch.no_grad():
        install_jev_head(model, g, tr, tr_idx, rng)
    state = torch.load(d_run / "jev_ce.pt")
    model.load_state_dict(state["tok"], strict=False)
    ti = torch.from_numpy(tr_idx)
    fisher_intercept(model, g, tr["ctx"][ti], tr["truth"][ti])
    ev = evaluate(model, None, g, te["ctx"], te["dt"], te["truth"])
    got = metrics(ev["c"], ev["q"])
    want = json.loads((d_run / "results.json").read_text())["results"]["jev_ce"]
    check = {"accuracy": got["accuracy"], "accuracy_run": want["accuracy"], "ece": got["ece"], "ece_run": want["ece"]}
    print(f"  EventGPT-D rebuilt: {json.dumps({k: round(v, 4) for k, v in check.items()})}", flush=True)
    return model.eval(), check


@torch.no_grad()
def rollout(
    model: Any,
    vocab: Vocab,
    fold: Fold,
    rows: pd.DataFrame,
    paths: int,
    block: int,
    seqs_per_batch: int,
    dev: torch.device,
    seed: int,
    cache: Path,
) -> np.ndarray:
    """(rows, paths, H) sampled pdpd bucket per held-out month, resumable from `cache`."""
    v = vocab
    gen = torch.Generator(device="cpu").manual_seed(seed)
    allowed = []
    for f in FIELDS:
        a = torch.zeros(v.size, dtype=torch.bool)
        a[[i for s, i in v.structured.items() if s.startswith(f"{f}=")]] = True
        allowed.append(a.to(dev))
    bucket_of = torch.full((v.size,), -1, dtype=torch.long)
    for j, b in enumerate(BUCKETS):
        bucket_of[v.id(f"pdpd={b}")] = j
    evt, name = v.id("[EVT]"), v.id("name=cb_pay")
    aligned = bool(getattr(model.cfg, "aligned", False))
    ci, n_hist = rows["client_idx"].to_numpy(), rows["n_hist"].to_numpy()
    last = fold.event_time[fold.client_events[ci] + n_hist - 1]
    bounds = rows[[f"bound_{j}" for j in range(H + 1)]].to_numpy()
    gaps = np.concatenate([(bounds[:, :1] - last[:, None]), np.diff(bounds[:, : H + 1], axis=1)[:, : H - 1]], 1)
    dts = v.dt_tokens(np.maximum(gaps, 0).astype(np.float64).ravel()).reshape(len(rows), H)
    prefixes = [
        history_tokens(fold, int(c), int(n), block - EV_LEN * H, aligned=aligned).astype(np.int64) for c, n in zip(ci, n_hist, strict=True)
    ]

    out = np.full((len(rows), paths, H), -1, dtype=np.int64)
    if cache.exists():
        z = np.load(cache)
        if z["out"].shape == out.shape:
            out = z["out"]
            print(f"    resumed {int((out[:, 0, 0] >= 0).sum()):,} rows from {cache}", flush=True)
    order = np.argsort([len(p) for p in prefixes])
    todo = [i for i in order if out[i, 0, 0] < 0]
    # MPS keeps a graph and its buffers for every input shape, so batches come in few shapes: the window is rounded up
    # to 256 tokens, each width gets a fixed number of cases from a token budget (seqs_per_batch sequences of 512
    # tokens), and a short last batch is padded with copies of its first case, whose samples are dropped
    budget = seqs_per_batch * 512
    wb = {i: -(-(len(prefixes[i]) + EV_LEN * H) // 256) * 256 for i in todo}
    batches = []
    for w_ in sorted(set(wb.values())):
        group = [i for i in todo if wb[i] == w_]
        per = max(1, budget // (paths * w_))
        batches += [(np.array(group[s : s + per]), per, w_) for s in range(0, len(group), per)]
    t0, done = time.time(), 0
    for b_i, (real, per, width) in enumerate(batches):
        idx = np.r_[real, np.repeat(real[:1], per - len(real))]
        n_seq = len(idx) * paths
        base = [prefixes[i] for i in idx]
        lengths = np.repeat([len(p) for p in base], paths)
        x = np.zeros((n_seq, width), dtype=np.int64)
        for r, p in enumerate(base):
            x[r * paths : (r + 1) * paths, : len(p)] = p
        xt = torch.from_numpy(x).to(dev)
        lt = torch.from_numpy(lengths).to(dev)
        rr = torch.arange(n_seq, device=dev)
        dt_t = torch.from_numpy(np.repeat(dts[idx], paths, 0)).to(dev)
        for j in range(H):
            h, event, _, _, _ = model.stages(xt, xt != 0)
            ctx = h[rr, event[rr, lt - 1]]
            e = torch.zeros(n_seq, EV_LEN, dtype=torch.long, device=dev)
            e[:, 0], e[:, 1], e[:, 2] = evt, name, dt_t[:, j]
            for f_i in range(len(FIELDS)):
                lp = masked_logp(model, decode(model, ctx, e[:, : 3 + f_i])[:, 2 + f_i], allowed[f_i].expand(n_seq, -1))
                e[:, 3 + f_i] = torch.multinomial(lp.exp().cpu(), 1, generator=gen).squeeze(1).to(dev)
            out[real, :, j] = bucket_of[e[:, 5].cpu()].view(len(idx), paths).numpy()[: len(real)]
            pos = lt.unsqueeze(1) + torch.arange(EV_LEN, device=dev).unsqueeze(0)
            xt[rr.unsqueeze(1), pos] = e
            lt = lt + EV_LEN
            del h, event, ctx, e
            if dev.type == "mps":
                torch.mps.empty_cache()
        done += len(real)
        if dev.type == "mps" and torch.mps.driver_allocated_memory() > 34 * 2**30:
            # the MPS driver's memory grows with every new shape and empty_cache does not return it; save and stop,
            # and a rerun resumes from the cache
            np.savez(cache, out=out)
            print(f"    {done:,}/{len(todo):,} rows, MPS driver memory over 34 GiB: saved, exiting for a restart", flush=True)
            raise SystemExit(75)
        if b_i % 10 == 0:
            np.savez(cache, out=out)
            print(f"    {done:,}/{len(todo):,} rows ({time.time() - t0:.0f}s)", flush=True)
    np.savez(cache, out=out)
    return out


# -------------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------------
def shares(sims: np.ndarray) -> np.ndarray:
    """(rows, H, NB) share of paths in each bucket per month."""
    return np.stack([(sims == b).mean(1) for b in range(NB)], -1)


def month_metrics(obs: np.ndarray, p: np.ndarray, smooth: int | None) -> dict[str, Any]:
    """Per held-out month: bucket shares, log loss, Brier, and calibration of P(late) and P(30+ days); rows with a month observed."""
    out = []
    for j in range(H):
        ok = obs[:, j] >= 0
        y, pj = obs[ok, j], p[ok, j]
        pl = (pj * smooth + 0.5) / (smooth + 0.5 * NB) if smooth else pj
        onehot = np.eye(NB)[y]
        row: dict[str, Any] = {
            "rows": int(ok.sum()),
            "observed_share": (onehot.mean(0)).round(4).tolist(),
            "predicted_share": pj.mean(0).round(4).tolist(),
            "log_loss": float(-np.mean(np.log(np.clip(pl[np.arange(len(y)), y], 1e-12, None)))),
            "brier": float(np.mean(((pj - onehot) ** 2).sum(1))),
        }
        for name, sl in (("late", LATE), ("dpd30", DPD30)):
            c, q = onehot[:, sl].sum(1), pj[:, sl].sum(1)
            s = calibration.summary(c, np.clip(q, 0, 1))
            if smooth:  # a share of paths can be exactly 0 or 1, whose logit breaks the slope fit; shrink it by half a path
                s["cal_slope"] = calibration.calibration_slope(c, (q * smooth + 0.5) / (smooth + 1))[1]
            row[name] = {k: s[k] for k in ("o_e", "ece", "cal_slope", "gini_gap")} | {"rate": float(c.mean()), "mean_q": float(q.mean())}
            if 0 < c.sum() < len(c):
                row[name]["auc"] = auc(c.astype(np.int64), q)[0]
        out.append(row)
    return {"months": out}


def path_metrics(obs: np.ndarray, sims: np.ndarray) -> dict[str, Any]:
    """Late, or 30+ days, in any observed month of the six: simulated share of paths against the outcome."""
    seen = obs >= 0
    out = {}
    for name, lo in (("ever_late", 1), ("ever_dpd30", 3)):
        c = (((obs >= lo) & (obs < 7)) & seen).any(1).astype(np.float64)
        hit = ((sims >= lo) & (sims < 7)) & seen[:, None, :]
        q = hit.any(2).mean(1)
        s = calibration.summary(c, q)
        p_ = sims.shape[1]  # a share of paths can be exactly 0 or 1, whose logit breaks the slope fit; shrink it by half a path
        s["cal_slope"] = calibration.calibration_slope(c, (q * p_ + 0.5) / (p_ + 1))[1]
        out[name] = {k: s[k] for k in ("o_e", "ece", "cal_slope", "gini_gap", "brier")} | {
            "rate": float(c.mean()),
            "mean_q": float(q.mean()),
        }
        if 0 < c.sum() < len(c):
            out[name]["auc"] = auc(c.astype(np.int64), q)[0]
    return out


def main() -> None:
    """Roll each generator forward six months on the test fold and compare with the roll-rate baselines and the observed paths."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--d-run", type=Path, default=None, help="rlcr run folder with rows.npz, jev_ce.pt and results.json")
    p.add_argument("--d-seed", type=int, default=0, help="the seed of that run, for its validation split and std(y) rows")
    p.add_argument("--arms", nargs="+", default=["eventgpt", "eventgpt-d"], choices=["eventgpt", "eventgpt-d"])
    p.add_argument("--paths", type=int, default=64)
    p.add_argument("--n-uniform", type=int, default=2000)
    p.add_argument("--n-arrears", type=int, default=1000, help="extra test rows in arrears now (b1 or worse), for the per-bucket view")
    p.add_argument("--train-cases", type=int, default=50_000, help="cases read per training fold for the roll-rate matrices")
    p.add_argument("--block", type=int, default=2048)
    p.add_argument("--seqs", type=int, default=128, help="sequences per batch (cases times paths)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    t0 = time.time()
    dev = device()
    vocab = Vocab.load(OUT / "vocab")
    test_fold = int((OUT / "test_fold.txt").read_text())
    a.out.mkdir(parents=True, exist_ok=True)

    train = []
    fold_test, test = None, pd.DataFrame()
    for k in range(N_FOLDS):
        fold = Fold(k, vocab)
        if k == test_fold:
            rows = horizon_rows(fold, vocab, k, None, a.seed)
            rng = np.random.default_rng(a.seed)
            uniform = np.zeros(len(rows), dtype=bool)
            uniform[rng.choice(len(rows), min(a.n_uniform, len(rows)), replace=False)] = True
            arrears_all = np.flatnonzero((rows["cur"].to_numpy() >= 1) & (rows["cur"].to_numpy() < 7) & ~uniform)
            arrears = np.zeros(len(rows), dtype=bool)
            arrears[rng.choice(arrears_all, min(a.n_arrears, len(arrears_all)), replace=False)] = True
            test = rows[uniform | arrears].copy()
            test["uniform"] = uniform[uniform | arrears]
            test = test.reset_index(drop=True)
            fold_test = fold
            print(f"  fold {k}: {len(rows):,} cases with a payment month before the cut, {len(test):,} test rows", flush=True)
        else:
            rows = horizon_rows(fold, vocab, k, a.train_cases, a.seed + k)
            train.append(rows)
            print(f"  fold {k}: {len(rows):,} training rows ({time.time() - t0:.0f}s)", flush=True)
            del fold
    assert fold_test is not None
    m, d = roll_rate(pd.concat(train, ignore_index=True))
    obs, cur = obs_matrix(test), test["cur"].to_numpy()
    uni = test["uniform"].to_numpy()
    print(f"observed months per uniform row: {(obs[uni] >= 0).sum(1).mean():.2f} of {H} ({time.time() - t0:.0f}s)", flush=True)

    probs: dict[str, np.ndarray] = {
        "markov": np.stack([np.linalg.matrix_power(m, j + 1)[cur] for j in range(H)], 1),
        "direct": np.stack([d[j][cur] for j in range(H)], 1),
    }
    sims: dict[str, np.ndarray] = {"markov": markov_paths(m, cur, a.paths, np.random.default_rng(a.seed))}
    checks = {}
    for arm in a.arms:
        model, checks[arm] = load_generator(arm, a.ckpt, a.d_run, a.d_seed, vocab)
        model = model.to(dev).eval()
        print(f"  {arm}: rolling {len(test):,} rows x {a.paths} paths ({time.time() - t0:.0f}s)", flush=True)
        sims[arm] = rollout(model, vocab, fold_test, test, a.paths, a.block, a.seqs, dev, a.seed, a.out / f"cache_{arm}.npz")
        probs[arm] = shares(sims[arm])
        del model

    results: dict[str, Any] = {
        "ckpt": str(a.ckpt),
        "d_run": str(a.d_run),
        "paths": a.paths,
        "checks": checks,
        "population": {},
        "paths_events": {},
        "by_bucket": {},
    }
    for name, pr in probs.items():
        results["population"][name] = month_metrics(obs[uni], pr[uni], a.paths if name in sims and name != "markov" else None)
    for name, s in sims.items():
        results["paths_events"][name] = path_metrics(obs[uni], s[uni])
    for b in range(1, 7):
        sel = cur == b
        if sel.sum() >= 50:
            results["by_bucket"][BUCKETS[b]] = {"rows": int(sel.sum())} | {name: path_metrics(obs[sel], s[sel]) for name, s in sims.items()}
    results["roll_rate_matrix"] = m.round(4).tolist()
    results["elapsed_s"] = round(time.time() - t0)
    (a.out / "results.json").write_text(json.dumps(results, indent=1))
    np.savez(a.out / "sims.npz", obs=obs, cur=cur, uniform=uni, **{f"sim_{k}": v for k, v in sims.items()})
    test.drop(columns=["client_idx"]).to_parquet(a.out / "rows.parquet")

    pd.set_option("display.width", 220)
    for target in ("late", "dpd30"):
        tab = {
            name: {f"m{j + 1}": f"{r['months'][j][target]['mean_q']:.3f}" for j in range(H)} for name, r in results["population"].items()
        }
        tab["observed"] = {f"m{j + 1}": f"{results['population']['markov']['months'][j][target]['rate']:.3f}" for j in range(H)}
        print(f"\nP({target}) by held-out month, mean over uniform rows")
        print(pd.DataFrame(tab).T.to_string())
        print(f"AUC of P({target})")
        print(
            pd.DataFrame(
                {
                    n: {f"m{j + 1}": round(r["months"][j][target].get("auc", np.nan), 3) for j in range(H)}
                    for n, r in results["population"].items()
                }
            ).T.to_string()
        )
    print("\nlog loss over buckets")
    print(
        pd.DataFrame(
            {n: {f"m{j + 1}": round(r["months"][j]["log_loss"], 3) for j in range(H)} for n, r in results["population"].items()}
        ).T.to_string()
    )
    print("\npath events over the observed months")
    print(pd.DataFrame({f"{n}_{e}": v for n, r in results["paths_events"].items() for e, v in r.items()}).T.round(3).to_string())
    print(f"written to {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
