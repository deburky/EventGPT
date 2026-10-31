"""Calibration of the monthly probabilities from sampled months: does a predicted 2% come true 2% of the time?

Reads rollout_scores.parquet (eventgpt.evaluation.rollout --save-scores) and, on the uniformly sampled rows only,
compares each event target's monthly probability (the rollout compensator, after the rule for earlier
occurrences) and the churn probability with the labels. Rows are grouped into BINS bins of equal count by the
predicted probability; each bin gives the mean prediction and the observed rate. The summary gives the ratio of
predicted to observed positives, the Brier score and the Brier score of the base rate, the label's prevalence
predicted for everyone. The one-step zero-shot score is a ranking score, the probability of the next event and not
of the month, so its ratio is printed only to show the difference.

Example: `uv run python -m eventgpt.evaluation.calibration data/runs/roll-alq-s42/rollout_scores.parquet`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eventgpt.evaluation.probe import UNDEFINED
from eventgpt.evaluation.zero_shot import FIRST

BINS = 10


def reliability(y: np.ndarray, p: np.ndarray, bins: int = BINS) -> pd.DataFrame:
    """Mean prediction, observed rate and row count in bins of equal count by prediction."""
    order = np.argsort(p, kind="stable")
    parts = np.array_split(order, bins)
    return pd.DataFrame(
        {
            "bin": range(bins),
            "predicted": [p[i].mean() for i in parts],
            "observed": [y[i].mean() for i in parts],
            "rows": [len(i) for i in parts],
        }
    )


def main() -> None:
    """Reliability bins and summary per label."""
    a = argparse.ArgumentParser()
    a.add_argument("scores", type=Path)
    a.add_argument("--out", type=Path, default=None, help="csv prefix; default calibration beside the scores")
    args = a.parse_args()
    s = pd.read_parquet(args.scores)
    s = s[s["in_rand"]]
    summary, bins = [], []
    targets = [(lab, s[f"rollout_{lab}"] * s[f"unseen_{lab}"], s[lab]) for lab in FIRST]
    targets.append(("churn", s["rollout_churn"], s["churn"]))
    for lab, p, y in targets:
        ok = (y != UNDEFINED).to_numpy()
        pk, yk = p.to_numpy()[ok].astype(float), y.to_numpy()[ok].astype(float)
        line = {
            "label": lab,
            "rows": int(ok.sum()),
            "positives": int(yk.sum()),
            "predicted_positives": float(pk.sum()),
            "ratio": float(pk.sum() / max(yk.sum(), 1)),
            "brier": float(((pk - yk) ** 2).mean()),
            "brier_base_rate": float(((yk.mean() - yk) ** 2).mean()),
        }
        if lab != "churn":
            one_step = (s[f"zero_shot_{lab}"] * s[f"unseen_{lab}"]).to_numpy()[ok]
            line["one_step_ratio"] = float(one_step.sum() / max(yk.sum(), 1))
        summary.append(line)
        bins.append(reliability(yk, pk).assign(label=lab))
    summary_df = pd.DataFrame(summary)
    print(summary_df.round(4).to_string(index=False))
    prefix = args.out or args.scores.with_name("calibration")
    summary_df.to_csv(f"{prefix}_summary.csv", index=False)
    pd.concat(bins).to_csv(f"{prefix}_bins.csv", index=False)
    print(f"written to {prefix}_summary.csv and {prefix}_bins.csv")


if __name__ == "__main__":
    main()
