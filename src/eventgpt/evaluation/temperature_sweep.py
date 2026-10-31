"""Choose the sampling temperature on half of the test clients, then report the chosen one on the other half.

Each temperature runs eventgpt.evaluation.rollout with --sequences on the selection half (--clients select). The
chosen temperature has the highest overlap between sampled and real months; it and temperature 1 are then run on
the reporting half (--clients report). Arguments after the checkpoint and --out-dir pass through to the rollout.

Example with R=data/runs/hier-alq-s42:
`uv run python -m eventgpt.evaluation.temperature_sweep $R/hier-alq-s42.pt --out-dir $R/temperature -- --probe-dir $R/probe-now-2k`
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

TEMPERATURES = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2)


def run(ckpt: Path, t: float, clients: str, n_random: int, out: Path, rest: list[str]) -> pd.DataFrame:
    """One rollout with sequence metrics; returns its sequence summary."""
    cmd = [sys.executable, "-m", "eventgpt.evaluation.rollout", str(ckpt), "--sequences", "--uniform-only", "--temperature", str(t)]
    cmd += ["--clients", clients, "--n-random", str(n_random), "--out", str(out), *rest]
    subprocess.run(cmd, check=True)
    return pd.read_csv(out.with_name(out.stem + "_sequences.csv"), index_col="metric")


def main() -> None:
    """Sweep on the selection half, report on the other."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-select", type=int, default=3_000, help="uniform rows per temperature on the selection half")
    p.add_argument("--n-report", type=int, default=10_000, help="uniform rows on the reporting half")
    p.add_argument("rest", nargs=argparse.REMAINDER, help="arguments for the rollout, after --")
    a = p.parse_args()
    rest = [r for r in a.rest if r != "--"]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in TEMPERATURES:
        s = run(a.ckpt, t, "select", a.n_select, a.out_dir / f"select_t{t}.csv", rest)
        rows.append({"temperature": t, **{f"{m}": s.loc[m, "model"] for m in s.index}, "repeat_overlap": s.loc["overlap", "repeat"]})
    sweep = pd.DataFrame(rows).set_index("temperature")
    print(sweep.round(4).to_string(), flush=True)
    sweep.to_csv(a.out_dir / "sweep.csv")
    best = TEMPERATURES[int(sweep["overlap"].to_numpy().argmax())]
    print(f"chosen temperature {best}", flush=True)
    for t in sorted({1.0, best}):
        run(a.ckpt, t, "report", a.n_report, a.out_dir / f"report_t{t}.csv", rest)


if __name__ == "__main__":
    main()
