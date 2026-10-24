"""The egpt command: one subcommand per entry point, with that module's own flags.

Example: `uv run egpt probe data/runs/hier-alq-s42/hier-alq-s42.pt --now`, the same as
`uv run python -m eventgpt.evaluation.probe ...`; `egpt <command> --help` shows a command's flags.
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "prepare": ("eventgpt.data.mbd", "build the token streams from MBD-mini"),
    "check-tokens": ("eventgpt.data.check_tokens", "check the invariants of the token streams"),
    "train": ("eventgpt.training.train", "pretrain a flat GPT or EventGPT"),
    "finetune": ("eventgpt.training.finetune", "fine-tune a checkpoint on the probe's labels"),
    "probe": ("eventgpt.evaluation.probe", "embed the probe rows and fit linear heads"),
    "zero-shot": ("eventgpt.evaluation.zero_shot", "score event targets from next-token probabilities"),
    "profile-readout": ("eventgpt.evaluation.profile_readout", "heads on the history state plus the profile tokens"),
    "auc-se": ("eventgpt.evaluation.auc_se", "AUC standard errors and paired differences, clustered by client"),
    "catboost": ("eventgpt.baselines.catboost", "CatBoost on hand-built features"),
    "token-counts": ("eventgpt.baselines.token_counts", "CatBoost on token counts over time windows"),
    "emb-boost": ("eventgpt.baselines.emb_boost", "CatBoost on a frozen embedding"),
    "label-curve": ("eventgpt.baselines.label_curve", "CatBoost on n labeled rows beside the zero-shot score"),
}


def main() -> None:
    """Run the named command's entry point with the remaining arguments."""
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        width = max(map(len, COMMANDS))
        print("usage: egpt <command> [args]\n\ncommands:")
        for name, (_, about) in COMMANDS.items():
            print(f"  {name.ljust(width)}  {about}")
        sys.exit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 2)
    name = sys.argv[1]
    sys.argv = [f"egpt {name}", *sys.argv[2:]]
    importlib.import_module(COMMANDS[name][0]).main()


if __name__ == "__main__":
    main()
