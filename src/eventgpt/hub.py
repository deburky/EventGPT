"""Released checkpoints: export a training checkpoint for the Hugging Face Hub and load one back.

A released checkpoint is a folder with the model weights (model.safetensors), its configuration (config.json)
and the vocabulary it was trained on (vocab.json, token ids and amount-bin edges). The vocabulary is part of
the model: a history must be tokenized with the same table before the model can read it.

Example: `uv run egpt export data/runs/hier-alqp-s42/hier-alqp-s42.pt --vocab /path/to/tokens/vocab --out hub/aligned-profile-s42`,
then `from eventgpt.hub import from_pretrained; model, vocab = from_pretrained("deburky/eventgpt", "aligned-profile-s42")`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_model, save_model
from torch import nn

from eventgpt.data.vocab import Vocab
from eventgpt.models.flat import GPT, GPTConfig
from eventgpt.models.hierarchy import EventGPT, EventGPTConfig

REPO = "deburky/eventgpt"


def build(meta: dict) -> nn.Module:
    """An untrained model of the architecture and size that meta describes."""
    if meta["arch"] == "eventgpt":
        return EventGPT(EventGPTConfig(**meta["cfg"]))
    return GPT(GPTConfig(**meta["cfg"]))


def export(ckpt: Path, vocab_dir: Path, out: Path) -> None:
    """Write a training checkpoint as model.safetensors, config.json and vocab.json in out."""
    ck = torch.load(ckpt, map_location="cpu")
    vocab = Vocab.load(vocab_dir)
    if vocab.fingerprint() != ck["vocab_sha256"]:
        raise SystemExit(f"{vocab_dir} is not the vocabulary {ckpt.name} was trained on")
    meta = {"arch": ck["arch"], "cfg": ck["cfg"], "step": ck["step"], "vocab_sha256": ck["vocab_sha256"]}
    model = build(meta)
    model.load_state_dict(ck["model"])
    out.mkdir(parents=True, exist_ok=True)
    save_model(model, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    vocab.save(out)


def load_folder(d: Path) -> tuple[nn.Module, Vocab]:
    """The model and vocabulary of an exported folder, the model in eval mode on the CPU."""
    meta = json.loads((d / "config.json").read_text())
    vocab = Vocab.load(d)
    if vocab.fingerprint() != meta["vocab_sha256"]:
        raise ValueError(f"vocab.json in {d} does not match the model")
    model = build(meta)
    load_model(model, str(d / "model.safetensors"))
    return model.eval(), vocab


def from_pretrained(repo_id: str = REPO, variant: str = "aligned-profile-s42", revision: str | None = None) -> tuple[nn.Module, Vocab]:
    """Download one released variant (a subfolder of the repository) and load it."""
    root = snapshot_download(repo_id, revision=revision, allow_patterns=[f"{variant}/*"])
    return load_folder(Path(root) / variant)


def main() -> None:
    """Export a checkpoint into a folder ready for upload."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--vocab", type=Path, required=True, help="the vocab directory of the token streams the model was trained on")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    export(a.ckpt, a.vocab, a.out)
    model, _ = load_folder(a.out)
    n = sum(t.numel() for t in model.parameters())
    print(f"{a.ckpt.name} -> {a.out} ({n / 1e6:.2f}M parameters, reloaded)")


if __name__ == "__main__":
    main()
