"""Next-token pretraining on packed client token streams. Runs on MPS when available, CUDA otherwise.

Example: `uv run python -m eventgpt.training.train --run mbd-hier --arch eventgpt --rope --steps 20000`
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eventgpt.data.vocab import EVT, USR, Vocab
from eventgpt.data.windows import AlignedStream, PackedStream
from eventgpt.models.flat import GPT, GPTConfig
from eventgpt.models.hierarchy import EventGPT, EventGPTConfig

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("MBD_OUT", "/Volumes/WD_BLACK/datasets/MBD-mini/tokens"))
CKPT = Path(os.environ.get("EGPT_CKPT", ROOT / "data" / "ckpt"))


def device() -> torch.device:
    """MPS if available, else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def token_class(vocab: Vocab) -> list[str]:
    """Class name per token id: the key of a key=value token, else the bare special name."""
    names = {v: k for k, v in vocab.structured.items()}
    return [names[t][1:-1].lower() if names[t].startswith("[") else names[t].split("=")[0] for t in range(vocab.size)]


@torch.no_grad()
def evaluate(
    model: Any, stream: PackedStream | AlignedStream, block: int, bsz: int, iters: int, dev: torch.device, classes: list[str]
) -> dict[str, float]:
    """Validation loss over fixed-seed batches, overall and per target class above 1% of tokens; restores train mode."""
    model.eval()
    rng = np.random.default_rng(0)
    cls = np.asarray(classes)
    losses: list[float] = []
    tot: dict[str, list[float]] = {}
    for _ in range(iters):
        x, y = stream.batch(block, bsz, rng, dev)
        logits, loss = model(x, y, None, True)
        losses.append(loss.item())
        flat_y = y.view(-1)
        nll = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), flat_y, reduction="none").float().cpu().numpy()
        hit = (logits.view(-1, logits.size(-1)).argmax(-1) == flat_y).cpu().numpy()
        yc = cls[flat_y.cpu().numpy()]
        for c in np.unique(yc):
            sel = yc == c
            tot.setdefault(f"{c}_loss", []).append(float(nll[sel].mean()))
            tot.setdefault(f"{c}_top1", []).append(float(hit[sel].mean()))
            tot.setdefault(f"{c}_share", []).append(float(sel.mean()))
    model.train()
    out = {k: float(np.mean(v)) for k, v in tot.items()}
    keep = {k.removesuffix("_share") for k, v in out.items() if k.endswith("_share") and v >= 0.01}
    rec = {k: round(v, 5) for k, v in out.items() if k.rsplit("_", 1)[0] in keep}
    return {"val_loss": float(np.mean(losses)), **rec}


def build(a: argparse.Namespace, vocab: Vocab) -> tuple[Any, Any]:
    """Model and config from the arguments."""
    if a.arch == "eventgpt":
        cfg: Any = EventGPTConfig(
            vocab_size=vocab.size,
            block_size=a.block,
            n_layer=a.n_layer,
            d_model=a.d_model,
            n_head=a.d_model // a.head_dim,
            causal=True,
            usr_id=vocab.id(USR),
            evt_id=vocab.id(EVT),
            rope=a.rope,
            n_experts=3 if a.moe else 1,  # kinds: other, trx, geo (EventGPT.set_vocab_tables)
            profile_path=a.profile_path,
            qk_norm=a.qk_norm,
            aligned=a.aligned,
        )
        model: Any = EventGPT(cfg)
        model.set_vocab_tables(vocab.structured)
        return model, cfg
    cfg = GPTConfig(
        vocab_size=vocab.size,
        block_size=a.block,
        n_layer=a.n_layer,
        d_model=a.d_model,
        n_head=a.d_model // a.head_dim,
        causal=True,
        usr_id=vocab.id(USR),
        qk_norm=a.qk_norm,
        aligned=a.aligned,
    )
    return GPT(cfg), cfg


def main() -> None:
    """Parse arguments, build the model, train, checkpoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default="local")
    p.add_argument("--arch", choices=["eventgpt", "gpt"], default="eventgpt")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=4, help="windows per forward pass")
    p.add_argument("--accum", type=int, default=4, help="forward passes per optimizer step; effective batch is batch * accum")
    p.add_argument("--block", type=int, default=2048)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--n-layer", type=int, default=6, help="history layers for eventgpt, all layers for gpt")
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--head-dim", type=int, default=32)
    p.add_argument("--warmup", type=int, default=0, help="linear warmup steps; 0 means min(200, steps // 10)")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rope", action="store_true", help="eventgpt: time-aware rotary positions over events, none inside an event")
    p.add_argument("--moe", action="store_true", help="eventgpt: one feed-forward per event kind in the event encoder, hard-routed")
    p.add_argument("--qk-norm", action="store_true", help="RMSNorm on queries and keys inside each head")
    p.add_argument("--profile-path", action="store_true", help="eventgpt: profile field embeddings added to every later event")
    p.add_argument(
        "--aligned", action="store_true", help="client-aligned windows with the gap from the client's first event (eventgpt.data.windows)"
    )
    p.add_argument(
        "--no-loss-on", type=str, default="", help="comma-separated token classes (dt,amt_bin,g4,g5) left out of the training loss"
    )
    a = p.parse_args()

    dev = device()
    torch.manual_seed(a.seed)
    vocab = Vocab.load(DATA / "vocab")
    train: PackedStream | AlignedStream
    val: PackedStream | AlignedStream
    if a.aligned:
        train, val = AlignedStream("train", vocab), AlignedStream("val", vocab)
    else:
        train, val = PackedStream("train"), PackedStream("val")
    classes = token_class(vocab)
    # targets of these classes are still read as input and still scored in evaluation, but give no gradient
    skip = {c for c in a.no_loss_on.split(",") if c}
    if skip - set(classes):
        raise SystemExit(f"unknown token classes: {sorted(skip - set(classes))}")
    no_loss = torch.tensor([c in skip for c in classes], device=dev)
    model, cfg = build(a, vocab)
    model = model.to(dev)
    use_amp = dev.type == "cuda"
    print(
        f"device={dev} arch={a.arch} rope={a.rope} moe={a.moe} qk_norm={a.qk_norm} aligned={a.aligned} no_loss_on={sorted(skip)} "
        f"vocab={vocab.size} "
        f"params={model.n_params() / 1e6:.2f}M train_tokens={len(train):,} val_tokens={len(val):,}",
        flush=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup = a.warmup if a.warmup else min(200, a.steps // 10)

    def lr_at(step: int) -> float:
        """Linear warmup, then cosine decay to 10% of the peak rate."""
        if step < warmup:
            return a.lr * step / warmup
        prog = (step - warmup) / max(1, a.steps - warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    CKPT.mkdir(parents=True, exist_ok=True)
    log_path = CKPT / f"{a.run}_log.jsonl"
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    tokens_seen = 0
    first_step = 1
    # A run name that already has a checkpoint continues from it: a spot VM can be reclaimed mid-run, and the
    # rerun picks up the weights, the optimizer moments, the step and the batch sampler where they were.
    state_path = CKPT / f"{a.run}_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location=dev, weights_only=False)
        model.load_state_dict(torch.load(CKPT / f"{a.run}.pt", map_location=dev, weights_only=False)["model"])
        opt.load_state_dict(state["opt"])
        rng.bit_generator.state = state["rng"]
        tokens_seen, first_step = state["tokens"], state["step"] + 1
        print(f"resuming {a.run} at step {first_step}", flush=True)
    model.train()
    for step in range(first_step, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            x, y = train.batch(a.block, a.batch, rng, dev)
            y = y.masked_fill(no_loss[y], -100)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(x, y, None, True)
            (loss / a.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens_seen += a.batch * a.accum * a.block
        if step % 50 == 0:
            el = time.time() - t0
            rate = tokens_seen / el
            print(f"step {step} loss {loss.item():.4f} lr {lr_at(step):.2e} tok/s {rate:,.0f} elapsed {el / 60:.1f}m", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            rec = {"step": step, "train_loss": loss.item(), "tokens": tokens_seen, "elapsed_s": round(time.time() - t0)}
            rec.update(evaluate(model, val, a.block, a.batch, 20, dev, classes))
            print(f"eval {json.dumps(rec)}", flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            torch.save(
                {"cfg": cfg.__dict__, "model": model.state_dict(), "step": step, "arch": a.arch, "vocab_sha256": vocab.fingerprint()},
                CKPT / f"{a.run}.pt",
            )
            torch.save({"opt": opt.state_dict(), "rng": rng.bit_generator.state, "step": step, "tokens": tokens_seen}, state_path)
    print("done", flush=True)


if __name__ == "__main__":
    main()
