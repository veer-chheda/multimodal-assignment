"""Parameter-budget accounting for the local inference path.

Two modes:
  approx  - sums the `approx_params` figures recorded in configs/*.yaml.
            No downloads, no torch required — works on any machine
            (including the local M2 with no ML stack installed) as a
            fast sanity check.
  exact   - actually instantiates every module in the active inference
            path (text/vision/audio encoders, fusion head, RL gate,
            response LLM) and sums real `numel()` counts. Requires
            torch/transformers and the model weights to download, so
            this mode is meant to be run on Kaggle.

Usage:
    python -m src.param_budget approx --config configs/small.yaml
    python -m src.param_budget exact  --config configs/large.yaml
"""
from __future__ import annotations

import argparse
import sys

import yaml

PARAM_BUDGET = 6_000_000_000


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def approx_total(cfg: dict) -> int:
    total = 0
    for _name, enc in cfg["encoders"].items():
        total += int(enc["approx_params"])
    total += int(cfg["responder"]["approx_params"])
    return total


def exact_total(cfg: dict) -> dict[str, int]:
    """Load every real module used at inference time and count numel()."""
    import torch  # noqa: F401  (import guarded here, not at module load time)

    from src.encoders import AudioEncoder, TextEncoder, VisionEncoder
    from src.fusion import FusionHead, RLGatePolicy
    from src.responder import Responder

    counts: dict[str, int] = {}

    text_enc = TextEncoder(cfg["encoders"]["text"]["checkpoint"])
    counts["text_encoder"] = sum(p.numel() for p in text_enc.model.parameters())

    vision_enc = VisionEncoder(cfg["encoders"]["vision"]["checkpoint"])
    counts["vision_encoder"] = sum(p.numel() for p in vision_enc.model.parameters())

    audio_enc = AudioEncoder(cfg["encoders"]["audio"]["checkpoint"])
    counts["audio_encoder"] = sum(p.numel() for p in audio_enc.model.parameters())

    fcfg = cfg["fusion"]
    fusion = FusionHead(
        embed_dims=fcfg["embed_dims"], hidden_dim=fcfg["hidden_dim"], num_classes=fcfg["num_classes"]
    )
    counts["fusion_head"] = sum(p.numel() for p in fusion.parameters())

    gate = RLGatePolicy(embed_dims=fcfg["embed_dims"], hidden_dim=fcfg["gate_hidden_dim"])
    counts["rl_gate_policy"] = sum(p.numel() for p in gate.parameters())

    responder = Responder(cfg["responder"]["checkpoint"])
    counts["response_llm"] = sum(p.numel() for p in responder.model.parameters())

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["approx", "exact"])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.mode == "approx":
        total = approx_total(cfg)
        print(f"[{cfg['name']}] approx total params (from config, no downloads): {total:,}")
    else:
        counts = exact_total(cfg)
        total = sum(counts.values())
        print(f"[{cfg['name']}] exact runtime param counts:")
        for k, v in counts.items():
            print(f"  {k:20s} {v:>15,}")
        print(f"  {'TOTAL':20s} {total:>15,}")

    print(f"budget: {PARAM_BUDGET:,}")
    if total > PARAM_BUDGET:
        print(f"OVER BUDGET by {total - PARAM_BUDGET:,} params", file=sys.stderr)
        sys.exit(1)
    print(f"OK — {PARAM_BUDGET - total:,} params of headroom")


if __name__ == "__main__":
    main()
