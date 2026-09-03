#!/usr/bin/env python3
"""Local CLI fallback — runs the core Text(+Vision) path on CPU without Kaggle
or MELD video files, to prove the repository is runnable outside Kaggle.

    python demo.py --text "I can't believe you did that!" --image path/to/face.jpg
    python demo.py --text "I can't believe you did that!"          # text-only

Slow on CPU (expect several seconds per utterance, mostly LLM generation) and,
unless --fusion-ckpt/--gate-ckpt point at weights produced by
notebooks/kaggle_train_and_demo.ipynb, the fusion head is randomly initialized
— this run only proves the pipeline is wired correctly end-to-end, not real
classification accuracy. See README "What was left out".
"""
from __future__ import annotations

import argparse
import json

from src.stream_demo import Pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True)
    ap.add_argument("--image", help="path to a face image (jpg/png); omitted = text-only")
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--fusion-ckpt", default=None, help="artifacts/fusion_small.pt from the Kaggle notebook, if you have one")
    ap.add_argument("--gate-ckpt", default=None, help="artifacts/gate_small.pt from the Kaggle notebook, if you have one")
    ap.add_argument("--no-rl-gate", action="store_true", help="use the static equal-weight fusion instead of the RL-gated one")
    args = ap.parse_args()

    pipeline = Pipeline(args.config, device="cpu", fusion_ckpt=args.fusion_ckpt, gate_ckpt=args.gate_ckpt)
    if not pipeline.trained:
        print("[warn] no --fusion-ckpt given: fusion head is randomly initialized. "
              "This run verifies the pipeline wiring, not real classification accuracy.\n")

    frames = None
    if args.image:
        from PIL import Image

        frames = [Image.open(args.image).convert("RGB")]

    tag = pipeline.tag("cli-demo", args.text, frames=frames, waveform=None, use_rl_gate=not args.no_rl_gate)
    print(json.dumps(tag.as_dict(), indent=2))


if __name__ == "__main__":
    main()
