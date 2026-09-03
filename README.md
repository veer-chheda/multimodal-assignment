# Real-Time Multimodal Emotion Prototype (MELD)

A character-robot prototype: each conversational **utterance** is turned into a **structured MELD
emotion tag** plus a **short grounded response**, one turn at a time, within a per-utterance latency
budget. Primary track (required): **Text + Vision**. Standout extension attempted: **tri-modal
(Text + Vision + Audio)** from the same checkpoint, plus an **RL component** with measured value.

## Real-time definition and assumptions

Facial expression and vocal tone are naturally scoped to an **utterance** (~2-5s of speech), not to a
single video frame. So "real-time" here means **per-utterance streaming**: as each conversational turn
arrives — simulated by iterating a MELD dialogue in speaking order, or from a short live webcam+mic
capture (`src/stream_demo.run_live`, local use only) — the system produces a structured tag and a
response before the next turn, targeting on the order of ~2 seconds per utterance on a T4 GPU. This is
a deliberate choice against 30fps frame-level video analysis, which doesn't match how expression/tone
carry meaning in a spoken turn.

## Architecture

```
                text ──► TextEncoder (MiniLM, 22.7M)  ──┐
utterance ──►  frames ──► VisionEncoder (ViT-FER, 85.8M) ─┼─► concat ──► RLGatePolicy ──► FusionHead ──► emotion + confidence
              waveform ─► AudioEncoder (Wav2Vec2-SER, 94.6M)┘         (REINFORCE gate)   (trained MLP)         │
                                                                                                                 ▼
                                                            emotion + confidence + modality cues ──► Responder (Qwen2.5-Instruct)
                                                                                                                 │
                                                                                                                 ▼
                                                                                                    grounded response text
```

- **Encoders** (`src/encoders.py`) are pretrained and **frozen** — no fine-tuning. Each produces a
  fixed-dim embedding (not classification logits), so the fusion head learns its own mapping onto MELD's
  7 emotion classes instead of relying on a lossy remap from each encoder's own label set.
- **Fusion head** (`src/fusion.py: FusionHead`) is the only classifier trained from scratch: a small MLP
  over the concatenated embeddings → 7-way MELD softmax. This produces the required **structured
  state/tags**.
- **Missing-modality masking**: a modality that isn't available for a given run is a zero vector. One
  trained checkpoint therefore serves both the required core (Text+Vision) demo and the tri-modal
  extension — see the notebook's ablation section for evidence this actually works, not just in theory.
- **RL-gated fusion** (`src/fusion.py: RLGatePolicy`) — the RL component. A tiny policy network learns a
  per-utterance, per-modality soft trust gate via REINFORCE, reward = whether the resulting
  classification is correct. Compared against the static (equal-weight) fusion baseline on held-out MELD
  data — see "Evidence" below.
- **Responder** (`src/responder.py`) — a small instruction-tuned LLM (Qwen2.5-Instruct), **prompted, not
  fine-tuned**, conditioned on the transcript, the predicted emotion + confidence, and short modality
  cues. Supports token-by-token streaming (`Responder.stream`) for a live UI.

### Why RL on the fusion gate, not on the response LLM

RL fine-tuning of the LLM's response generation (e.g. PPO with an emotion-alignment reward) was
considered but rejected for this timebox: it needs a reward model, is much more expensive to train and
debug on a single T4, and its "value" is qualitative and hard to measure cleanly. The fusion-gate policy
is a tiny network with an unambiguous, gold-labeled reward signal (MELD's emotion labels) and produces a
directly comparable accuracy/F1 delta against a clear baseline — a focused, measurable RL application
rather than an ambitious, hard-to-verify one.

## Parameter budget (constraint: ≤ 6B total, all required components counted)

| component | small config | large config |
|---|---|---|
| Text encoder (MiniLM-L6-v2) | 22.7M | 22.7M |
| Vision encoder (ViT-face-expression) | 85.8M | 85.8M |
| Audio encoder (Wav2Vec2-base-SER) | 94.6M | 94.6M |
| Fusion head (trained) | <1M | <1M |
| RL gate policy (trained) | <0.1M | <0.1M |
| Response LLM | Qwen2.5-1.5B-Instruct (1.54B) | Qwen2.5-3B-Instruct (3.09B) |
| **Total** | **≈1.74B** | **≈3.29B** |

Both configs stay well clear of the 6B cap by design — the "large" config deliberately does not push all
the way to 6B (see trade-off note above). `src/param_budget.py` verifies this two ways:

```bash
python -m src.param_budget approx --config configs/small.yaml   # from config, no downloads
python -m src.param_budget exact  --config configs/large.yaml   # loads real weights, exact numel()
```

Quantization is not used to reduce the reported count (the constraint is on parameter count, not
storage precision) — both configs run in fp16/fp32 as loaded from the Hub.

## Setup

**Primary path — Kaggle** (no local GPU required): open `notebooks/kaggle_train_and_demo.ipynb`,
attach this repo (as a Kaggle Dataset or Utility Script) and the
[zaber666/meld-dataset](https://www.kaggle.com/datasets/zaber666/meld-dataset) Kaggle dataset, set
accelerator to GPU (T4 x1 or x2), and run top to bottom. It trains the fusion head + RL gate, reports
static-vs-RL-gated accuracy/F1, reports a Text+Vision-only vs tri-modal ablation, runs the streaming
demo over one real MELD conversation, and saves the (tiny, <1MB) trained checkpoints to `artifacts/`.

**Local fallback** (CPU, no MELD video needed — proves the repo runs outside Kaggle):
```bash
pip install -r requirements.txt
python demo.py --text "I can't believe you did that!" --image path/to/a/face.jpg
python demo.py --text "I can't believe you did that!"   # text-only, no image
```
Pass `--fusion-ckpt artifacts/fusion_small.pt --gate-ckpt artifacts/gate_small.pt` once you've copied
those down from a Kaggle run; without them the fusion head is randomly initialized and the run only
proves the pipeline is wired correctly end-to-end, not real classification accuracy (the CLI prints an
explicit warning when this is the case).

## Data

[MELD](https://github.com/declare-lab/MELD) (declare-lab) — conversational text, per-utterance video
(with embedded audio), and 7-way emotion + 3-way sentiment labels, from *Friends*. This project uses the
[zaber666/meld-dataset](https://www.kaggle.com/datasets/zaber666/meld-dataset) Kaggle mirror, whose
layout nests each split's CSV and videos under a same-named subfolder (`train/train_sent_emo.csv`,
`train/train_splits/*.mp4`, and similarly for `dev`/`test`). `src/data_meld.py` reads that layout and
extracts keyframes + audio per utterance clip via `ffmpeg`. MELD is for research/non-commercial use;
cite the MELD paper (Poria et al., ACL 2019) if reusing this pipeline.

## Evidence (fill in exact numbers from your notebook run)

- **Static vs RL-gated fusion** (test split, N=200 subsample): accuracy/F1 delta — see notebook §5 output.
- **Text+Vision-only vs full tri-modal** ablation (same checkpoint): accuracy/F1 — see notebook §6 output.
- **Hardware / observed resource requirements**: Kaggle T4 (16GB VRAM), peak GPU memory and average
  per-utterance latency — see notebook §8-9 output. Only a single T4 was needed at this parameter scale
  (dual T4 wasn't required for either config; noted here rather than adding unused model-parallel
  complexity).

## What was completed

- Core Text+Vision track: frozen face-expression encoder + text encoder → trained fusion head →
  structured MELD tag → grounded response text from a small instruction-tuned LLM, demonstrated
  end-to-end on real MELD utterances via the Kaggle notebook.
- Tri-modal extension: audio branch (frozen speech-emotion encoder) added to the same fusion
  architecture via modality masking, with an ablation showing it changes predictions (notebook §6).
- RL component: REINFORCE-trained modality-gating policy, with a measured accuracy/F1 delta against a
  static-fusion baseline (notebook §5) — not just implemented, but evaluated.
- Streaming per-utterance demo over a real MELD conversation, with per-utterance latency and peak GPU
  memory reported.
- Local CPU fallback path (`demo.py`) so the repo is runnable without Kaggle, for review.
- Exact and approximate parameter-budget accounting, verified under the 6B cap for both model sizes.

## What was intentionally left out

- **No fine-tuning of any pretrained encoder or the response LLM** — all are used frozen/prompted. This
  keeps the parameter count exactly equal to the off-the-shelf checkpoint sizes (easy to audit against
  the 6B cap) and fit the timebox; the fusion head and RL gate are the only trained parameters.
- **No ASR** — the live webcam+mic path (`src/stream_demo.run_live`) asks the user to type what was
  said rather than transcribing it, since speech-to-text is orthogonal to the multimodal-emotion task
  this challenge is about.
- **No RL fine-tuning of the response LLM** (e.g. PPO with a learned reward model) — see "Why RL on the
  fusion gate, not on the response LLM" above.
- **Full MELD is not used by default** — the notebook subsamples (`N_TRAIN`/`N_TEST`) for a fast,
  correct prototype within the 2-3 day timebox; the pipeline itself has no hard limit and the sample
  size is a single config constant to raise.
- **No face detection/cropping** — vision keyframes are full video frames (MELD clips are already
  fairly tightly framed on speakers), not detected/cropped faces. A dedicated face detector would
  likely improve the vision encoder's signal but adds another component to the parameter budget for
  uncertain gain at this scale.

## Limitations

- The fusion head and RL gate are trained on a MELD subsample; both are small enough to train fast, but
  reported accuracy/F1 should be read as prototype-scale evidence, not a claim of SOTA MELD performance.
- The Ekman-7 label sets of the pretrained FER/SER encoders don't perfectly align with MELD's 7 classes
  in general (this is exactly why embeddings, not their logits, feed the fusion head — see Architecture)
  — but the encoders' own training distributions still differ from MELD/*Friends*, which likely caps
  how much signal they contribute versus an encoder fine-tuned on MELD directly.
- `demo.py`'s local fallback has no MELD video, so it only exercises the Text(+Vision, from a static
  image) path — audio and the streaming/dialogue behavior are only demonstrated inside the Kaggle
  notebook.
- Latency figures are single-utterance, single-GPU measurements from the notebook run, not a
  load-tested serving benchmark.

## External / generated components — attribution

- `sentence-transformers/all-MiniLM-L6-v2` — text embeddings (Reimers & Gurevych).
- `trpakov/vit-face-expression` — facial-expression ViT (Hugging Face Hub, community checkpoint,
  Ekman-7 labels).
- `superb/wav2vec2-base-superb-er` — speech-emotion Wav2Vec2 (SUPERB benchmark checkpoint).
- `Qwen/Qwen2.5-1.5B-Instruct` / `Qwen/Qwen2.5-3B-Instruct` (Alibaba Qwen team) — response generation.
- MELD dataset (declare-lab) — data foundation, see "Data" above.
- AI-assisted development (Claude Code) was used to help write this repository's code and notebook; the
  architecture decisions, trade-offs, and evaluation above are understood and owned by the author, and
  every component is swappable/explainable from `src/`.
