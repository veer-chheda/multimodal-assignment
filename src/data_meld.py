"""MELD CSV loading + per-utterance frame/audio extraction via ffmpeg.

The official MELD.Raw release (declare-lab) lays each split out flat under
one root:

    <meld_root>/
      train_sent_emo.csv, dev_sent_emo.csv, test_sent_emo.csv
      train_splits/dia{D}_utt{U}.mp4
      dev_splits_complete/dia{D}_utt{U}.mp4
      output_repeated_splits_test/dia{D}_utt{U}.mp4

but Kaggle mirrors sometimes re-nest each split under a same-named
subfolder (`<meld_root>/test/test_sent_emo.csv`, etc). `load_split` tries
both so a one-off mount-path quirk doesn't need a code change — it uses
whichever layout actually has the CSV for that split, and raises a clear
error listing every path it tried if neither does.

Each CSV row is one utterance: Utterance, Emotion, Sentiment, Dialogue_ID,
Utterance_ID, Speaker, ... . Emotion values are lowercased to match
src.schema.MELD_EMOTIONS.
"""
from __future__ import annotations

import csv
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLIT_VIDEO_DIRS = {
    "train": "train_splits",
    "dev": "dev_splits_complete",
    "test": "output_repeated_splits_test",
}
SPLIT_CSVS = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}


@dataclass
class Utterance:
    utterance_id: str  # f"dia{Dialogue_ID}_utt{Utterance_ID}"
    dialogue_id: int
    utt_index: int
    text: str
    emotion: str
    sentiment: str
    speaker: str
    video_path: Path


def _first_existing(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"none of these paths exist:\n  {tried}")


def load_split(meld_root: str | Path, split: str, limit: int | None = None) -> list[Utterance]:
    root = Path(meld_root)
    csv_path = _first_existing([root / SPLIT_CSVS[split], root / split / SPLIT_CSVS[split]])
    video_dir = _first_existing([root / SPLIT_VIDEO_DIRS[split], root / split / SPLIT_VIDEO_DIRS[split]])

    rows: list[Utterance] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            dia, utt = int(r["Dialogue_ID"]), int(r["Utterance_ID"])
            uid = f"dia{dia}_utt{utt}"
            rows.append(
                Utterance(
                    utterance_id=uid,
                    dialogue_id=dia,
                    utt_index=utt,
                    text=r["Utterance"].strip(),
                    emotion=r["Emotion"].strip().lower(),
                    sentiment=r["Sentiment"].strip().lower(),
                    speaker=r["Speaker"].strip(),
                    video_path=video_dir / f"{uid}.mp4",
                )
            )
            if limit and len(rows) >= limit:
                break
    return rows


def dialogue(utterances: list[Utterance], dialogue_id: int) -> list[Utterance]:
    """All utterances of one conversation, in speaking order — used to simulate
    turns 'arriving over time' for the streaming demo."""
    return sorted((u for u in utterances if u.dialogue_id == dialogue_id), key=lambda u: u.utt_index)


def _run(cmd: list[str], video_path: Path) -> subprocess.CompletedProcess:
    """subprocess.run wrapper that raises a RuntimeError carrying ffmpeg/ffprobe's actual
    stderr (and the file size, to distinguish a truncated/empty file from a bad codec) —
    a bare CalledProcessError's str() only shows the exit code, which isn't enough to
    diagnose a bad clip from the caller."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        size = video_path.stat().st_size if video_path.exists() else -1
        raise RuntimeError(
            f"`{' '.join(cmd[:1])}` failed on {video_path} ({size} bytes): {result.stderr.strip()[-500:]}"
        )
    return result


def _clip_duration_seconds(video_path: Path) -> float:
    out = _run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrapping_selectors=1:nokey=1", str(video_path)],
        video_path,
    )
    return float(out.stdout.strip())


def extract_keyframes(video_path: str | Path, n_frames: int = 3):
    """`n_frames` evenly-spaced keyframes as PIL Images, via ffmpeg."""
    from PIL import Image

    video_path = Path(video_path)
    duration = _clip_duration_seconds(video_path)
    fps = max(n_frames / duration, 0.1) if duration > 0 else 1.0

    with tempfile.TemporaryDirectory() as tmp:
        out_pattern = str(Path(tmp) / "frame_%02d.png")
        _run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps},scale=224:224",
             "-frames:v", str(n_frames), out_pattern],
            video_path,
        )
        frames = sorted(Path(tmp).glob("frame_*.png"))
        return [Image.open(fp).convert("RGB").copy() for fp in frames]


def extract_audio(video_path: str | Path, sample_rate: int = 16_000) -> np.ndarray:
    """Mono float32 waveform at `sample_rate`, via ffmpeg -> wav -> soundfile."""
    import soundfile as sf

    video_path = Path(video_path)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        _run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-ac", "1", "-ar", str(sample_rate), "-vn", str(wav_path),
            ],
            video_path,
        )
        waveform, sr = sf.read(str(wav_path), dtype="float32")
        assert sr == sample_rate
        return waveform
