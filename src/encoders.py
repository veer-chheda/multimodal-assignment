"""Pretrained, frozen modality encoders.

Each encoder exposes a fixed-dimensional embedding so the fusion head
(src/fusion.py) can concatenate them regardless of which modalities are
present for a given utterance. All three are used unmodified from the
Hub — no fine-tuning — so their parameter counts are exactly what
`src/param_budget.py` reports.
"""
from __future__ import annotations

import numpy as np


class TextEncoder:
    """sentence-transformers/all-MiniLM-L6-v2 -> 384-dim sentence embedding."""

    embed_dim = 384

    def __init__(self, checkpoint: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(checkpoint, device=device)
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        return self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]


class VisionEncoder:
    """trpakov/vit-face-expression (ViT-base, Ekman-7 facial expression) -> 768-dim CLS embedding.

    We take the penultimate (last hidden state, CLS token) embedding rather
    than the 7-way classification logits, so the fusion head can learn its
    own mapping onto the MELD label set instead of relying on a lossy
    FER-label -> MELD-label remap.
    """

    embed_dim = 768

    def __init__(self, checkpoint: str = "trpakov/vit-face-expression", device: str = "cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForImageClassification.from_pretrained(checkpoint, output_hidden_states=True)
        self.model.to(device).eval()
        self._torch = torch

    def encode(self, frames: list) -> np.ndarray:
        """frames: list of PIL.Image keyframes sampled from one utterance clip.
        Returns the mean-pooled CLS embedding across frames (1 embedding per utterance)."""
        torch = self._torch
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        cls_embeds = out.hidden_states[-1][:, 0, :]  # (num_frames, 768)
        return cls_embeds.mean(dim=0).cpu().numpy()


class AudioEncoder:
    """superb/wav2vec2-base-superb-er (Wav2Vec2-base, speech emotion) -> 768-dim pooled embedding."""

    embed_dim = 768
    sample_rate = 16_000

    def __init__(self, checkpoint: str = "superb/wav2vec2-base-superb-er", device: str = "cpu"):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
        self.model = AutoModelForAudioClassification.from_pretrained(checkpoint, output_hidden_states=True)
        self.model.to(device).eval()
        self._torch = torch

    def encode(self, waveform: np.ndarray) -> np.ndarray:
        """waveform: 1-D float32 array sampled at self.sample_rate."""
        torch = self._torch
        inputs = self.feature_extractor(
            waveform, sampling_rate=self.sample_rate, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        return out.hidden_states[-1].mean(dim=1)[0].cpu().numpy()  # mean over time
