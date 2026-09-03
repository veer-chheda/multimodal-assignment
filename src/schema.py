"""Structured output contract produced for every utterance."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

MELD_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


@dataclass
class ModalityGate:
    text: float
    vision: float
    audio: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class UtteranceTag:
    """The structured state/tags required by the assignment, plus grounded response text."""

    utterance_id: str
    text: str
    emotion: str
    emotion_confidence: float
    class_probs: dict = field(default_factory=dict)
    modality_gate: ModalityGate | None = None
    modalities_used: list[str] = field(default_factory=list)
    response_text: str = ""
    latency_ms: float | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        if self.modality_gate is not None:
            d["modality_gate"] = self.modality_gate.as_dict()
        return d

    def __post_init__(self):
        assert self.emotion in MELD_EMOTIONS, f"unknown emotion label: {self.emotion}"
