"""Per-utterance streaming pipeline: text (+ vision) (+ audio) -> structured
UtteranceTag with grounded response text, one conversational turn at a time.

"Real-time" here means per-utterance streaming (see README): each turn is
processed and answered within a target latency budget before the next turn
arrives, not frame-by-frame video analysis. Turns "arrive" either by
iterating a MELD dialogue in speaking order (`run_meld_dialogue`) or from a
short live webcam+mic capture (`run_live`, local use only).
"""
from __future__ import annotations

import time

import torch
import yaml

from src.encoders import AudioEncoder, TextEncoder, VisionEncoder
from src.fusion import FusionHead, RLGatePolicy, concat_embeddings, evaluate  # noqa: F401 (evaluate re-exported)
from src.responder import Responder
from src.schema import MELD_EMOTIONS, ModalityGate, UtteranceTag


class Pipeline:
    def __init__(self, config_path: str, device: str = "cpu", fusion_ckpt: str | None = None, gate_ckpt: str | None = None):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        ecfg = self.cfg["encoders"]

        self.text_encoder = TextEncoder(ecfg["text"]["checkpoint"], device=device)
        self.vision_encoder: VisionEncoder | None = None
        self.audio_encoder: AudioEncoder | None = None  # lazy: only needed for tri-modal runs

        fcfg = self.cfg["fusion"]
        self.embed_dims = fcfg["embed_dims"]
        self.fusion = FusionHead(self.embed_dims, fcfg["hidden_dim"], fcfg["num_classes"])
        self.gate = RLGatePolicy(self.embed_dims, fcfg["gate_hidden_dim"])
        self.trained = False
        if fusion_ckpt:
            self.fusion.load_state_dict(torch.load(fusion_ckpt, map_location="cpu"))
            self.trained = True
        if gate_ckpt:
            self.gate.load_state_dict(torch.load(gate_ckpt, map_location="cpu"))
        self.fusion.eval()
        self.gate.eval()

        rcfg = self.cfg["responder"]
        self.responder = Responder(rcfg["checkpoint"], device=device, max_new_tokens=rcfg["max_new_tokens"], temperature=rcfg["temperature"])

    def _ensure_vision(self):
        if self.vision_encoder is None:
            self.vision_encoder = VisionEncoder(self.cfg["encoders"]["vision"]["checkpoint"])
        return self.vision_encoder

    def _ensure_audio(self):
        if self.audio_encoder is None:
            self.audio_encoder = AudioEncoder(self.cfg["encoders"]["audio"]["checkpoint"])
        return self.audio_encoder

    def tag(self, utterance_id: str, text: str, frames=None, waveform=None, use_rl_gate: bool = True) -> UtteranceTag:
        t0 = time.perf_counter()
        embeds = {"text": self.text_encoder.encode(text), "vision": None, "audio": None}
        used = ["text"]
        if frames:
            embeds["vision"] = self._ensure_vision().encode(frames)
            used.append("vision")
        if waveform is not None:
            embeds["audio"] = self._ensure_audio().encode(waveform)
            used.append("audio")

        x = concat_embeddings(embeds, self.embed_dims).unsqueeze(0)
        gate_probs = None
        with torch.no_grad():
            if use_rl_gate:
                gate_probs = self.gate.gate_probs(x)
                x_in = self.gate.apply_gate(x, self.embed_dims, gate_probs)
            else:
                x_in = x
            logits = self.fusion(x_in)
            probs = torch.softmax(logits, dim=-1)[0]

        top = int(probs.argmax())
        emotion = MELD_EMOTIONS[top]
        confidence = float(probs[top])
        class_probs = {MELD_EMOTIONS[i]: float(probs[i]) for i in range(len(MELD_EMOTIONS))}

        cues = []
        if "vision" in used:
            cues.append("facial expression observed")
        if "audio" in used:
            cues.append("vocal tone observed")

        response = self.responder.generate(text, emotion, confidence, modality_cues=cues)
        latency_ms = (time.perf_counter() - t0) * 1000

        mg = None
        if gate_probs is not None:
            g = gate_probs[0].tolist()
            mg = ModalityGate(text=g[0], vision=g[1], audio=g[2])

        return UtteranceTag(
            utterance_id=utterance_id,
            text=text,
            emotion=emotion,
            emotion_confidence=confidence,
            class_probs=class_probs,
            modality_gate=mg,
            modalities_used=used,
            response_text=response,
            latency_ms=latency_ms,
        )


def run_meld_dialogue(pipeline: Pipeline, meld_root: str, split: str, dialogue_id: int, n_frames: int = 3, use_audio: bool = True):
    """Simulates turns arriving over time by iterating one MELD conversation
    in speaking order, extracting frames/audio per clip on the fly."""
    from src.data_meld import dialogue, extract_audio, extract_keyframes, load_split

    utterances = dialogue(load_split(meld_root, split), dialogue_id)
    for u in utterances:
        frames = extract_keyframes(u.video_path, n_frames=n_frames)
        waveform = extract_audio(u.video_path) if use_audio else None
        yield u, pipeline.tag(u.utterance_id, u.text, frames=frames, waveform=waveform)


def run_live(pipeline: Pipeline, seconds: float = 3.0, n_frames: int = 3):
    """Local-only: grabs `seconds` of webcam frames + mic audio, then tags them.
    Requires opencv-python and sounddevice, and an actual camera/mic — meant for
    a live demo on the user's own machine, not for the Kaggle notebook."""
    import cv2
    import sounddevice as sd
    from PIL import Image

    cap = cv2.VideoCapture(0)
    frames, t_end = [], time.time() + seconds
    audio = sd.rec(int(16_000 * seconds), samplerate=16_000, channels=1, dtype="float32")
    while time.time() < t_end and len(frames) < n_frames * 5:
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        time.sleep(seconds / (n_frames * 5))
    sd.wait()
    cap.release()

    step = max(len(frames) // n_frames, 1)
    sampled = frames[::step][:n_frames]
    text = input("You said (type it — ASR is out of scope for this prototype): ").strip()
    return pipeline.tag("live", text, frames=sampled, waveform=audio.reshape(-1))
