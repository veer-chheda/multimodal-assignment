"""Grounded response generation.

The response LLM is used frozen / prompted only (no fine-tuning) — it is
conditioned on the transcript, the predicted MELD emotion + confidence,
and (when available) short human-readable modality cues, so the reply is
grounded in what the fusion head actually inferred rather than free-form.
"""
from __future__ import annotations

from threading import Thread

SYSTEM_PROMPT = (
    "You are the voice of a character robot having a short spoken conversation. "
    "You are told the other speaker's utterance and the emotion your perception "
    "system detected from their tone/expression. Reply in 1-2 short, natural "
    "sentences that acknowledge that emotion appropriately. Do not mention the "
    "word 'emotion', a confidence score, or that you are an AI."
)


def build_prompt(utterance: str, emotion: str, confidence: float, modality_cues: list[str] | None = None) -> str:
    cues = f" Additional cues: {', '.join(modality_cues)}." if modality_cues else ""
    return (
        f"Speaker said: \"{utterance}\"\n"
        f"Detected emotion: {emotion} (confidence {confidence:.2f}).{cues}\n"
        f"Respond as the robot:"
    )


class Responder:
    def __init__(self, checkpoint: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cpu", max_new_tokens: int = 60, temperature: float = 0.7):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.float16 if device != "cpu" else torch.float32
        ).to(device).eval()
        self._torch = torch

    def _messages(self, utterance: str, emotion: str, confidence: float, modality_cues):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(utterance, emotion, confidence, modality_cues)},
        ]

    def generate(self, utterance: str, emotion: str, confidence: float, modality_cues: list[str] | None = None) -> str:
        """Blocking, full-string generation."""
        torch = self._torch
        messages = self._messages(utterance, emotion, confidence, modality_cues)
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0, input_ids.shape[-1] :], skip_special_tokens=True).strip()

    def stream(self, utterance: str, emotion: str, confidence: float, modality_cues: list[str] | None = None):
        """Yields response text incrementally (token-by-token) for a live/streaming UI."""
        from transformers import TextIteratorStreamer

        messages = self._messages(utterance, emotion, confidence, modality_cues)
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        for chunk in streamer:
            yield chunk
        thread.join()
