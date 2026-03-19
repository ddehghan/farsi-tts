"""Chatterbox TTS with Persian fine-tuned weights."""

import os

import torch
from safetensors.torch import load_file as load_safetensors

from .base import TTSModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ChatterboxModel(TTSModel):
    name = "chatterbox"

    def __init__(self, weights_path=None, language_id=None, temperature=0.7,
                 cfg_weight=0.5, top_p=0.5, exaggeration=0.6):
        self.weights_path = weights_path or os.path.join(PROJECT_ROOT, "models", "t3_fa.safetensors")
        self.language_id = language_id
        self.temperature = temperature
        self.cfg_weight = cfg_weight
        self.top_p = top_p
        self.exaggeration = exaggeration
        self.model = None

    def load(self, device: str) -> None:
        from chatterbox import mtl_tts

        self.model = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device=device)
        t3_state = load_safetensors(self.weights_path, device=device)
        missing, unexpected = self.model.t3.load_state_dict(t3_state, strict=False)
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        self.model.t3.to(device).eval()
        print(f"Chatterbox model loaded on {device}")

    def generate(self, text: str) -> tuple[torch.Tensor, int]:
        wav = self.model.generate(
            text,
            language_id=self.language_id,
            temperature=self.temperature,
            cfg_weight=self.cfg_weight,
            top_p=self.top_p,
            exaggeration=self.exaggeration,
        )
        return wav, self.model.sr
