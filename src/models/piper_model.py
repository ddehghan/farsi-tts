"""Piper TTS — fast, lightweight, good Ezafe accuracy for Persian."""

import os

import numpy as np
import torch

from .base import TTSModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_VOICE = os.path.join(PROJECT_ROOT, "models", "fa_IR-mana-medium.onnx")


class PiperModel(TTSModel):
    name = "piper"

    def __init__(self, model_path=None, speaker_id=None, length_scale=None,
                 noise_scale=None, noise_w_scale=None):
        self.model_path = model_path or DEFAULT_VOICE
        # Derive name from voice file: fa_IR-mana-medium.onnx -> piper_mana
        voice_basename = os.path.splitext(os.path.basename(self.model_path))[0]
        parts = voice_basename.split("-")
        self.name = f"piper_{parts[1]}" if len(parts) >= 2 else "piper"
        self.speaker_id = speaker_id
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale
        self.voice = None

    def load(self, device: str) -> None:
        try:
            from piper.voice import PiperVoice
        except ImportError:
            raise ImportError(
                "Piper is not installed. Install with:\n"
                "  uv pip install piper-tts\n"
                "Then download a voice:\n"
                "  python src/download_piper_voice.py"
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Piper voice not found: {self.model_path}\n"
                "Download with: python src/download_piper_voice.py"
            )

        self.voice = PiperVoice.load(self.model_path)
        print(f"Piper model loaded from {self.model_path}")
        print(f"Sample rate: {self.voice.config.sample_rate}")

    def generate(self, text: str) -> tuple[torch.Tensor, int]:
        from piper.config import SynthesisConfig

        syn_config = SynthesisConfig(
            speaker_id=self.speaker_id,
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w_scale,
        )

        chunks = list(self.voice.synthesize(text, syn_config=syn_config))
        sample_rate = chunks[0].sample_rate

        audio = np.concatenate([c.audio_float_array for c in chunks])
        wav = torch.from_numpy(audio).unsqueeze(0).float()
        return wav, sample_rate
