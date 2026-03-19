"""MeloTTS — fast multilingual TTS, CPU-friendly.

Supported languages: EN, ES, FR, ZH, JP, KR (no Farsi).

Install:
  git clone https://github.com/myshell-ai/MeloTTS.git
  cd MeloTTS && pip install -e . && python -m unidic download
"""

import os
import tempfile

import torch
import torchaudio as ta

from .base import TTSModel


class MeloModel(TTSModel):
    name = "melo"

    def __init__(self, language="EN", speaker="EN-US", speed=1.0):
        self.language = language
        self.speaker = speaker
        self.speed = speed
        self.model = None
        self.speaker_id = None

    def load(self, device: str) -> None:
        try:
            from melo.api import TTS
        except ImportError:
            raise ImportError(
                "MeloTTS is not installed. Install with:\n"
                "  git clone https://github.com/myshell-ai/MeloTTS.git\n"
                "  cd MeloTTS && pip install -e . && python -m unidic download"
            )

        self.model = TTS(language=self.language, device=device)
        speaker_ids = self.model.hps.data.spk2id
        if self.speaker not in speaker_ids:
            available = ", ".join(speaker_ids.keys())
            raise ValueError(f"Unknown speaker '{self.speaker}'. Available: {available}")
        self.speaker_id = speaker_ids[self.speaker]
        print(f"MeloTTS loaded (language={self.language}, speaker={self.speaker}, device={device})")

    def generate(self, text: str) -> tuple[torch.Tensor, int]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            self.model.tts_to_file(text, self.speaker_id, tmp.name, speed=self.speed)
            wav, sr = ta.load(tmp.name)
        return wav, sr
