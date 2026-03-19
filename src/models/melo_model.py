"""MeloTTS — fast multilingual TTS, good for simple web apps."""

import io
import tempfile

import torch
import torchaudio as ta

from .base import TTSModel


class MeloModel(TTSModel):
    name = "melo"

    def __init__(self, language="EN", speaker_id=None, speed=1.0):
        self.language = language
        self.speaker_id = speaker_id
        self.speed = speed
        self.model = None

    def load(self, device: str) -> None:
        try:
            from melo.api import TTS
        except ImportError:
            raise ImportError(
                "MeloTTS is not installed. Install with:\n"
                "  uv pip install git+https://github.com/myshell-ai/MeloTTS.git"
            )

        self.device = device
        self.model = TTS(language=self.language, device=device)
        if self.speaker_id is None:
            speaker_ids = self.model.hps.data.spk2id
            self.speaker_id = list(speaker_ids.values())[0]
        print(f"MeloTTS model loaded on {device}")

    def generate(self, text: str) -> tuple[torch.Tensor, int]:
        # MeloTTS writes to file, so we use a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            self.model.tts_to_file(text, self.speaker_id, tmp.name, speed=self.speed)
            wav, sr = ta.load(tmp.name)
        return wav, sr
