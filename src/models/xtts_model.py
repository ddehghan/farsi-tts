"""XTTS-v2 — high-fidelity with voice cloning support."""

import torch

from .base import TTSModel


class XTTSModel(TTSModel):
    name = "xtts"

    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                 speaker_wav=None, language="fa"):
        self.model_name = model_name
        self.speaker_wav = speaker_wav
        self.language = language
        self.tts = None

    def load(self, device: str) -> None:
        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError(
                "Coqui TTS is not installed. Install with:\n"
                "  uv pip install TTS"
            )

        self.tts = TTS(self.model_name).to(device)
        print(f"XTTS-v2 model loaded on {device}")

    def generate(self, text: str) -> tuple[torch.Tensor, int]:
        if self.speaker_wav:
            wav_list = self.tts.tts(text=text, speaker_wav=self.speaker_wav, language=self.language)
        else:
            wav_list = self.tts.tts(text=text, language=self.language)

        wav = torch.tensor(wav_list, dtype=torch.float32).unsqueeze(0)
        sample_rate = self.tts.synthesizer.output_sample_rate
        return wav, sample_rate
