"""XTTS-v2 — high-fidelity with voice cloning support.

Supported languages:
  en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi

NOTE: Persian (fa) is NOT supported. Falls back to English.
For Persian TTS, use chatterbox or piper.

Requires a speaker reference WAV (~6+ seconds) for voice cloning.

Install: uv pip install coqui-tts
  - Requires transformers >= 4.53 (conflicts with chatterbox's ==4.52 pin).
  - May need a separate venv. See README for details.
"""

import torch

from .base import TTSModel

SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr",
    "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
]


class XTTSModel(TTSModel):
    name = "xtts"

    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                 speaker_wav=None, language="en"):
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
                "  uv pip install coqui-tts\n"
                "NOTE: Requires transformers >= 4.53 — may conflict with chatterbox.\n"
                "Consider using a separate venv."
            )

        if not self.speaker_wav:
            raise ValueError(
                "XTTS-v2 requires a speaker reference WAV for voice cloning.\n"
                "Pass --speaker-wav /path/to/reference.wav (at least 6 seconds)."
            )

        if self.language not in SUPPORTED_LANGUAGES:
            print(f"WARNING: language '{self.language}' is not supported by XTTS-v2.")
            print(f"Supported: {', '.join(SUPPORTED_LANGUAGES)}")
            print(f"Falling back to 'en' (English).")
            self.language = "en"

        self.tts = TTS(self.model_name, gpu=(device == "cuda"))
        if device == "mps":
            self.tts.synthesizer.tts_model.to("mps")
        print(f"XTTS-v2 model loaded (device={device}, language={self.language})")

    def generate(self, text: str) -> tuple[torch.Tensor, int]:
        wav_list = self.tts.tts(
            text=text,
            speaker_wav=self.speaker_wav,
            language=self.language,
        )
        wav = torch.tensor(wav_list, dtype=torch.float32).unsqueeze(0)
        sample_rate = self.tts.synthesizer.output_sample_rate
        return wav, sample_rate
