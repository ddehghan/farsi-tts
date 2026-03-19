"""Fish Speech — large-scale LLM-based TTS for advanced/experimental use."""

import torch

from .base import TTSModel


class FishModel(TTSModel):
    name = "fish"

    def __init__(self, model_name="fishaudio/fish-speech-1.5", reference_audio=None,
                 reference_text=None):
        self.model_name = model_name
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.model = None

    def load(self, device: str) -> None:
        try:
            from fish_speech.inference import TTSInference
        except ImportError:
            raise ImportError(
                "Fish Speech is not installed. See:\n"
                "  https://github.com/fishaudio/fish-speech\n"
                "Requires GPU with significant VRAM."
            )

        self.device = device
        self.model = TTSInference(self.model_name, device=device)
        print(f"Fish Speech model loaded on {device}")

    def generate(self, text: str) -> tuple[torch.Tensor, int]:
        result = self.model.inference(
            text=text,
            reference_audio=self.reference_audio,
            reference_text=self.reference_text,
        )
        wav = torch.from_numpy(result["audio"]).unsqueeze(0).float()
        return wav, result["sample_rate"]
