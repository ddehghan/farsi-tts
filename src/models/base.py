"""Base class for TTS model adapters."""

from abc import ABC, abstractmethod

import torch


class TTSModel(ABC):
    """Common interface for all TTS models."""

    name: str  # Short identifier used in filenames (e.g. "chatterbox")

    @abstractmethod
    def load(self, device: str) -> None:
        """Load model weights onto the given device."""

    @abstractmethod
    def generate(self, text: str) -> tuple[torch.Tensor, int]:
        """Generate speech from text.

        Returns:
            (waveform, sample_rate) — waveform shape is (1, num_samples).
        """

    @staticmethod
    def get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
