"""TTS model registry — add new models here."""

from .base import TTSModel
from .chatterbox_model import ChatterboxModel
from .fish_model import FishModel
from .melo_model import MeloModel
from .piper_model import PiperModel
from .xtts_model import XTTSModel

MODELS: dict[str, type[TTSModel]] = {
    "chatterbox": ChatterboxModel,
    "piper": PiperModel,
    "xtts": XTTSModel,
    "melo": MeloModel,
    "fish": FishModel,
}


def get_model(name: str, **kwargs) -> TTSModel:
    """Get a model instance by name."""
    if name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODELS[name](**kwargs)


def list_models() -> list[str]:
    """Return available model names."""
    return list(MODELS.keys())
