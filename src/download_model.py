"""Download the Persian TTS model weights from HuggingFace."""

import os
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "Thomcles/Chatterbox-TTS-Persian-Farsi"
FILENAME = "t3_fa.safetensors"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def download_model(force=False):
    target = os.path.join(MODELS_DIR, FILENAME)
    if os.path.exists(target) and not force:
        print("Model already present at:", target)
        return target

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Downloading {FILENAME} from HuggingFace...")
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        token=os.environ.get("HF_TOKEN"),
    )
    shutil.copy(downloaded, target)
    print("Saved to:", target)
    return target


if __name__ == "__main__":
    download_model()
