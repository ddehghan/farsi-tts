"""Generate speech from a text file using the Persian Chatterbox TTS model."""

import argparse
import os
import sys

import torch
import torchaudio as ta
from safetensors.torch import load_file as load_safetensors

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from chatterbox import mtl_tts


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(t3_path, device):
    model = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device=device)
    t3_state = load_safetensors(t3_path, device=device)
    missing, unexpected = model.t3.load_state_dict(t3_state, strict=False)
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    model.t3.to(device).eval()
    print("Model loaded")
    return model


def generate(model, text, device):
    return model.generate(
        text,
        language_id=None,
        temperature=0.7,
        cfg_weight=0.5,
        top_p=0.5,
        exaggeration=0.6,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate Persian TTS from a text file")
    parser.add_argument("input_file", help="Path to input text file")
    parser.add_argument("--model", default=os.path.join(PROJECT_ROOT, "models", "t3_fa.safetensors"),
                        help="Path to model weights")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "output"),
                        help="Directory for output wav files")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Run: python src/download_model.py")
        sys.exit(1)

    text = open(args.input_file).read().strip()
    if not text:
        print("Input file is empty")
        sys.exit(1)

    device = get_device()
    print(f"Using device: {device}")

    model = load_model(args.model, device)
    wav = generate(model, text, device)

    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.input_file))[0]
    output_path = os.path.join(args.output_dir, f"{basename}.wav")
    ta.save(output_path, wav, model.sr)
    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    main()
