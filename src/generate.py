"""Generate speech from a JSON input file using the Persian Chatterbox TTS model."""

import argparse
import json
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


def generate(model, text):
    return model.generate(
        text,
        language_id=None,
        temperature=0.7,
        cfg_weight=0.5,
        top_p=0.5,
        exaggeration=0.6,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate Persian TTS from a JSON input file")
    parser.add_argument("input_file", help="Path to JSON input file")
    parser.add_argument("--model", default=os.path.join(PROJECT_ROOT, "models", "t3_fa.safetensors"),
                        help="Path to model weights")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Run: python src/download_model.py")
        sys.exit(1)

    with open(args.input_file) as f:
        entries = json.load(f)

    # Output goes in a folder next to the input file, named after it
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    output_dir = os.path.join(os.path.dirname(args.input_file), input_basename)
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    model = load_model(args.model, device)

    for i, entry in enumerate(entries):
        filename = entry["filename"]
        text = entry["text"]
        print(f"\n[{i+1}/{len(entries)}] Generating: {filename}")
        wav = generate(model, text)
        output_path = os.path.join(output_dir, filename)
        ta.save(output_path, wav, model.sr)
        print(f"Saved: {output_path}")

    print(f"\nDone. {len(entries)} files saved to {output_dir}")


if __name__ == "__main__":
    main()
