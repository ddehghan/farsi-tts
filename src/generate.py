"""Generate speech from a JSON input file using swappable TTS models."""

import argparse
import json
import os
import sys

import torchaudio as ta

from models import get_model, list_models


def main():
    parser = argparse.ArgumentParser(description="Generate TTS from a JSON input file")
    parser.add_argument("input_file", nargs="?", help="Path to JSON input file")
    parser.add_argument("-m", "--model", default="chatterbox",
                        choices=list_models(),
                        help="TTS model to use (default: chatterbox)")
    parser.add_argument("--model-path", help="Path to model weights (model-specific)")
    parser.add_argument("--speaker-wav", help="Reference speaker audio for voice cloning (xtts, fish)")
    parser.add_argument("--language", help="Language code override (default: model-specific)")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    args = parser.parse_args()

    if args.list_models:
        print("Available models:", ", ".join(list_models()))
        return

    if not args.input_file:
        parser.error("input_file is required")

    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        sys.exit(1)

    with open(args.input_file) as f:
        entries = json.load(f)

    # Build model kwargs from CLI args
    model_kwargs = {}
    if args.model_path:
        # Map to model-specific param names
        param_map = {
            "chatterbox": "weights_path",
            "piper": "model_path",
            "xtts": "model_name",
            "fish": "model_name",
        }
        key = param_map.get(args.model, "model_path")
        model_kwargs[key] = args.model_path
    if args.speaker_wav:
        model_kwargs["speaker_wav"] = args.speaker_wav
        if args.model == "fish":
            model_kwargs["reference_audio"] = args.speaker_wav
    if args.language:
        model_kwargs["language"] = args.language
        if args.model == "chatterbox":
            model_kwargs["language_id"] = args.language

    tts = get_model(args.model, **model_kwargs)

    from models.base import TTSModel
    device = TTSModel.get_device()
    print(f"Using device: {device}")
    print(f"Using model: {args.model}")

    tts.load(device)

    # Output dir: input dir / input basename / model name
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    output_dir = os.path.join(os.path.dirname(args.input_file), input_basename)
    os.makedirs(output_dir, exist_ok=True)

    for i, entry in enumerate(entries):
        base_name = os.path.splitext(entry["filename"])[0]
        output_filename = f"{base_name}_{tts.name}.wav"
        text = entry["text"]

        print(f"\n[{i+1}/{len(entries)}] Generating: {output_filename}")
        wav, sr = tts.generate(text)
        output_path = os.path.join(output_dir, output_filename)
        ta.save(output_path, wav, sr)
        print(f"Saved: {output_path}")

    print(f"\nDone. {len(entries)} files saved to {output_dir}")


if __name__ == "__main__":
    main()
