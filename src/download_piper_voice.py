"""Download Persian Piper TTS voices from HuggingFace."""

import argparse
import os
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

VOICES = {
    "amir": "fa/fa_IR/amir/medium/fa_IR-amir-medium.onnx",
    "ganji": "fa/fa_IR/ganji/medium/fa_IR-ganji-medium.onnx",
    "ganji_adabi": "fa/fa_IR/ganji_adabi/medium/fa_IR-ganji_adabi-medium.onnx",
    "gyro": "fa/fa_IR/gyro/medium/fa_IR-gyro-medium.onnx",
    "reza_ibrahim": "fa/fa_IR/reza_ibrahim/medium/fa_IR-reza_ibrahim-medium.onnx",
}


def download_file(url, dest):
    print(f"  Downloading {os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest)


def download_voice(voice_name, force=False):
    if voice_name not in VOICES:
        print(f"Unknown voice: {voice_name}")
        print(f"Available: {', '.join(VOICES.keys())}")
        return

    onnx_path = VOICES[voice_name]
    onnx_filename = os.path.basename(onnx_path)
    json_filename = onnx_filename + ".json"

    onnx_dest = os.path.join(MODELS_DIR, onnx_filename)
    json_dest = os.path.join(MODELS_DIR, json_filename)

    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(onnx_dest) and os.path.exists(json_dest) and not force:
        print(f"Voice '{voice_name}' already downloaded: {onnx_dest}")
        return

    print(f"Downloading Piper voice: {voice_name}")
    download_file(f"{BASE_URL}/{onnx_path}", onnx_dest)
    download_file(f"{BASE_URL}/{onnx_path}.json", json_dest)
    print(f"Saved to: {onnx_dest}")


def main():
    parser = argparse.ArgumentParser(description="Download Persian Piper TTS voices")
    parser.add_argument("voice", nargs="?", default="amir",
                        choices=list(VOICES.keys()),
                        help="Voice to download (default: amir)")
    parser.add_argument("--all", action="store_true", help="Download all Persian voices")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    parser.add_argument("--list", action="store_true", help="List available voices")
    args = parser.parse_args()

    if args.list:
        print("Available Persian Piper voices:")
        for name in VOICES:
            print(f"  {name}")
        return

    if args.all:
        for name in VOICES:
            download_voice(name, force=args.force)
    else:
        download_voice(args.voice, force=args.force)


if __name__ == "__main__":
    main()
