# Persian TTS with Chatterbox

Text-to-speech for Persian/Farsi using [Chatterbox](https://github.com/resemble-ai/chatterbox) with fine-tuned weights.

## Samples

[Listen to the samples here](https://ddehghan.github.io/farsi-tts/) — five prose clips covering history, cooking, science, daily conversation, and geography.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- [direnv](https://direnv.net/docs/installation.html) — auto-loads `.envrc` when you `cd` into the project

## Setup

### 1. Create virtual environment and install dependencies

```bash
uv venv
uv pip install -e .
```

This installs chatterbox and all its dependencies automatically. The `pyproject.toml` includes overrides for `numpy`, `transformers`, and `setuptools` to ensure compatibility with Python 3.12.

### 2. Get a Hugging Face token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is sufficient)
3. Add it to your `.envrc`:

```bash
source .venv/bin/activate
export HF_TOKEN=hf_your_token_here
```

Then run `direnv allow`.

### 3. Download model weights

```bash
python src/download_model.py
```

Or manually download `t3_fa.safetensors` from [Thomcles/Chatterbox-TTS-Persian-Farsi](https://huggingface.co/Thomcles/Chatterbox-TTS-Persian-Farsi) and place it in `models/`.

### 4. Run

Create a JSON file in `input/` with an array of entries:

```json
[
  {"filename": "my_clip.wav", "text": "متن فارسی شما اینجا"},
  {"filename": "another_clip.wav", "text": "متن دیگر"}
]
```

Then run:

```bash
python src/generate.py input/samples.json
```

Output wavs are saved to `input/samples/` (a folder next to the JSON, named after it):

```
input/
├── samples.json
└── samples/
    ├── my_clip.wav
    └── another_clip.wav
```

## Project structure

```
├── docs/             # documentation
├── input/            # input JSON files and generated wav output
├── models/           # model weights (.safetensors)
├── src/
│   ├── download_model.py  # download model from HuggingFace
│   └── generate.py        # TTS generation script
├── .envrc
└── pyproject.toml
```
