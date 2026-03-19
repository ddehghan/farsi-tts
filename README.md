# Persian TTS with Chatterbox

Text-to-speech for Persian/Farsi using [Chatterbox](https://github.com/resemble-ai/chatterbox) with fine-tuned weights.

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

Create a text file in `input/` (e.g. `input/my_text.txt`), then:

```bash
python src/generate.py input/my_text.txt
```

Output is saved to `output/my_text.wav` (named after the input file).

## Project structure

```
├── docs/             # documentation
├── input/            # input text files
├── models/           # model weights (.safetensors)
├── output/           # generated wav files
├── src/
│   ├── download_model.py  # download model from HuggingFace
│   └── generate.py        # TTS generation script
├── .envrc
└── pyproject.toml
```
