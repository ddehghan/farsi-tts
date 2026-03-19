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

## Available models

| Model | Language | Install | Notes |
|-------|----------|---------|-------|
| chatterbox | Farsi | included | Default. Uses fine-tuned Persian weights |
| piper | Farsi | `uv pip install piper-tts` | Fast, lightweight, CPU-friendly |
| xtts | English (17 langs, no Farsi) | `uv pip install coqui-tts` | High-fidelity, requires `--speaker-wav` |
| melo | English (6 langs, no Farsi) | `uv pip install git+https://github.com/myshell-ai/MeloTTS.git` | Fast, CPU-friendly, multiple accents |

List available models:

```bash
python src/generate.py --list-models
```

## Generating audio

### Farsi with Chatterbox (default)

```bash
python src/generate.py input/samples.json
```

### Farsi with Piper (Mana voice — recommended)

[Mana](https://huggingface.co/MahtaFetrat/Mana-Persian-Piper) is trained on 114h of data with improved Ezafe accuracy.

```bash
python src/download_piper_voice.py mana
python src/generate.py input/samples.json -m piper
```

### Farsi with Piper (standard voices)

```bash
python src/download_piper_voice.py amir
python src/generate.py input/samples.json -m piper --model-path models/fa_IR-amir-medium.onnx
```

Other voices: `ganji`, `ganji_adabi`, `gyro`, `reza_ibrahim`

```bash
python src/download_piper_voice.py --list
```

### English with XTTS-v2

XTTS-v2 requires a speaker reference WAV (at least 6 seconds) for voice cloning:

```bash
python src/generate.py input/samples_en.json -m xtts --speaker-wav input/samples/daily_conversation_piper.wav
```

### English with MeloTTS

```bash
python src/generate.py input/samples_en.json -m melo
```

### Input format

Create a JSON file with an array of entries:

```json
[
  {"filename": "my_clip.wav", "text": "متن فارسی شما اینجا"},
  {"filename": "another_clip.wav", "text": "متن دیگر"}
]
```

### Output naming

Output files include the model name: `{name}_{model}.wav`

```
input/
├── samples.json
└── samples/
    ├── history_persepolis_chatterbox.wav
    ├── history_persepolis_piper.wav
    └── ...
```

## Project structure

```
├── docs/             # documentation
├── input/            # input JSON files and generated wav output
├── models/           # model weights (.safetensors, .onnx)
├── src/
│   ├── models/            # TTS model adapters
│   │   ├── base.py             # abstract TTSModel interface
│   │   ├── chatterbox_model.py # Chatterbox Persian
│   │   ├── piper_model.py      # Piper
│   │   ├── xtts_model.py       # XTTS-v2
│   │   ├── melo_model.py       # MeloTTS
│   │   └── melo_model.py        # MeloTTS
│   ├── download_model.py       # download Chatterbox weights
│   ├── download_piper_voice.py # download Piper Persian voices
│   └── generate.py             # main generation script
├── .envrc
└── pyproject.toml
```
