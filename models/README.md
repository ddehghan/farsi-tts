# Models

This directory holds model weight files. They are not checked into git due to their size.

## Required file

- **`t3_fa.safetensors`** — Persian/Farsi fine-tuned T3 weights for Chatterbox TTS

## How to get it

### Option 1: Auto-download

```bash
python src/download_model.py
```

Requires `HF_TOKEN` environment variable (see root README).

### Option 2: Manual download

1. Go to [Thomcles/Chatterbox-TTS-Persian-Farsi](https://huggingface.co/Thomcles/Chatterbox-TTS-Persian-Farsi)
2. Download `t3_fa.safetensors`
3. Place it in this `models/` directory
