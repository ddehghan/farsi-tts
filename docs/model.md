# Model

This project uses the [Chatterbox](https://github.com/resemble-ai/chatterbox) multilingual TTS model with Persian/Farsi fine-tuned weights.

## Base model

- **Chatterbox TTS** by Resemble AI — open-source multilingual text-to-speech
- Installed as a pip package from the git repo

## Persian fine-tune

- **Repo**: [Thomcles/Chatterbox-TTS-Persian-Farsi](https://huggingface.co/Thomcles/Chatterbox-TTS-Persian-Farsi)
- **File**: `t3_fa.safetensors`
- **What it replaces**: The T3 (text-to-token) component of the multilingual model
- After loading the base multilingual model, the fine-tuned T3 weights are swapped in via `load_state_dict`

## Download

```bash
python src/download_model.py
```

Requires `HF_TOKEN` environment variable set (see README).
