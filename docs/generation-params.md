# Generation Parameters

The `generate.py` script uses these parameters when calling `ChatterboxMultilingualTTS.generate()`:

| Parameter      | Default | Description                                                                 |
|----------------|---------|-----------------------------------------------------------------------------|
| `language_id`  | `None`  | Language identifier. `None` lets the model auto-detect.                     |
| `temperature`  | `0.7`   | Controls randomness. Lower = more deterministic, higher = more varied.      |
| `cfg_weight`   | `0.5`   | Classifier-free guidance weight. Higher = stronger adherence to the prompt. |
| `top_p`        | `0.5`   | Nucleus sampling threshold. Lower = fewer token choices, more focused.      |
| `exaggeration` | `0.6`   | Controls expressiveness/prosody. Higher = more dramatic speech.             |

## Tips

- For clearer, more stable output, lower `temperature` and `top_p` (e.g. 0.5/0.3).
- For more natural, varied speech, raise `temperature` and `exaggeration`.
- If output sounds garbled, try reducing `exaggeration` to 0.3-0.4.
