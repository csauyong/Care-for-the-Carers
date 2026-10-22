# Fine-tuned emotion detector (LoRA)

`train_emotion_lora.py` fine-tunes a multilingual transformer to classify the
emotions in a message. The bot can then use it as its stress detector
(`CAREBOT_DETECTOR=finetuned`), and it is scored with the same benchmark as the
keyword and LLM detectors.

| | |
| --- | --- |
| Base model | `FacebookAI/xlm-roberta-base` (multilingual, including Chinese) |
| Task | Multi-label emotion classification, 28 labels |
| Data | [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) (simplified), plus your own data with `--extra-train-csv` |
| Method | LoRA (rank 16) on the attention query/value projections, plus a new classification head |
| How it is used | Stress is the highest probability among 8 distress emotions (sadness, fear, nervousness, grief, anger, annoyance, disappointment, remorse), mapped to the bot's 0–3 scale. **Crisis detection still comes from the keyword screen.** |

## Run it

You need a GPU and internet access to Hugging Face. A free Google Colab T4 is
enough: 3 epochs should take well under an hour (not yet timed).

```bash
pip install -e ".[finetune]"
python finetune/train_emotion_lora.py --output-dir models/emotion-xlmr-lora

# Score it on the same labelled messages as the other detectors:
CAREBOT_FINETUNED_MODEL=models/emotion-xlmr-lora/merged \
  carebot --provider scripted --detector finetuned eval detector

# Compare with the baselines:
carebot --provider scripted --detector keyword eval detector
carebot --detector llm eval detector          # needs ANTHROPIC_API_KEY
```

The script writes:

- `adapter/`: the LoRA weights alone (a few MB), to share or version.
- `merged/`: base model and LoRA merged, which the bot loads with plain `transformers`.
- `metrics.json`: validation micro-F1, macro-F1, F1 for each label, and `stress_f1`
  (the binary signal the bot uses), plus the training history.

## Doing it properly

- **Never train on `evals/detector_cases.yaml`.** It is the test set. Tune the
  threshold or epochs on GoEmotions validation data or your own dev data.
- **Cantonese.** GoEmotions is English, so Cantonese performance comes from
  XLM-R's cross-lingual transfer. Measure it on the Cantonese rows of the
  detector cases. To improve it, label a few hundred real (anonymised) Cantonese
  messages in the same CSV format (`text`, `labels` joined by `;`) and pass
  `--extra-train-csv`.
- **Report the comparison.** Report the fine-tuned model's precision and recall
  next to the keyword and LLM detectors on the same cases, plus its size and
  speed. A small local classifier runs with no API cost.

## What has been verified

`tests/test_finetune.py` (run in CI by the `finetune-smoke` job) builds a tiny
model locally and runs this script end to end: training, merging, saving,
loading it as the bot's detector, and detecting. That proves the pipeline
works. Accuracy can only come from a real training run on the real base model,
and that hasn't been done yet.
