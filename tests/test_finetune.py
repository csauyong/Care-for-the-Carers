"""Smoke test for the LoRA training script and FinetunedDetector, with a tiny local model.

Checks the pipeline end to end (train, merge, save, load, detect) without
network access. It says nothing about accuracy: that needs the real base model
(see finetune/README.md). Skipped unless the `finetune` extra is installed.
"""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
ROWS = [
    ("I keep crying every night", "sadness;grief"),
    ("I am so worried and nervous", "fear;nervousness"),
    ("I am so angry with everyone", "anger;annoyance"),
    ("Thank you today was a happy day", "joy;gratitude"),
    ("What time does the centre open", "neutral"),
    ("我 好 傷心", "sadness"),
]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaForSequenceClassification

    root = tmp_path_factory.mktemp("ft")
    with (root / "train.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "labels"])
        writer.writerows(ROWS * 5)

    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    special = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tok.train_from_iterator([t for t, _ in ROWS], trainers.WordLevelTrainer(special_tokens=special))
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", pad_token="<pad>", eos_token="</s>", unk_token="<unk>", mask_token="<mask>"
    )
    base = root / "tiny-xlmr"
    fast.save_pretrained(base)
    config = XLMRobertaConfig(
        vocab_size=len(fast), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64, pad_token_id=fast.pad_token_id,
    )
    XLMRobertaForSequenceClassification(config).save_pretrained(base)
    return root


def test_train_merge_and_detect(tiny_model):
    out = tiny_model / "out"
    cmd = [
        sys.executable, str(ROOT / "finetune/train_emotion_lora.py"),
        "--base-model", str(tiny_model / "tiny-xlmr"),
        "--train-csv", str(tiny_model / "train.csv"),
        "--epochs", "2", "--batch-size", "8", "--lora-r", "4",
        "--output-dir", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    assert (out / "adapter" / "adapter_config.json").exists()
    assert (out / "merged" / "carebot_labels.json").exists()
    assert "stress_f1" in (out / "metrics.json").read_text()

    from carebot.signals import FinetunedDetector

    detector = FinetunedDetector(str(out / "merged"), device="cpu")
    calm = detector.detect("What time does the centre open")
    assert not calm.crisis and 0 <= calm.stress <= 3
    assert any(":" in e for e in calm.evidence)  # top emotions with probabilities
    # Crisis always comes from the keyword screen, whatever the classifier says.
    assert detector.detect("I want to die").crisis
