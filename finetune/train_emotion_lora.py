"""Fine-tune a multilingual emotion classifier with LoRA, for use as the bot's stress detector.

Default setup: XLM-RoBERTa base (covers English and Chinese), multi-label
emotion classification on GoEmotions (28 labels). LoRA trains about 1% of the
weights, so it fits a free Colab T4 GPU in well under an hour.

    pip install -e ".[finetune]"
    python finetune/train_emotion_lora.py --output-dir models/emotion-xlmr-lora
    CAREBOT_FINETUNED_MODEL=models/emotion-xlmr-lora/merged carebot --detector finetuned eval detector

GoEmotions is English only. Cantonese comes from XLM-R's cross-lingual
transfer, so measure it on the Cantonese cases in evals/detector_cases.yaml,
and add your own labelled Cantonese messages with --extra-train-csv.
Never train on evals/detector_cases.yaml: it is the test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]
# Must match carebot.signals.DISTRESS_EMOTIONS (written into the model folder below).
DISTRESS = ["anger", "annoyance", "disappointment", "fear", "grief", "nervousness", "remorse", "sadness"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="FacebookAI/xlm-roberta-base")
    p.add_argument("--dataset", default="google-research-datasets/go_emotions", help="Hugging Face dataset id")
    p.add_argument("--train-csv", help="use a local CSV instead: columns `text` and `labels` (names joined by ;)")
    p.add_argument("--eval-csv", help="local validation CSV, same format")
    p.add_argument("--extra-train-csv", help="extra labelled data (e.g. your Cantonese messages), same format")
    p.add_argument("--output-dir", default="models/emotion-xlmr-lora")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--target-modules", default="query,value", help="attention projections to adapt")
    p.add_argument("--max-train-samples", type=int, help="subsample for quick experiments")
    p.add_argument("--threshold", type=float, default=0.5, help="probability above which a label is predicted")
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def read_csv(path: str, labels: list[str]) -> list[tuple[str, list[int]]]:
    index = {name: i for i, name in enumerate(labels)}
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            names = [n.strip() for n in row["labels"].split(";") if n.strip()]
            unknown = [n for n in names if n not in index]
            if unknown:
                raise ValueError(f"{path}: unknown labels {unknown}; use names from {labels}")
            rows.append((row["text"], [index[n] for n in names]))
    return rows


def load_data(args) -> tuple[list, list]:
    if args.train_csv:
        train = read_csv(args.train_csv, GOEMOTIONS_LABELS)
        val = read_csv(args.eval_csv, GOEMOTIONS_LABELS) if args.eval_csv else []
    else:
        from datasets import load_dataset

        ds = load_dataset(args.dataset, "simplified")
        train = [(r["text"], r["labels"]) for r in ds["train"]]
        val = [(r["text"], r["labels"]) for r in ds["validation"]]
    if args.extra_train_csv:
        train += read_csv(args.extra_train_csv, GOEMOTIONS_LABELS)
    random.Random(args.seed).shuffle(train)
    if args.max_train_samples:
        train = train[: args.max_train_samples]
    if not val:  # hold out 10% if no validation set was given
        cut = max(1, len(train) // 10)
        val, train = train[:cut], train[cut:]
    return train, val


def batches(rows, tokenizer, args, shuffle: bool):
    def collate(items):
        enc = tokenizer([t for t, _ in items], padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        target = torch.zeros(len(items), len(GOEMOTIONS_LABELS))
        for i, (_, labels) in enumerate(items):
            target[i, labels] = 1.0
        enc["labels"] = target
        return enc

    return DataLoader(rows, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def evaluate(model, loader, device, threshold: float) -> dict:
    model.eval()
    tp = torch.zeros(len(GOEMOTIONS_LABELS))
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    stress = {"tp": 0, "fp": 0, "fn": 0}
    distress_idx = [GOEMOTIONS_LABELS.index(d) for d in DISTRESS]
    for batch in loader:
        labels = batch.pop("labels")
        probs = torch.sigmoid(model(**{k: v.to(device) for k, v in batch.items()}).logits).cpu()
        pred = (probs >= threshold).float()
        tp += (pred * labels).sum(0)
        fp += (pred * (1 - labels)).sum(0)
        fn += ((1 - pred) * labels).sum(0)
        # The number the bot actually uses: is any distress emotion present?
        p_stress = pred[:, distress_idx].amax(1).bool()
        y_stress = labels[:, distress_idx].amax(1).bool()
        stress["tp"] += int((p_stress & y_stress).sum())
        stress["fp"] += int((p_stress & ~y_stress).sum())
        stress["fn"] += int((~p_stress & y_stress).sum())
    model.train()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    micro = float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()).clamp(min=1))
    s = stress
    return {
        "micro_f1": round(micro, 4),
        "macro_f1": round(float(f1.mean()), 4),
        "stress_f1": round(2 * s["tp"] / max(2 * s["tp"] + s["fp"] + s["fn"], 1), 4),
        "per_label_f1": {name: round(float(v), 4) for name, v in zip(GOEMOTIONS_LABELS, f1)},
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train, val = load_data(args)
    print(f"{len(train)} training and {len(val)} validation examples; device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(GOEMOTIONS_LABELS),
        problem_type="multi_label_classification",  # sigmoid + BCE: a message can carry several emotions
        id2label=dict(enumerate(GOEMOTIONS_LABELS)),
        label2id={n: i for i, n in enumerate(GOEMOTIONS_LABELS)},
        ignore_mismatched_sizes=True,
    )
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # also keeps the new classification head trainable
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","),
    )
    model = get_peft_model(model, lora).to(device)
    model.print_trainable_parameters()

    train_loader = batches(train, tokenizer, args, shuffle=True)
    val_loader = batches(val, tokenizer, args, shuffle=False)
    steps = max(1, math.ceil(len(train_loader) * args.epochs))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)

    history, step, started = [], 0, time.time()
    model.train()
    while step < steps:
        for batch in train_loader:
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 50 == 0 or step == steps:
                print(f"step {step}/{steps} loss {loss.item():.4f} ({time.time() - started:.0f}s)")
            if step % len(train_loader) == 0 or step == steps:
                metrics = evaluate(model, val_loader, device, args.threshold)
                history.append({"step": step, **{k: v for k, v in metrics.items() if k != "per_label_f1"}})
                print(f"  validation: micro-F1 {metrics['micro_f1']:.3f} macro-F1 {metrics['macro_f1']:.3f} "
                      f"stress-F1 {metrics['stress_f1']:.3f}")
            if step >= steps:
                break

    model.save_pretrained(out / "adapter")  # the LoRA weights alone (a few MB)
    merged = model.merge_and_unload()  # base + LoRA folded together: loads with plain transformers
    merged.save_pretrained(out / "merged")
    tokenizer.save_pretrained(out / "merged")
    meta = {"labels": GOEMOTIONS_LABELS, "distress": DISTRESS, "threshold": args.threshold, "base_model": args.base_model}
    (out / "merged" / "carebot_labels.json").write_text(json.dumps(meta, indent=2))
    final = evaluate(merged, val_loader, device, args.threshold)
    report = {"args": vars(args), "train_examples": len(train), "validation": final, "history": history}
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved to {out}. Final validation stress-F1 {final['stress_f1']:.3f}, micro-F1 {final['micro_f1']:.3f}.")
    print(f"Next: CAREBOT_FINETUNED_MODEL={out / 'merged'} carebot --detector finetuned eval detector")


if __name__ == "__main__":
    main()
