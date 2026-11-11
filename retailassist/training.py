import hashlib
import json
import re
import time
from pathlib import Path
from .config import SETTINGS, DATASET, BASE_MODEL, SENTIMENT_MODEL, EMBEDDING_MODEL, INTENTS, local_model_path

def normalize(text): return re.sub(r"\s+", " ", str(text)).strip()
def fingerprint(text): return normalize(text).casefold()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))

def download_models(settings=SETTINGS):
    from huggingface_hub import HfApi, snapshot_download
    manifest = {}
    for name in [BASE_MODEL, SENTIMENT_MODEL, EMBEDDING_MODEL]:
        info = HfApi().model_info(name)
        names = [x.rfilename for x in info.siblings]
        weight = "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
        allowed = [x for x in names if x == weight or x.endswith((".json", ".txt", ".model")) or x == "README.md"]
        # Exclude alternate runtimes and training checkpoints; retain SentenceTransformer pooling config.
        allowed = [x for x in allowed if not x.startswith(("onnx/", "openvino/", "flax/"))]
        snapshot_download(name, revision=info.sha, allow_patterns=allowed)
        manifest[name] = {"revision": info.sha, "weight_file": weight}
        dump(settings.artifacts / "model_downloads.json", manifest)
    return manifest

def prepare_data(settings=SETTINGS):
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    info = HfApi().dataset_info(DATASET)
    filename = next(x.rfilename for x in info.siblings if x.rfilename.endswith(".csv"))
    raw = Path(hf_hub_download(DATASET, filename, repo_type="dataset", revision=info.sha))
    frame = pd.read_csv(raw)[["instruction", "intent"]].dropna().rename(columns={"instruction": "text"})
    original_count = len(frame)
    frame["text"] = frame["text"].map(normalize)
    frame = frame[frame.text.str.len() > 0]
    frame["key"] = frame.text.map(fingerprint)
    conflicts = frame.groupby("key").intent.nunique()
    bad = set(conflicts[conflicts > 1].index)
    frame = frame[~frame.key.isin(bad)].drop_duplicates("key").reset_index(drop=True)
    labels = sorted(frame.intent.unique())
    if labels != INTENTS:
        raise ValueError(f"Dataset labels changed; inspect before training: {labels}")
    train, remainder = train_test_split(frame, test_size=.2, random_state=42, stratify=frame.intent)
    validation, test = train_test_split(remainder, test_size=.5, random_state=42, stratify=remainder.intent)
    target = settings.root / "data/processed"
    target.mkdir(parents=True, exist_ok=True)
    split_info = {}
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        part[["text", "intent"]].to_json(target / f"{name}.jsonl", orient="records", lines=True)
        split_info[name] = {"rows": len(part), "sha256": hashlib.sha256((target/f"{name}.jsonl").read_bytes()).hexdigest()}
    # Bounded cross-split audit: character n-grams catch template-like matches beyond exact duplicates.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=12000)
    vectors = vectorizer.fit_transform(train.text)
    sample = test.sample(min(500, len(test)), random_state=42)
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(vectors)
    distance, _ = neighbors.kneighbors(vectorizer.transform(sample.text))
    audit = {"sampled_test_rows": len(sample), "similarity_threshold": .95,
             "near_duplicate_count": int((distance[:, 0] <= .05).sum()),
             "limitation": "Hybrid synthetic templates may still inflate public test performance; independently authored workplace tickets are evaluated separately."}
    manifest = {"dataset": DATASET, "revision": info.sha, "filename": filename,
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "license": "CDLA-Sharing-1.0",
        "original_rows": original_count, "deduplicated_rows": len(frame), "conflicting_keys_removed": len(bad),
        "labels": labels, "seed": 42, "split": split_info, "near_duplicate_audit": audit,
        "features": "instruction only; supplied response and category are excluded"}
    dump(target / "manifest.json", manifest)
    return manifest

def score_labels(gold, predicted, labels):
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    return {"accuracy": float(accuracy_score(gold, predicted)),
            "macro_f1": float(f1_score(gold, predicted, labels=labels, average="macro", zero_division=0)),
            "per_class": classification_report(gold, predicted, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(gold, predicted, labels=labels).tolist()}

def train(settings=SETTINGS, epochs=3, smoke=False):
    import numpy as np
    import pandas as pd
    import torch
    from datasets import Dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import f1_score, accuracy_score
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                              Trainer, DataCollatorWithPadding, set_seed)
    set_seed(42)
    torch.set_num_threads(6)
    processed = settings.root / "data/processed"
    frames = {name: pd.read_json(processed / f"{name}.jsonl", lines=True) for name in ["train", "validation", "test"]}
    if smoke:
        frames = {name: frame.groupby("intent", group_keys=False).head(2) for name, frame in frames.items()}
    labels = INTENTS
    label2id = {label: i for i, label in enumerate(labels)}
    start = time.perf_counter()
    baseline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=500, random_state=42))
    baseline.fit(frames["train"].text, frames["train"].intent)
    baseline_metrics = score_labels(frames["test"].intent, baseline.predict(frames["test"].text), labels)
    base_path = local_model_path(BASE_MODEL, settings)
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    def convert(frame):
        values = {"text": frame.text.tolist(), "labels": [label2id[x] for x in frame.intent]}
        dataset = Dataset.from_dict(values)
        return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=128), batched=True, remove_columns=["text"])
    tokenized = {k: convert(v) for k, v in frames.items()}
    model = AutoModelForSequenceClassification.from_pretrained(base_path, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=label2id, local_files_only=True)
    output = settings.artifacts / "models" / ("smoke-intent" if smoke else "intent")
    args = TrainingArguments(output_dir=str(output / "checkpoints"), num_train_epochs=epochs,
        per_device_train_batch_size=16, per_device_eval_batch_size=32, learning_rate=2e-5,
        weight_decay=.01, warmup_ratio=.1, eval_strategy="epoch", save_strategy="epoch",
        logging_strategy="steps", logging_steps=100 if not smoke else 1, save_total_limit=1,
        save_only_model=True, load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, report_to="none", seed=42, data_seed=42, dataloader_num_workers=0,
        fp16=False, bf16=False, use_cpu=not torch.backends.mps.is_available(), disable_tqdm=True)
    def metrics(prediction):
        logits, gold = prediction
        predicted = np.argmax(logits, axis=-1)
        return {"macro_f1": f1_score(gold, predicted, average="macro"), "accuracy": accuracy_score(gold, predicted)}
    trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)
    print(f"Training on {args.device}; rows={len(frames['train'])}; epochs={epochs}", flush=True)
    trainer.train()
    predictions = trainer.predict(tokenized["test"])
    predicted_ids = predictions.predictions.argmax(-1)
    fine_metrics = score_labels(frames["test"].intent, [labels[i] for i in predicted_ids], labels)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    trainer.state.save_to_json(str(output / "trainer_state.json"))
    config = {"base_model": BASE_MODEL, "labels": labels, "epochs": epochs, "batch_size": 16,
              "max_length": 128, "learning_rate": 2e-5, "weight_decay": .01, "seed": 42,
              "device": str(args.device), "smoke_only": smoke,
              "selected_checkpoint": trainer.state.best_model_checkpoint,
              "dataset_manifest": json.loads((processed / "manifest.json").read_text())}
    dump(output / "training_manifest.json", config)
    results = {"baseline": baseline_metrics, "fine_tuned": fine_metrics, "labels": labels,
               "training": config, "duration_seconds": round(time.perf_counter()-start, 2),
               "log_history": trainer.state.log_history}
    name = "smoke_training_metrics.json" if smoke else "training_metrics.json"
    dump(settings.evaluation / name, results)
    if not smoke: training_plots(results, settings.evaluation)
    return {"device": str(args.device), "baseline_macro_f1": baseline_metrics["macro_f1"],
            "fine_tuned_macro_f1": fine_metrics["macro_f1"], "checkpoint": str(output)}

def training_plots(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    history = results["log_history"]
    fig, ax = plt.subplots(figsize=(8, 4))
    losses = [x for x in history if "loss" in x]
    vals = [x for x in history if "eval_loss" in x]
    ax.plot([x["epoch"] for x in losses], [x["loss"] for x in losses], label="Training loss")
    ax.plot([x["epoch"] for x in vals], [x["eval_loss"] for x in vals], "o-", label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Actual DistilBERT fine-tuning")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "training_curve.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 10))
    matrix = np.array(results["fine_tuned"]["confusion_matrix"])
    image = ax.imshow(matrix, cmap="Greens")
    ax.set_xticks(range(len(INTENTS)), INTENTS, rotation=90, fontsize=7)
    ax.set_yticks(range(len(INTENTS)), INTENTS, fontsize=7)
    ax.set(xlabel="Predicted intent", ylabel="Actual intent", title="Held-out public test set")
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(output / "confusion_matrix.png", dpi=160); plt.close(fig)
