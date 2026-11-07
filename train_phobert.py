"""Fine-tune PhoBERT using the mean of chunk probabilities per document."""

import argparse
import faulthandler
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path


def configure_runtime_storage():
    runtime = Path(__file__).resolve().parent / ".tools"
    locations = {
        "HF_HOME": runtime / "huggingface",
        "HF_HUB_CACHE": runtime / "hf-cache",
        "HF_XET_CACHE": runtime / "huggingface" / "xet",
        "HF_ASSETS_CACHE": runtime / "huggingface" / "assets",
        "TORCH_HOME": runtime / "torch",
        "TORCHINDUCTOR_CACHE_DIR": runtime / "torchinductor",
        "TRITON_CACHE_DIR": runtime / "triton",
        "CUDA_CACHE_PATH": runtime / "cuda-cache",
        "TEMP": runtime / "tmp",
        "TMP": runtime / "tmp",
        "TMPDIR": runtime / "tmp",
    }
    for variable, directory in locations.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)


# Cache libraries read these settings during import.
configure_runtime_storage()

import pandas as pd
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaModel, RobertaPreTrainedModel, set_seed
from transformers.modeling_outputs import SequenceClassifierOutput


MODEL = "vinai/phobert-base"
LABELS = {0: "negative", 1: "neutral", 2: "positive"}
MAX_LENGTH = 256
SEED = 42


class PhoBERTClassifier(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids, attention_mask=None):
        encoded = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(encoded.last_hidden_state[:, 0]))
        return SequenceClassifierOutput(logits=logits)


def split_documents(path):
    data = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    data = data[["sentences", "label"]].copy()
    if not data.label.isin(LABELS).all() or not data.sentences.map(
        lambda text: isinstance(text, str) and bool(text.strip())
    ).all():
        raise ValueError("Expected non-empty sentences and integer labels 0, 1, 2.")
    data["sentences"] = data.sentences.map(lambda text: " ".join(text.split()))
    label_counts = data.groupby("sentences").label.transform("nunique")
    data = data.loc[label_counts.eq(1)].drop_duplicates("sentences")
    train, holdout = train_test_split(
        data, test_size=0.2, stratify=data.label, random_state=SEED
    )
    valid, test = train_test_split(
        holdout, test_size=0.5, stratify=holdout.label, random_state=SEED
    )
    return train, valid, test


def encode_document(text, tokenizer, overlap, max_length=MAX_LENGTH):
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
    window = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if not 0 <= overlap < window:
        raise ValueError(f"overlap must be between 0 and {window - 1}.")
    chunks = []
    for start in range(0, max(1, len(ids)), window - overlap):
        end = start + window
        chunks.append({"input_ids": tokenizer.build_inputs_with_special_tokens(ids[start:end])})
        if end >= len(ids):
            break
    return chunks


def make_loader(frame, tokenizer, overlap, batch_size, shuffle=False, max_length=MAX_LENGTH):
    documents = [
        (encode_document(text, tokenizer, overlap, max_length), int(label))
        for text, label in frame.itertuples(index=False, name=None)
    ]

    def collate(batch):
        chunks, owners, labels = [], [], []
        for index, (document_chunks, label) in enumerate(batch):
            chunks.extend(document_chunks)
            owners.extend([index] * len(document_chunks))
            labels.append(label)
        inputs = tokenizer.pad(
            chunks, padding=True, return_attention_mask=True, return_tensors="pt"
        )
        return inputs, torch.tensor(owners), torch.tensor(labels)

    return DataLoader(documents, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def document_probabilities(model, inputs, owners, num_documents):
    chunk_scores = model(**inputs).logits.float().softmax(dim=-1)
    totals = chunk_scores.new_zeros((num_documents, len(LABELS)))
    totals = totals.index_add(0, owners, chunk_scores)
    counts = torch.bincount(owners, minlength=num_documents)
    return totals / counts.unsqueeze(1)


def run_epoch(model, loader, device, optimizer=None, accumulation_steps=1):
    training = optimizer is not None
    model.train(training)
    actual, predicted = [], []
    total_loss = 0.0
    accumulated_documents = 0
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step, (inputs, owners, labels) in enumerate(loader, start=1):
            inputs = inputs.to(device)
            owners, labels = owners.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                probs = document_probabilities(model, inputs, owners, len(labels))
            loss = F.nll_loss(probs.clamp_min(1e-12).log(), labels)
            if training:
                (loss * len(labels)).backward()
                accumulated_documents += len(labels)
                if step % accumulation_steps == 0 or step == len(loader):
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(accumulated_documents)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_documents = 0
            total_loss += loss.item() * len(labels)
            actual.extend(labels.tolist())
            predicted.extend(probs.argmax(dim=-1).tolist())
    return total_loss / len(actual), actual, predicted


def pretrained_directory():
    tools_dir = Path(__file__).resolve().parent / ".tools"
    local_model = tools_dir / "phobert-base"
    required = ("pytorch_model.bin", "config.json", "vocab.txt", "bpe.codes")
    if all((local_model / name).is_file() for name in required):
        return local_model
    try:
        weights = hf_hub_download(
            MODEL, "pytorch_model.bin", cache_dir=tools_dir / "hf-cache", local_files_only=True
        )
        directory = Path(weights).parent
        if all((directory / name).is_file() for name in ("config.json", "vocab.txt", "bpe.codes")):
            return directory
    except LocalEntryNotFoundError:
        pass
    return Path(snapshot_download(
        MODEL,
        cache_dir=tools_dir / "hf-cache",
        allow_patterns=["config.json", "vocab.txt", "bpe.codes", "tokenizer*", "pytorch_model.bin"],
    ))


def replace_checkpoint(temporary, destination):
    for attempt in range(8):
        try:
            temporary.replace(destination)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            if attempt == 7:
                raise RuntimeError(
                    f"Cannot replace {destination}. The new checkpoint is still at "
                    f"{temporary}. Close programs reading the checkpoint before retrying."
                ) from error
            time.sleep(min(0.25 * 2**attempt, 2))


def prune_old_best_checkpoints(trial_dir, best_dir, completed_epoch):
    trial_dir = trial_dir.resolve()
    current_best = trial_dir / best_dir
    if not (current_best / "config.json").is_file() or not (
        current_best / "model.safetensors"
    ).is_file():
        return

    for candidate in trial_dir.iterdir():
        match = re.fullmatch(r"best_epoch_(\d+)", candidate.name)
        if (
            not match or candidate == current_best
            or int(match.group(1)) > completed_epoch
            or not candidate.is_dir() or candidate.resolve() != candidate
            or not (candidate / "model.safetensors").is_file()
        ):
            continue
        try:
            shutil.rmtree(candidate)
        except OSError as error:
            print(f"Could not remove old checkpoint {candidate}: {error}", flush=True)


def last_checkpoint_paths(trial_dir):
    numbered = [
        (int(match.group(1)), path)
        for path in trial_dir.glob("last_epoch_*.pt")
        if (match := re.fullmatch(r"last_epoch_(\d+)\.pt", path.name)) and path.is_file()
    ]
    paths = [path for _, path in sorted(numbered, reverse=True)]
    legacy = trial_dir / "last.pt"
    if legacy.is_file():
        paths.append(legacy)
    return paths


def train_trial(train, valid, tokenizer, config, checkpoint, device, pretrained):
    set_seed(SEED)
    overlap = config["overlap"]
    max_length = config["max_length"]
    micro_batch = config["micro_batch_size"]
    train_loader = make_loader(train, tokenizer, overlap, micro_batch, True, max_length)
    valid_loader = make_loader(valid, tokenizer, overlap, micro_batch, max_length=max_length)
    checkpoint.mkdir(parents=True, exist_ok=True)
    saved_paths = last_checkpoint_paths(checkpoint)
    state = torch.load(saved_paths[0], map_location="cpu", weights_only=True) if saved_paths else None
    if state:
        from transformers import AutoConfig

        model = PhoBERTClassifier(AutoConfig.from_pretrained(checkpoint / state["best_dir"]))
        model.load_state_dict(state["model"])
    else:
        model = PhoBERTClassifier.from_pretrained(
            pretrained, use_safetensors=False, num_labels=len(LABELS), id2label=LABELS,
            label2id={name: idx for idx, name in LABELS.items()},
        )
    model.to(device)
    if device.type == "cuda":
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    model.config.chunk_max_length = max_length
    model.config.chunk_overlap = overlap
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=0.01,
        fused=device.type == "cuda", foreach=False,
    )
    history = []
    best_score, best_epoch = -1.0, 0
    best_dir = None
    start_epoch = 1
    if state:
        # AdamW uses the saved group flags to decide where to restore step tensors.
        for group in state["optimizer"]["param_groups"]:
            group["fused"] = device.type == "cuda"
            group["foreach"] = False
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        best_score, best_epoch = state["best_score"], state["best_epoch"]
        best_dir = state["best_dir"]
        start_epoch = state["epoch"] + 1
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        del state
        prune_old_best_checkpoints(checkpoint, best_dir, start_epoch - 1)
        print(f"{checkpoint.name}: resuming at epoch {start_epoch}", flush=True)
    accumulation = config["batch_size"] // micro_batch
    for epoch in range(start_epoch, config["epochs"] + 1):
        if config["patience"] and best_epoch and epoch - 1 - best_epoch >= config["patience"]:
            break
        train_loss, _, _ = run_epoch(model, train_loader, device, optimizer, accumulation)
        valid_loss, actual, predicted = run_epoch(model, valid_loader, device)
        score = macro_f1(actual, predicted)
        metrics = dict(epoch=epoch, train_loss=train_loss, val_loss=valid_loss, val_macro_f1=score)
        history.append(metrics)
        print(checkpoint.name, metrics, flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_dir = f"best_epoch_{epoch:02d}"
            model.save_pretrained(checkpoint / best_dir)
            tokenizer.save_pretrained(checkpoint / best_dir)
        last_path = checkpoint / f"last_epoch_{epoch:04d}.pt"
        temporary = last_path.with_suffix(".tmp")
        print(f"Saving resume checkpoint for epoch {epoch} to {temporary} ...", flush=True)
        torch.save({
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "history": history, "best_score": best_score, "best_epoch": best_epoch,
            "best_dir": best_dir, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }, temporary)
        replace_checkpoint(temporary, last_path)
        for previous in last_checkpoint_paths(checkpoint):
            if previous != last_path:
                try:
                    previous.unlink()
                except OSError as error:
                    print(f"Could not remove older resume checkpoint {previous}: {error}", flush=True)
        prune_old_best_checkpoints(checkpoint, best_dir, epoch)
        history_path = checkpoint / "history.csv"
        temporary_history = history_path.with_suffix(".tmp")
        pd.DataFrame(history).to_csv(temporary_history, index=False)
        replace_checkpoint(temporary_history, history_path)
        print(f"Saved {last_path} (epoch {epoch})", flush=True)

    pd.DataFrame(history).to_csv(checkpoint / "history.csv", index=False)
    return {**config, "checkpoint": f"{checkpoint.name}/{best_dir}", "best_epoch": best_epoch,
            "val_macro_f1": best_score}


def macro_f1(actual, predicted):
    return classification_report(
        actual, predicted, labels=list(LABELS), output_dict=True, zero_division=0
    )["macro avg"]["f1-score"]


def tune(args, train, valid, device, dataset_hash):
    config = {key: getattr(args, key) for key in (
        "learning_rate", "batch_size", "micro_batch_size", "epochs",
        "max_length", "overlap", "patience"
    )}
    alternatives = [
        {"learning_rate": 3e-5 if args.learning_rate != 3e-5 else 2e-5},
        {"batch_size": 8 if args.batch_size != 8 else 16},
        {"max_length": 128 if args.max_length == 256 else 256},
        {"epochs": 5 if args.epochs != 5 else 3},
    ]
    configs = [config]
    if args.sweep:
        configs.extend({**config, **change} for change in alternatives)

    if any(trial["batch_size"] % trial["micro_batch_size"] for trial in configs):
        raise ValueError("Effective batch size must be divisible by micro batch size.")
    manifest_path = args.output / "run.json"
    manifest = {"model": MODEL, "seed": SEED, "dataset_sha256": dataset_hash, "trials": configs}
    if args.resume:
        if not manifest_path.is_file():
            raise SystemExit(
                f"Cannot resume: {manifest_path} is missing. "
                "Start a new run without --resume, using a new output directory "
                "if the current directory already exists."
            )
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        differences = [
            key for key in manifest
            if key != "trials" and saved_manifest.get(key) != manifest[key]
        ]
        saved_configs = saved_manifest.get("trials", [])
        if len(saved_configs) != len(configs) or any(
            {key: value for key, value in saved.items() if key != "patience"}
            != {key: value for key, value in current.items() if key != "patience"}
            for saved, current in zip(saved_configs, configs)
        ):
            differences.append("trials")
        if differences:
            raise SystemExit(
                f"Cannot resume {args.output}: changed {', '.join(differences)}. "
                "Use the original CSV and training options."
            )
        if saved_configs != configs:
            temporary = manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            replace_checkpoint(temporary, manifest_path)
            print(f"Resume early-stopping patience: {args.patience}", flush=True)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    pretrained = pretrained_directory()
    tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False)
    results = []
    for index, trial in enumerate(configs, start=1):
        results.append(train_trial(train, valid, tokenizer, trial,
                                   args.output / f"trial_{index:02d}", device, pretrained))
        pd.DataFrame(results).to_csv(args.output / "validation_trials.csv", index=False)

    best = max(results, key=lambda result: result["val_macro_f1"])
    (args.output / "selection.json").write_text(
        json.dumps({"dataset_sha256": dataset_hash, "best": best}, indent=2), encoding="utf-8"
    )
    print(f"Selected on validation: {best['checkpoint']}", flush=True)


if __name__ == "__main__":
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["tune", "evaluate"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("phobert_sentiment"))
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--max-length", type=int, choices=[128, 256], default=256)
    parser.add_argument("--batch-size", type=int, default=16, help="Effective documents per update")
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=2, help="0 disables early stopping")
    parser.add_argument("--sweep", action="store_true", help="Compare four one-factor variants")
    parser.add_argument("--resume", action="store_true", help="Resume tuning from the last saved epoch")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.micro_batch_size, args.learning_rate) <= 0:
        parser.error("Epochs, batch sizes and learning rate must be positive.")
    shortest = 128 if args.sweep else args.max_length
    if args.patience < 0 or not 0 <= args.overlap < shortest - 2:
        parser.error("Patience must be nonnegative and overlap smaller than chunk content length.")
    train, valid, test = split_documents(args.input)
    dataset_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Device: {device} ({device_name}); train={len(train):,}, validation={len(valid):,}, test={len(test):,}", flush=True)
    if args.mode == "tune":
        tune(args, train, valid, device, dataset_hash)
    else:
        from evaluate import evaluate

        evaluate(args, train, test, device, dataset_hash)
