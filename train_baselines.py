"""TF-IDF + Logistic Regression baseline on the same split as PhoBERT."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

from train_phobert import LABELS, SEED, macro_f1, split_documents


MODELS = {
    "tfidf_logistic": {
        "parameter": "C",
        "grid": (0.1, 1.0, 10.0, 100.0),
        "classifier": lambda value: LogisticRegression(C=value, max_iter=1000, random_state=SEED),
    },
}


def build_model(name, value):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=60000,
                                 tokenizer=str.split, token_pattern=None, lowercase=False)
    return make_pipeline(vectorizer, MODELS[name]["classifier"](value))


def tune(args, train, valid, dataset_hash):
    args.output.mkdir(parents=True, exist_ok=True)
    trials, best = [], {}
    for name, spec in MODELS.items():
        scores = {}
        for value in spec["grid"]:
            model = build_model(name, value).fit(train.sentences, train.label)
            scores[value] = macro_f1(valid.label, model.predict(valid.sentences))
            trials.append({"model": name, "parameter": spec["parameter"], "value": value,
                           "val_macro_f1": scores[value]})
            print(f"{name} {spec['parameter']}={value}: val_macro_f1={scores[value]:.4f}", flush=True)
        best[name] = max(scores, key=scores.get)

    pd.DataFrame(trials).to_csv(args.output / "validation_trials.csv", index=False)
    (args.output / "selection.json").write_text(
        json.dumps({"dataset_sha256": dataset_hash, "best": best}, indent=2), encoding="utf-8"
    )
    print(f"Selected on validation: {best}", flush=True)


def evaluate(args, train, test, dataset_hash):
    selection = json.loads((args.output / "selection.json").read_text(encoding="utf-8"))
    if selection["dataset_sha256"] != dataset_hash:
        raise ValueError("Test evaluation requires the same CSV used for tuning.")

    comparison = []
    for name, value in selection["best"].items():
        model = build_model(name, value).fit(train.sentences, train.label)
        predicted = model.predict(test.sentences)
        report = classification_report(
            test.label, predicted, labels=list(LABELS), target_names=list(LABELS.values()),
            output_dict=True, zero_division=0
        )
        macro = report["macro avg"]
        comparison.append({
            "model": name, "accuracy": accuracy_score(test.label, predicted),
            "precision_macro": macro["precision"], "recall_macro": macro["recall"],
            "f1_weighted": report["weighted avg"]["f1-score"], "macro_f1": macro["f1-score"],
        })
        matrix = confusion_matrix(test.label, predicted, labels=list(LABELS))
        pd.DataFrame(matrix, index=list(LABELS.values()), columns=list(LABELS.values())).to_csv(
            args.output / f"confusion_{name}.csv", index_label="true_label"
        )
        pd.DataFrame(report).T.to_csv(args.output / f"report_{name}.csv")
    summary = pd.DataFrame(comparison).sort_values("macro_f1", ascending=False)
    summary.to_csv(args.output / "test_comparison.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["tune", "evaluate"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("baseline_sentiment"))
    args = parser.parse_args()
    train, valid, test = split_documents(args.input)
    dataset_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    print(f"train={len(train):,}, validation={len(valid):,}, test={len(test):,}", flush=True)
    if args.mode == "tune":
        tune(args, train, valid, dataset_hash)
    else:
        evaluate(args, train, test, dataset_hash)
