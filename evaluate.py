"""Evaluate the selected PhoBERT checkpoint and TF-IDF baseline on test data."""

import argparse
import hashlib
import json
from pathlib import Path

from train_phobert import (
    LABELS, SEED, PhoBERTClassifier, make_loader, document_probabilities, split_documents,
)

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from transformers import AutoTokenizer
from train_baselines import build_model, MODELS
from plot_learning_curves import plot_curves


def predict_phobert(model, loader, device):
    model.eval()
    probabilities = []
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.inference_mode():
        for step, (inputs, owners, labels) in enumerate(loader, start=1):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                scores = document_probabilities(model, inputs.to(device), owners.to(device), len(labels))
            probabilities.extend(scores.cpu().tolist())
            if step % 250 == 0:
                print(f"Test inference: {len(probabilities):,} documents", flush=True)
    return probabilities


def draw_confusion(matrix, name, output):
    from matplotlib import pyplot as plt

    normalized = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, values, title, percentage in zip(
        axes, (matrix, normalized), ("Counts", "Normalized by true class"), (False, True)
    ):
        axis.imshow(values, cmap="Blues", vmin=0, vmax=1 if percentage else max(1, matrix.max()))
        axis.set(xticks=range(3), yticks=range(3), xticklabels=list(LABELS.values()),
                 yticklabels=list(LABELS.values()), xlabel="Predicted", ylabel="True", title=title)
        for row in range(3):
            for col in range(3):
                value = values[row, col]
                text = f"{value:.1%}" if percentage else str(value)
                axis.text(col, row, text, ha="center", va="center",
                          color="white" if value > values.max() / 2 else "black")
    fig.suptitle(name)
    fig.savefig(output / f"confusion_{name}.png", dpi=180)
    plt.close(fig)


def markdown_table(frame):
    values = frame.copy()
    for column in values.select_dtypes(include="float"):
        values[column] = values[column].map(lambda value: f"{value:.4f}")
    lines = ["| " + " | ".join(values.columns) + " |", "| " + " | ".join(["---"] * len(values.columns)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in values.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_report(summary, output, best, baseline, train, valid, test, history_path):
    history = pd.read_csv(history_path)
    models = summary.set_index("model")
    pho, lr = models.loc["phobert"], models.loc["tfidf_logistic"]
    gap = (pho.macro_f1 - lr.macro_f1) * 100
    sections = [
        "# Đánh giá mô hình phân loại cảm xúc tiếng Việt",
        "## Thiết lập thực nghiệm",
        f"Dữ liệu được chia ở cấp văn bản với seed {SEED}, phân tầng theo nhãn. "
        f"Tập train có {len(train):,} văn bản, validation có {len(valid):,} và test có {len(test):,}. "
        "Văn bản trùng được xử lý trước khi chia. Các mô hình dùng cùng tập test và cùng nhãn "
        "0 = tiêu cực, 1 = trung lập, 2 = tích cực. Cột label là mục tiêu huấn luyện.",
        f"PhoBERT dùng vinai/phobert-base và head Dropout → Linear. "
        f"Max length = {best['max_length']}, overlap = {best['overlap']}, "
        f"learning rate = {best['learning_rate']}, batch hiệu dụng = {best['batch_size']}, "
        "AdamW và weight decay = 0.01. Xác suất từng lớp của văn bản là trung bình "
        "xác suất softmax qua các chunk. Loss huấn luyện là negative log-likelihood ở cấp văn bản.",
        f"Đã hoàn tất {int(history.epoch.max())} epoch. Checkpoint đánh giá là epoch {best['best_epoch']}, "
        f"được chọn theo validation macro-F1 = {best['val_macro_f1']:.4f}.",
        "Baseline dùng TF-IDF unigram và bigram, tối đa 60.000 đặc trưng. "
        f"Logistic Regression thử C ∈ {{0.1, 1, 10, 100}} trên validation, chọn C = {baseline['C']} "
        f"với validation macro-F1 = {baseline['val_macro_f1']:.4f}. Vectorizer được fit trên train. "
        "Tập test được dùng để đánh giá cấu hình đã chọn.",
        "## Phân bố nhãn test",
        markdown_table(test.label.value_counts().sort_index().rename_axis("label").reset_index(name="support")),
        "## So sánh kết quả test",
        markdown_table(summary),
        "Precision và recall trong bảng là macro average. F1 weighted được cân theo số mẫu từng lớp. "
        "Macro-F1 là trung bình F1 không trọng số của ba lớp và là tiêu chí so sánh chính. "
        "Error rate là tỷ lệ dự đoán sai. Log loss đo xác suất gán cho lớp đúng, giá trị thấp hơn tốt hơn.",
        f"PhoBERT đạt accuracy {pho.accuracy:.2%} và macro-F1 {pho.macro_f1:.4f}. "
        f"TF-IDF + Logistic Regression đạt accuracy {lr.accuracy:.2%} và macro-F1 {lr.macro_f1:.4f}. "
        f"Chênh lệch macro-F1 (PhoBERT − baseline) là {gap:+.2f} điểm phần trăm. "
        "Kết quả này mô tả một lần chia dữ liệu, chưa có kiểm định ý nghĩa thống kê.",
        "## Kết quả từng lớp và phân tích lỗi",
    ]
    for name in ("phobert", "tfidf_logistic"):
        report = pd.read_csv(output / f"report_{name}.csv", index_col=0)
        matrix = pd.read_csv(output / f"confusion_{name}.csv", index_col=0)
        report = report.loc[list(LABELS.values()), ["precision", "recall", "f1-score", "support"]]
        report["support"] = report.support.astype(int)
        sections.extend([f"### {name}", markdown_table(report.reset_index(names="class")),
                         f"![Confusion matrix {name}](confusion_{name}.png)"])
        mistakes = sorted(((int(matrix.iloc[row, col]), LABELS[row], LABELS[col])
                           for row in range(3) for col in range(3) if row != col), reverse=True)
        for count, actual, predicted in mistakes[:3]:
            sections.append(f"- Nhãn thật {actual} bị dự đoán thành {predicted} ở {count} văn bản.")
        errors = pd.read_csv(output / f"errors_{name}.csv")
        sections.append(f"Tổng cộng {len(errors)} văn bản sai. Danh sách đầy đủ và điểm xác suất "
                        f"của ba lớp nằm trong errors_{name}.csv, được sắp theo độ tự tin giảm dần. "
                        "Cần đọc từng văn bản để kết luận nguyên nhân ngôn ngữ của lỗi.")
        confident_errors = int(errors.confidence.ge(0.9).sum())
        sections.append(f"Trong các dự đoán sai, {confident_errors} trường hợp có xác suất "
                        "của lớp được chọn từ 0.9 trở lên. Đây là nhóm lỗi cần ưu tiên rà soát.")
    sections.append(
        f"Log loss test của PhoBERT là {pho.log_loss:.4f}, cao hơn baseline {lr.log_loss:.4f}. "
        "Vì vậy, ưu thế về macro-F1 cần được trình bày cùng hạn chế về chất lượng xác suất "
        "dự đoán. Lỗi có độ tự tin cao bị phạt mạnh trong log loss."
    )
    paired = pd.read_csv(output / "paired_predictions.csv")
    pho_ok = paired.predicted_phobert.eq(paired.true_label)
    lr_ok = paired.predicted_tfidf_logistic.eq(paired.true_label)
    sections.extend([
        "## Đối chiếu dự đoán trên từng văn bản",
        f"Cả hai đúng ở {int((pho_ok & lr_ok).sum())} văn bản, cả hai sai ở "
        f"{int((~pho_ok & ~lr_ok).sum())} văn bản. PhoBERT đúng và baseline sai ở "
        f"{int((pho_ok & ~lr_ok).sum())} văn bản, baseline đúng và PhoBERT sai ở "
        f"{int((~pho_ok & lr_ok).sum())} văn bản. Chi tiết nằm trong paired_predictions.csv.",
        "## Learning curves",
        "![Learning curves](learning_curves.png)",
        f"Ở epoch cuối, train loss = {history.iloc[-1].train_loss:.4f}, "
        f"validation loss = {history.iloc[-1].val_loss:.4f} và validation macro-F1 = "
        f"{history.iloc[-1].val_macro_f1:.4f}. Best checkpoint được chọn bằng validation.",
        "Khi train loss giảm trong khi validation loss tăng, đó là dấu hiệu overfitting. "
        f"Macro-F1 validation tốt nhất ở epoch {best['best_epoch']}, "
        f"với giá trị {best['val_macro_f1']:.4f}, so với "
        f"{history.iloc[-1].val_macro_f1:.4f} ở epoch {int(history.iloc[-1].epoch)}. "
        "Báo cáo test sử dụng checkpoint tốt nhất theo validation.",
        "## Giới hạn kết quả",
        "Thực nghiệm sử dụng tập sampled tương ứng với checkpoint. Nhãn đầu vào được kế thừa "
        "từ dataset đã chuẩn bị. Chưa thực hiện đánh giá riêng theo miền dữ liệu hoặc kiểm tra "
        "độc lập chất lượng nhãn. Hash CSV, chỉ số dòng của từng tập và cấu hình lựa chọn "
        "được lưu trong evaluation_manifest.json và các tệp *_indices.csv để tái lập kết quả.",
    ])
    (output / "model_evaluation_report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def evaluate(args, train, test, device, dataset_hash):
    selection = json.loads((args.output / "selection.json").read_text(encoding="utf-8"))
    if selection["dataset_sha256"] != dataset_hash:
        raise ValueError("Test evaluation requires the same CSV used for tuning.")
    best = selection["best"]
    checkpoint = args.output / best["checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    model = PhoBERTClassifier.from_pretrained(checkpoint).to(device)
    loader = make_loader(test, tokenizer, model.config.chunk_overlap,
                         args.micro_batch_size, max_length=model.config.chunk_max_length)
    phobert = predict_phobert(model, loader, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    valid = split_documents(args.input)[1]
    trials = []
    for value in MODELS["tfidf_logistic"]["grid"]:
        candidate = build_model("tfidf_logistic", value).fit(train.sentences, train.label)
        score = classification_report(valid.label, candidate.predict(valid.sentences),
                                      labels=list(LABELS), output_dict=True, zero_division=0)["macro avg"]["f1-score"]
        trials.append({"C": value, "val_macro_f1": score})
        print(f"Logistic Regression: C={value}, validation macro-F1={score:.6f}", flush=True)
    baseline_best = max(trials, key=lambda trial: trial["val_macro_f1"])
    logistic = build_model("tfidf_logistic", baseline_best["C"]).fit(train.sentences, train.label)
    probabilities = {
        "tfidf_logistic": logistic.predict_proba(test.sentences),
        "phobert": phobert,
    }
    results_dir = args.output / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trials).to_csv(results_dir / "baseline_validation_trials.csv", index=False)
    (results_dir / "evaluation_manifest.json").write_text(json.dumps({
        "dataset_sha256": dataset_hash, "seed": SEED,
        "sizes": {"train": len(train), "validation": len(valid), "test": len(test)},
        "phobert_selection": best, "baseline_selection": baseline_best,
        "labels": LABELS, "probability_columns_used_for_training": False,
        "test_used_for_hyperparameter_selection": False,
    }, indent=2), encoding="utf-8")
    for split, frame in (("train", train), ("validation", valid), ("test", test)):
        pd.DataFrame({"source_row_index": frame.index, "label": frame.label.to_numpy()}).to_csv(
            results_dir / f"{split}_indices.csv", index=False)
    actual = test.label.to_numpy()
    comparison = []
    all_predictions = test.rename(columns={"label": "true_label"}).copy()
    all_predictions.insert(0, "source_row_index", test.index)
    for name, scores in probabilities.items():
        scores = torch.as_tensor(scores).numpy()
        predicted = scores.argmax(axis=1)
        report = classification_report(
            actual, predicted, labels=list(LABELS), target_names=list(LABELS.values()),
            output_dict=True, zero_division=0
        )
        macro = report["macro avg"]
        comparison.append({
            "model": name, "accuracy": accuracy_score(actual, predicted),
            "precision_macro": macro["precision"], "recall_macro": macro["recall"],
            "f1_weighted": report["weighted avg"]["f1-score"], "macro_f1": macro["f1-score"],
            "log_loss": log_loss(actual, scores, labels=list(LABELS)),
            "errors": int((actual != predicted).sum()), "test_size": len(test),
            "error_rate": float((actual != predicted).mean()),
        })
        matrix = confusion_matrix(actual, predicted, labels=list(LABELS))
        pd.DataFrame(matrix, index=list(LABELS.values()), columns=list(LABELS.values())).to_csv(
            results_dir / f"confusion_{name}.csv", index_label="true_label"
        )
        pd.DataFrame(report).T.to_csv(results_dir / f"report_{name}.csv")
        per_document = all_predictions[["source_row_index", "sentences", "true_label"]].copy()
        per_document["predicted_label"] = predicted
        per_document["confidence"] = scores.max(axis=1)
        for label, label_name in LABELS.items():
            per_document[f"prob_{label_name}"] = scores[:, label]
        per_document.to_csv(results_dir / f"predictions_{name}.csv", index=False, encoding="utf-8-sig")
        per_document.loc[actual != predicted].sort_values("confidence", ascending=False).to_csv(
            results_dir / f"errors_{name}.csv", index=False, encoding="utf-8-sig")
        all_predictions[f"predicted_{name}"] = predicted
        draw_confusion(matrix, name, results_dir)
    summary = pd.DataFrame(comparison).sort_values("macro_f1", ascending=False)
    summary.to_csv(results_dir / "test_comparison.csv", index=False)
    all_predictions.to_csv(results_dir / "paired_predictions.csv", index=False, encoding="utf-8-sig")
    history_path = checkpoint.parent / "history.csv"
    plot_curves(history_path, results_dir / "learning_curves.png")
    write_report(summary, results_dir, best, baseline_best, train, valid, test, history_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("phobert_sentiment"),
                        help="Training run directory containing selection.json")
    parser.add_argument("--micro-batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.micro_batch_size < 1:
        parser.error("--micro-batch-size must be positive.")
    train, _, test = split_documents(args.input)
    dataset_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, test documents: {len(test):,}", flush=True)
    evaluate(args, train, test, device, dataset_hash)
