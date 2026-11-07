"""Plot training and validation metrics recorded by train_phobert.py."""

import argparse
import os
from pathlib import Path

cache = Path(__file__).resolve().parent / ".tools" / "matplotlib"
cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_curves(history_path, output_path):
    if not history_path.is_file():
        raise SystemExit(f"History not available yet: {history_path}")
    history = pd.read_csv(history_path)
    epoch = history["epoch"]
    best = history.loc[history["val_macro_f1"].idxmax()]
    fig, (loss, score) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    loss.plot(epoch, history["train_loss"], marker=".", label="Train")
    loss.plot(epoch, history["val_loss"], marker=".", label="Validation")
    loss.set(ylabel="Loss", title="Learning curves")
    loss.legend()
    score.plot(epoch, history["val_macro_f1"], marker=".")
    score.scatter(best["epoch"], best["val_macro_f1"], color="tab:red", label="Best epoch")
    score.set(ylabel="Macro-F1", title="Validation")
    score.legend()
    for axis in (loss, score):
        axis.set_xlabel("Epoch")
        axis.xaxis.get_major_locator().set_params(integer=True)
        axis.grid(alpha=0.25)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot_curves(args.history, args.output or args.history.with_name("learning_curves.png"))
