"""Vietnamese word segmentation with the local VnCoreNLP installation."""

import argparse
import csv
import os
from functools import lru_cache
from pathlib import Path

from prepare_data import clean_text


ROOT = Path(__file__).resolve().parent
_DLL_HANDLES = []


@lru_cache(maxsize=1)
def get_segmenter():
    java_home = ROOT / ".tools" / "java"
    model_dir = ROOT / ".tools" / "vncorenlp"
    if not (java_home / "bin" / "java.exe").exists():
        raise RuntimeError("Local Java is missing. Run setup_vncorenlp.ps1 first.")
    os.environ["JAVA_HOME"] = str(java_home)
    os.environ["PATH"] = str(java_home / "bin") + os.pathsep + os.environ["PATH"]
    if os.name == "nt":
        for directory in [java_home / "bin", java_home / "bin" / "server"]:
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
    import py_vncorenlp

    previous_dir = Path.cwd()
    try:
        return py_vncorenlp.VnCoreNLP(
            annotators=["wseg"], save_dir=model_dir.as_posix())
    finally:
        # The upstream wrapper changes cwd when loading models.
        os.chdir(previous_dir)


def preprocess_text(text):
    text = clean_text(text)
    if not text:
        return ""
    return " ".join(get_segmenter().word_segment(text))


def preprocess_csv(input_path, output_path):
    input_path, output_path = input_path.resolve(), output_path.resolve()
    if input_path == output_path:
        raise ValueError("Choose a separate output file to preserve source text.")
    with input_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["sentences", "label"]:
            raise ValueError("Expected exactly the columns sentences,label")
        get_segmenter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            count = 0
            for row in reader:
                row["sentences"] = preprocess_text(row["sentences"])
                writer.writerow(row)
                count += 1
                if count % 1000 == 0:
                    print(f"Processed {count:,} records", flush=True)
    print(f"Saved {count:,} records to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="Tôi rất hài lòng với chất lượng sản phẩm.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.input:
        if not args.output:
            parser.error("--input requires --output")
        preprocess_csv(args.input, args.output)
    else:
        if args.output:
            parser.error("--output requires --input")
        print(preprocess_text(args.text))
