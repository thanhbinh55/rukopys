#!/usr/bin/env python3
"""Stage baseline LoRA and fixed gold split for resumable V2 training."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


ROOT = Path("/kaggle/working/rukopys")
KINPUT = Path("/kaggle/input")
ART = ROOT / "local_working" / "htr_artifacts"
OUTPUT = ART / "qwen3_lora"


def has_raw_data(path: Path) -> bool:
    return (
        (path / "train" / "metadata.jsonl").exists()
        and (path / "train" / "images").exists()
    )


def find_raw_root() -> Path:
    candidates = [
        Path("/kaggle/input/datasets/bnthanh/rukopys-dataset/rukopys_raw"),
        Path("/kaggle/input/datasets/bnthanh/rukopys-dataset"),
        KINPUT / "rukopys-dataset" / "rukopys_raw",
        KINPUT / "rukopys-dataset",
    ]
    for path in candidates:
        if has_raw_data(path):
            return path
    for path in KINPUT.glob("**/rukopys_raw"):
        if has_raw_data(path):
            return path
    raise FileNotFoundError("Attach bnthanh/rukopys-dataset")


def find_recognizer_root() -> Path:
    candidates = [
        Path("/kaggle/input/datasets/ngovietan/htr-02-train-recognizer"),
        KINPUT / "htr-02-train-recognizer",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(KINPUT.glob("**/htr-02-train-recognizer"))
    if matches:
        return matches[0]
    raise FileNotFoundError("Attach ngovietan/htr-02-train-recognizer")


def find_first(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = list(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"copied {source} -> {destination}")


def copy_dir(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    print(f"copied {source} -> {destination}")


def checkpoint_step(path: Path) -> int:
    tail = path.name.split("-")[-1]
    return int(tail) if tail.isdigit() else -1


def resume_root_from_env() -> Path | None:
    explicit = os.getenv("RESUME_TRAINING_INPUT", "").strip()
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"RESUME_TRAINING_INPUT does not exist: {path}")
        return path
    candidates = []
    for checkpoint in KINPUT.glob("**/qwen3_lora/checkpoint-*"):
        if (checkpoint / "trainer_state.json").exists():
            candidates.append(checkpoint.parent)
    if not candidates:
        return None
    candidates.sort(
        key=lambda root: max(
            [checkpoint_step(path) for path in root.glob("checkpoint-*")] or [-1]
        ),
        reverse=True,
    )
    return candidates[0]


def main() -> None:
    if not ROOT.exists():
        raise FileNotFoundError(f"Clone the repository to {ROOT} first")
    ART.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data_root = find_raw_root()
    recognizer_root = find_recognizer_root()
    print("DATA_ROOT =", data_root)
    print("BASELINE_RECOGNIZER_ROOT =", recognizer_root)

    baseline = find_first(
        recognizer_root,
        ["**/qwen3_lora/best_checkpoint", "**/best_checkpoint"],
    )
    if baseline is None:
        raise FileNotFoundError("Baseline best_checkpoint was not found")
    copy_dir(baseline, OUTPUT / "baseline_checkpoint")

    for name in (
        "config.json",
        "train_split.jsonl",
        "valid_split.jsonl",
        "valid_gt.csv",
        "scorable_stats.json",
    ):
        source = find_first(
            recognizer_root,
            [f"**/htr_artifacts/{name}", f"**/{name}"],
        )
        if source:
            copy_file(source, ART / name)

    resume_root = resume_root_from_env()
    if resume_root:
        print("RESUME_ROOT =", resume_root)
        for source in sorted(resume_root.glob("checkpoint-*"), key=checkpoint_step):
            if (source / "trainer_state.json").exists():
                copy_dir(source, OUTPUT / source.name)
    else:
        print("No previous V2 trainer checkpoint attached; starting a new V2 run.")

    config_path = ART / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )
    config.update(
        {
            "DATA_ROOT": str(data_root),
            "RUKOPYS_ROOT": str(data_root),
            "TRAIN_IMG": str(data_root / "train" / "images"),
            "TEST_IMG": str(data_root / "test" / "images"),
            "TRAIN_META": str(data_root / "train" / "metadata.jsonl"),
            "TEST_META": str(data_root / "test" / "metadata.jsonl"),
            "SILVER_META": str(data_root / "silver" / "metadata.jsonl"),
            "SILVER_IMG": str(data_root / "silver" / "images"),
            "ART": str(ART),
        }
    )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    required = [
        OUTPUT / "baseline_checkpoint" / "adapter_model.safetensors",
        ART / "train_split.jsonl",
        ART / "valid_split.jsonl",
        ART / "config.json",
    ]
    print("\nArtifact check")
    for path in required:
        print("OK" if path.exists() else "MISSING", path)
    if not all(path.exists() for path in required):
        raise RuntimeError("Gold training staging is incomplete")


if __name__ == "__main__":
    main()
