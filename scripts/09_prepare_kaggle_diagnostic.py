#!/usr/bin/env python3
"""Stage existing Phase 1/2 artifacts for the full validation diagnostic."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path("/kaggle/working/rukopys")
KINPUT = Path("/kaggle/input")
ART = ROOT / "local_working" / "htr_artifacts"


def has_raw_data(path: Path) -> bool:
    return (
        (path / "train" / "metadata.jsonl").exists()
        and (path / "train" / "images").exists()
    )


def first_root(candidates: list[Path], pattern: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    matches = list(KINPUT.glob(pattern))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No Kaggle input matched {pattern}")


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


def find_first(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = list(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def main() -> None:
    if not ROOT.exists():
        raise FileNotFoundError(
            f"{ROOT} does not exist. Clone the repository before running this script."
        )
    ART.mkdir(parents=True, exist_ok=True)
    data_root = find_raw_root()
    detector_root = first_root(
        [
            Path("/kaggle/input/datasets/bnthanh/htr-01-train-detector-output"),
            KINPUT / "htr-01-train-detector-output",
        ],
        "**/htr-01-train-detector-output",
    )
    recognizer_root = first_root(
        [
            Path("/kaggle/input/datasets/ngovietan/htr-02-train-recognizer"),
            KINPUT / "htr-02-train-recognizer",
        ],
        "**/htr-02-train-recognizer",
    )
    print("DATA_ROOT =", data_root)
    print("DET_ROOT  =", detector_root)
    print("REC_ROOT  =", recognizer_root)

    detector = find_first(
        detector_root,
        ["**/det_yolo/best.pt", "**/best.pt"],
    )
    if detector is None:
        raise FileNotFoundError("Detector best.pt was not found")
    copy_file(detector, ART / "det_yolo" / "best.pt")

    adapter = find_first(
        recognizer_root,
        ["**/qwen3_lora/best_checkpoint", "**/best_checkpoint"],
    )
    if adapter is None:
        checkpoints = sorted(
            recognizer_root.glob("**/checkpoint-*"),
            key=lambda path: int(path.name.split("-")[-1])
            if path.name.split("-")[-1].isdigit()
            else -1,
        )
        adapter = checkpoints[-1] if checkpoints else None
    if adapter is None:
        raise FileNotFoundError("Recognizer LoRA checkpoint was not found")
    copy_dir(adapter, ART / "qwen3_lora" / "best_checkpoint")

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

    if not (ART / "valid_split.jsonl").exists():
        raise FileNotFoundError("valid_split.jsonl was not found in recognizer output")

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
            "ART": str(ART),
        }
    )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    required = [
        ART / "det_yolo" / "best.pt",
        ART / "qwen3_lora" / "best_checkpoint" / "adapter_model.safetensors",
        ART / "config.json",
        ART / "valid_split.jsonl",
    ]
    print("\nArtifact check")
    for path in required:
        print("OK" if path.exists() else "MISSING", path)
    if not all(path.exists() for path in required):
        raise RuntimeError("Artifact staging is incomplete")


if __name__ == "__main__":
    main()
