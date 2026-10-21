"""Evaluate one retained checkpoint on one frozen cohort under the locked CPU protocol.

The caller supplies data downloaded from ScienceDB and a checkpoint downloaded
from the associated GitHub Release. Source images and labels are never edited.
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from evaluate_locked import CONFIG, PACKAGE, check_v4, checksum, evaluate, write_yaml


def build_gold(gold_root, cohort, destination):
    manifest = json.loads((PACKAGE / "cohorts" / f"{cohort}.json").read_text(encoding="utf-8"))
    if manifest["count"] != len(manifest["images"]):
        raise ValueError(f"Cohort manifest count mismatch: {cohort}")
    image_dir, label_dir = destination / "images/test", destination / "labels/test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    counts = Counter()
    for entry in manifest["images"]:
        name = entry["image"]
        image = gold_root / "images" / name
        label = gold_root / "labels" / (Path(name).stem + ".txt")
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing Gold image/label: {name}")
        if checksum(image) != entry["md5"]:
            raise ValueError(f"Gold image MD5 differs from frozen manifest: {name}")
        shutil.copy2(image, image_dir / name)
        kept = []
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls = int(line.split()[0])
            if cls in (0, 1):
                kept.append(line)
                counts[cls] += 1
            elif cls != 2:
                raise ValueError(f"Unexpected Gold class {cls}: {name}")
        (label_dir / label.name).write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    if cohort == "Gold350" and counts != Counter({0: 1319, 1: 751}):
        raise ValueError(f"Gold350 class counts differ from paper: {counts}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--cohort", required=True, choices=("v4_test", "Gold350", "Gold402", "GoldB"))
    parser.add_argument("--v4-root", type=Path, help="Required for v4_test")
    parser.add_argument("--gold-root", type=Path, help="Required for Gold cohorts")
    parser.add_argument("--out", required=True, type=Path, help="New output directory")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    if args.cohort == "v4_test":
        if args.v4_root is None:
            parser.error("--v4-root is required for v4_test")
        check_v4(args.v4_root)
    elif args.gold_root is None:
        parser.error("--gold-root is required for Gold cohorts")

    args.out.mkdir(parents=True)
    cohort_root = args.out / "eval_data"
    if args.cohort == "v4_test":
        for kind in ("images", "labels"):
            target = cohort_root / kind / "test"
            target.parent.mkdir(parents=True, exist_ok=True)
            source = args.v4_root / "test" / kind
            try:
                target.symlink_to(source.resolve(), target_is_directory=True)
            except OSError:
                shutil.copytree(source, target)
        counts = None
    else:
        counts = build_gold(args.gold_root, args.cohort, cohort_root)

    data_yaml = args.out / "cohort.yaml"
    write_yaml(cohort_root, data_yaml)
    sys.path.insert(0, str(PACKAGE / "model_runtime"))
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(4)
    model = YOLO(str(args.weights))
    if len(model.names) != 2:
        raise ValueError(f"Expected two-class checkpoint, got {model.names}")
    result = {
        "checkpoint": args.weights.name,
        "cohort": args.cohort,
        "configuration": CONFIG,
        "two_class_boxes": dict(counts) if counts is not None else None,
        "mAP50": evaluate(model, data_yaml, args.cohort, args.out),
    }
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
